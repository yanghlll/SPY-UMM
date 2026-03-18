"""
SPY-UMM Training Script.

Text-to-image post-training via spy game self-play using Show-o2.
Adapted from Show-o2's train_stage_one.py with spy game loop replacing
the standard dataloader-driven training.
"""

import os
import sys
import json
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

# Add Show-o2 to path for model imports
SHOWO2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Show-o', 'show-o2'))
# Remove any existing Show-o2 entries (e.g. from PYTHONPATH) to avoid duplicates
sys.path = [p for p in sys.path if os.path.abspath(p) != SHOWO2_DIR]
sys.path.insert(0, SHOWO2_DIR)

# --- Show-o2 imports (must come first, before SPY-UMM's models package) ---
from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.lr_schedulers import get_scheduler
from models.my_logging import set_verbosity_info, set_verbosity_error
from models.misc import prepare_gen_input, get_text_tokenizer, get_weight_type
from utils import (get_config, flatten_omega_conf, AverageMeter, denorm,
                   path_to_llm_name, _freeze_params, save_images_as_grid)
from transport import Sampler, create_transport

# --- Swap module cache: remove Show-o2's 'models' so SPY-UMM's can load ---
_showo2_models_cache = {k: sys.modules[k] for k in list(sys.modules.keys())
                        if k == 'models' or k.startswith('models.')}
for k in _showo2_models_cache:
    del sys.modules[k]
if SHOWO2_DIR in sys.path:
    sys.path.remove(SHOWO2_DIR)

# --- SPY-UMM imports (now 'models' resolves to SPY-UMM/models) ---
from data import SpyGameDataGenerator, VisionZeroDataAdapter
from models.showo2_spy_wrapper import Showo2SpyWrapper
from training.phase_controller import PhaseController
from training.reward_weighted_flow import RewardWeightedFlowMatchingLoss
from training.grpo_voting import VotingGRPO, generate_and_score_votes
from training.rewards import vote_accuracy_reward, vote_format_reward
from training.flow_grpo import FlowGRPO, FlowGRPOConfig

# Restore Show-o2's models in module cache (so already-imported refs stay valid)
sys.modules.update(_showo2_models_cache)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = get_logger(__name__, log_level="INFO")


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
    )

    # For spy game, batch_size = 1 game per step
    total_batch_size_per_gpu = config.training.batch_size
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = total_batch_size_per_gpu

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # WandB init
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS                #
    #########################
    logger.info("Loading models...")
    weight_type = get_weight_type(config)

    # VAE
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(
            vae_pth=config.model.vae_model.pretrained_model_path,
            dtype=weight_type,
            device=accelerator.device
        )
    else:
        raise NotImplementedError(f"VAE type {config.model.vae_model.type} not supported")

    # Tokenizer
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path]
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    # Show-o2 model
    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path,
            use_safetensors=False
        ).to(accelerator.device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(accelerator.device)

    # Freeze specified parameters
    # 这里冻住了什么？冻住了哪些东西？
    _freeze_params(model, config.model.showo.frozen_params)

    # Time embedding adjustment
    # mmu和t2i都要加吗？
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
        config.dataset.preprocessing.num_mixed_modal_tokens += 1

    # Gradient checkpointing
    if config.model.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    logger.info(f"Model loaded. Frozen params: {config.model.showo.frozen_params}")

    #########################
    # TRANSPORT & SAMPLER   #
    #########################
    # Transport → 生成图像潜在空间的 diffusion / flow 模型。
    # Sampler → 用于采样生成图像
    
    preproc_config = config.dataset.preprocessing

    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=preproc_config.num_t2i_image_tokens,
    )
    sampler = Sampler(transport)

    #########################
    # SPY WRAPPER           #
    #########################
    spy_wrapper = Showo2SpyWrapper(
        model=model,
        vae_model=vae_model,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        transport=transport,
        sampler=sampler,
        config=config,
    )

    #########################
    # OPTIMIZER             #
    #########################
    optimizer_config = config.optimizer.params

    # Separate parameter groups with different learning rates
    ve_params = []      # Visual encoder (generation path)
    proj_params = []    # Projection / diffusion heads
    showo_params = []   # LLM backbone

    ve_names = ['image_embedder_gen', 'fusion_proj']
    proj_names = ['diff_proj', 'diffusion_head_a', 'diffusion_head_b',
                  'time_embed', 'time_embed_proj']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(n in name for n in ve_names):
            ve_params.append(param)
        elif any(n in name for n in proj_names):
            proj_params.append(param)
        else:
            showo_params.append(param)

    optimizer = AdamW(
        [
            {"params": ve_params, "lr": optimizer_config.learning_rate_ve},
            {"params": proj_params, "lr": optimizer_config.learning_rate_proj},
            {"params": showo_params, "lr": optimizer_config.learning_rate_showo},
        ],
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

    # Log parameter counts
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_trainable:,}")
    logger.info(f"  VE params: {sum(p.numel() for p in ve_params):,}")
    logger.info(f"  Proj params: {sum(p.numel() for p in proj_params):,}")
    logger.info(f"  Showo params: {sum(p.numel() for p in showo_params):,}")

    #########################
    # LR SCHEDULER          #
    #########################
    warmup_steps = config.lr_scheduler.params.warmup_steps
    if warmup_steps is None:
        warmup_ratio = config.lr_scheduler.params.get("warmup_ratio", 0.05)
        warmup_steps = int(warmup_ratio * config.training.max_train_steps)

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=warmup_steps,
    )

    #########################
    # GAME INFRASTRUCTURE   #
    #########################
    game_config = config.game
    game_generator = SpyGameDataGenerator(
        num_players=game_config.num_players,
        num_objects_min=game_config.num_objects_min,
        num_objects_max=game_config.num_objects_max,
        num_to_modify=game_config.num_objects_to_modify,
    )

    phase_controller = PhaseController(
        mode=config.training.phase_mode,
        cycle_length=config.training.cycle_length,
    )

    rw_flow_loss = RewardWeightedFlowMatchingLoss(
        reward_baseline_ema=config.training.reward_baseline_ema,
        reward_clamp_min=config.training.reward_clamp_min,
    ).to(accelerator.device)

    voting_grpo = VotingGRPO(
        beta=config.training.beta,
        epsilon=config.training.epsilon,
    )

    # Flow-GRPO for image generation phase
    flow_grpo_config = FlowGRPOConfig(
        sde_noise_scale=config.training.get("sde_noise_scale", 0.7),
        group_size=config.training.get("flow_grpo_group_size", 4),
        epsilon=config.training.epsilon,
        beta=config.training.get("flow_grpo_beta", 0.01),
        num_train_steps=config.training.get("flow_grpo_train_steps", 10),
        num_inference_steps=config.transport.num_inference_steps,
        fast_mode=config.training.get("flow_grpo_fast_mode", False),
        sde_window_size=config.training.get("sde_window_size", 2),
        sde_window_range=tuple(config.training.get("sde_window_range", [0.1, 0.9])),
        do_shift=config.transport.do_shift,
        time_shifting_factor=config.transport.time_shifting_factor,
    )
    flow_grpo = FlowGRPO(flow_grpo_config).to(accelerator.device)
    use_flow_grpo = config.training.get("use_flow_grpo", False)
    if use_flow_grpo:
        logger.info(f"Flow-GRPO enabled: group_size={flow_grpo_config.group_size}, "
                    f"train_steps={flow_grpo_config.num_train_steps}, "
                    f"fast_mode={flow_grpo_config.fast_mode}")

    # Reference model for KL penalty (frozen copy)
    ref_model = None
    if use_flow_grpo and flow_grpo_config.beta > 0:
        import copy
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        logger.info("Reference model created for Flow-GRPO KL penalty")

    # Vision-Zero data adapter
    vz_images_dir = config.dataset.get(
        "vision_zero_images_dir",
        "/adialab/usr/shadabk/MedUMM/data/Vision-Zero/clevr-dataset/output/replacement_images"
    )
    vz_adapter = None
    if os.path.isdir(vz_images_dir):
        vz_adapter = VisionZeroDataAdapter(
            images_dir=vz_images_dir,
            vae_model=vae_model,
            image_size=preproc_config.resolution,
            device=accelerator.device,
            dtype=weight_type,
        )
        logger.info(f"Vision-Zero adapter loaded: {len(vz_adapter)} image pairs")
    else:
        logger.warning(f"Vision-Zero images not found at {vz_images_dir}, "
                       "using self-play targets only")

    #########################
    # CHECKPOINT RESUME     #
    #########################
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        output_dir = config.experiment.output_dir
        dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if dirs else None
        if path is not None:
            path = os.path.join(output_dir, path)
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // config.game.epoch_size

            accelerator.print(f"Resuming from checkpoint {path}/unwrapped_model/pytorch_model.bin")
            state_dict = torch.load(
                f'{path}/unwrapped_model/pytorch_model.bin',
                map_location="cpu"
            )
            if config.model.showo.params_not_load is not None:
                keys_to_delete = [
                    k for k in state_dict
                    if any(n in k for n in config.model.showo.params_not_load)
                ]
                for k in keys_to_delete:
                    del state_dict[k]
            model.load_state_dict(
                state_dict,
                strict=(config.model.showo.params_not_load is None)
            )
            del state_dict

    #########################
    # ACCELERATOR PREPARE   #
    #########################
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # Update spy_wrapper to use the unwrapped model on the correct device
    spy_wrapper.model = accelerator.unwrap_model(model)

    #########################
    # TRAINING LOOP         #
    #########################
    logger.info("***** Running SPY-UMM Training *****")
    logger.info(f"  Num players per game = {game_config.num_players}")
    logger.info(f"  Epoch size = {game_config.epoch_size}")
    logger.info(f"  Max training steps = {config.training.max_train_steps}")
    logger.info(f"  Phase mode = {config.training.phase_mode}")
    logger.info(f"  Gradient accumulation steps = {config.training.gradient_accumulation_steps}")

    num_players = game_config.num_players
    num_train_epochs = math.ceil(config.training.max_train_steps / config.game.epoch_size)

    batch_time_m = AverageMeter()
    end = time.time()

    # Running stats
    spy_caught_count = 0
    total_games_played = 0

    for epoch in range(first_epoch, num_train_epochs):
        model.train()

        for sample_idx in range(config.game.epoch_size):
            if global_step >= config.training.max_train_steps:
                break

            # =========================================
            # Step A: Generate game data
            # =========================================
            game_data = game_generator.generate_game(epoch, sample_idx)

            # =========================================
            # Step B: Determine active phase
            # =========================================
            train_gen = phase_controller.should_train_generation(global_step)
            train_vote = phase_controller.should_train_voting(global_step)

            gen_loss = torch.tensor(0.0, device=accelerator.device)
            vote_loss = torch.tensor(0.0, device=accelerator.device)
            player_votes = [None] * num_players
            flow_grpo_metrics = {}  # Initialize for logging scope

            # =========================================
            # Step C: Generate images (no_grad)
            # =========================================
            prompts = [
                game_generator.format_generation_prompt_simple(game_data, pid)
                for pid in range(1, num_players + 1)
            ]

            with torch.no_grad():
                gen_result = spy_wrapper.generate_images(
                    prompts,
                    guidance_scale=config.transport.guidance_scale,
                    num_steps=config.transport.num_inference_steps,
                )
                generated_latents = gen_result['latents']  # [num_players, C, H, W]

            # =========================================
            # Step D-F: Voting phase (GRPO)
            # =========================================
            if train_vote:
                voting_prompt = game_generator.format_voting_prompt(game_data)

                # D: Generate G vote completions and score them
                vote_result = generate_and_score_votes(
                    spy_wrapper,
                    [generated_latents[i:i+1] for i in range(num_players)],
                    voting_prompt,
                    correct_spy=game_data['spy_player'],
                    num_generations=config.training.num_generations,
                    max_tokens=config.training.max_vote_tokens,
                    temperature=config.training.vote_temperature,
                )

                # Extract votes for game reward calculation
                for resp in vote_result['responses']:
                    vote_info = game_generator.extract_vote(resp)
                    player_votes.append(vote_info)
                # Only keep first num_players votes for game outcome
                player_votes = player_votes[:num_players]

                # E: Compute old log probs (no_grad) and current log probs (with grad)
                advantages = voting_grpo.compute_advantages(
                    vote_result['total_rewards']
                )

                old_logprobs_list = []
                current_logprobs_list = []
                completion_ids_list = []
                completion_masks_list = []

                for resp in vote_result['responses']:
                    comp_ids = text_tokenizer(
                        resp, add_special_tokens=False, return_tensors='pt'
                    )['input_ids'].to(accelerator.device)

                    # Truncate if too long
                    if comp_ids.shape[1] > config.training.max_vote_tokens:
                        comp_ids = comp_ids[:, :config.training.max_vote_tokens]

                    completion_ids_list.append(comp_ids)
                    completion_masks_list.append(
                        torch.ones_like(comp_ids, dtype=torch.float)
                    )

                    # Old log probs (no gradient)
                    with torch.no_grad():
                        old_lp = spy_wrapper.compute_voting_logprobs(
                            generated_latents[0:1],
                            voting_prompt,
                            comp_ids,
                        )
                    old_logprobs_list.append(old_lp)

                # F: Compute current log probs (with gradient)
                for comp_ids in completion_ids_list:
                    cur_lp = spy_wrapper.compute_voting_logprobs(
                        generated_latents[0:1],
                        voting_prompt,
                        comp_ids,
                    )
                    current_logprobs_list.append(cur_lp)

                # Pad to uniform length and stack
                max_len = max(c.shape[1] for c in completion_ids_list)
                padded_current = []
                padded_old = []
                padded_masks = []

                for cur_lp, old_lp, mask in zip(
                    current_logprobs_list, old_logprobs_list, completion_masks_list
                ):
                    pad_len = max_len - cur_lp.shape[1]
                    if pad_len > 0:
                        cur_lp = F.pad(cur_lp, (0, pad_len), value=0.0)
                        old_lp = F.pad(old_lp, (0, pad_len), value=0.0)
                        mask = F.pad(mask, (0, pad_len), value=0.0)
                    padded_current.append(cur_lp)
                    padded_old.append(old_lp)
                    padded_masks.append(mask)

                current_logprobs = torch.cat(padded_current, dim=0)  # [G, L]
                old_logprobs = torch.cat(padded_old, dim=0)
                completion_masks = torch.cat(padded_masks, dim=0)

                grpo_result = voting_grpo.compute_loss(
                    current_logprobs,
                    old_logprobs,
                    advantages.to(accelerator.device),
                    completion_masks,
                )
                vote_loss = grpo_result['loss']

            # =========================================
            # Step G-I: Generation phase training
            # =========================================
            if train_gen:
                # G: Compute game outcome from votes
                game_outcome = game_generator.calculate_game_rewards(
                    game_data, player_votes
                )
                gen_rewards = game_generator.compute_generation_rewards(game_outcome)
                reward_tensor = torch.tensor(
                    gen_rewards, dtype=torch.float32, device=accelerator.device
                )

                is_spy = torch.zeros(num_players, dtype=torch.bool,
                                     device=accelerator.device)
                is_spy[game_data['spy_player'] - 1] = True

                # Track game outcome
                if game_outcome['spy_caught']:
                    spy_caught_count += 1
                total_games_played += 1

                if use_flow_grpo:
                    # ---- Flow-GRPO: proper policy gradient on flow head ----
                    # Generate G images per player via SDE, compute advantages
                    # ACROSS all players (spy vs civilian rewards create variance).
                    G = flow_grpo_config.group_size
                    T_train = flow_grpo_config.num_train_steps

                    # Phase 1: Generate all SDE trajectories (no_grad)
                    all_trajectories = []   # [num_players][G] trajectories
                    all_old_logprobs = []   # [num_players][G] old log probs
                    all_vfn_kwargs = []     # [num_players][G] velocity fn kwargs
                    flat_rewards = []       # [num_players * G] rewards

                    for pid in range(num_players):
                        prompt = prompts[pid]
                        player_trajectories = []
                        player_old_lps = []
                        player_vfn_kwargs = []

                        for g in range(G):
                            with torch.no_grad():
                                sde_result = spy_wrapper.generate_images_sde(
                                    [prompt], flow_grpo,
                                    num_steps=T_train,
                                    guidance_scale=0.0,  # No CFG during training
                                )
                                player_trajectories.append(sde_result['trajectory'])
                                player_vfn_kwargs.append(sde_result['velocity_fn_kwargs'])

                                # Select fast steps if enabled
                                if flow_grpo_config.fast_mode:
                                    step_indices = flow_grpo.select_fast_steps(
                                        T_train, accelerator.device
                                    )
                                else:
                                    step_indices = None

                                # Compute old log probs
                                old_lp = spy_wrapper.compute_flow_grpo_logprobs(
                                    flow_grpo,
                                    sde_result['trajectory'],
                                    sde_result['velocity_fn_kwargs'],
                                    step_indices=step_indices,
                                )
                                player_old_lps.append(old_lp)

                            # Each generation gets the player's game reward
                            flat_rewards.append(gen_rewards[pid])

                        all_trajectories.append(player_trajectories)
                        all_old_logprobs.append(torch.cat(player_old_lps, dim=0))
                        all_vfn_kwargs.append(player_vfn_kwargs)

                    # Phase 2: Compute group-relative advantages across ALL players
                    # This gives variance: spy reward != civilian reward
                    all_rewards_tensor = torch.tensor(
                        flat_rewards, dtype=torch.float32,
                        device=accelerator.device,
                    )  # [num_players * G]
                    all_advantages = FlowGRPO.compute_advantages(all_rewards_tensor)

                    # Phase 3: Recompute log probs with gradient and GRPO loss
                    flow_grpo_loss = torch.tensor(0.0, device=accelerator.device)

                    flat_idx = 0
                    all_current_lps = []
                    all_old_lps_flat = []
                    all_kl_flat = []

                    for pid in range(num_players):
                        for g in range(G):
                            traj = all_trajectories[pid][g]
                            vfn_kwargs = all_vfn_kwargs[pid][g]

                            if flow_grpo_config.fast_mode:
                                step_indices = flow_grpo.select_fast_steps(
                                    T_train, accelerator.device
                                )
                            else:
                                step_indices = None

                            cur_lp = spy_wrapper.compute_flow_grpo_logprobs(
                                flow_grpo, traj, vfn_kwargs,
                                step_indices=step_indices,
                            )
                            all_current_lps.append(cur_lp)
                            all_old_lps_flat.append(
                                all_old_logprobs[pid][g:g+1]
                            )

                            # KL penalty with reference model
                            if ref_model is not None and flow_grpo_config.beta > 0:
                                # Create ref velocity fn using ref_model
                                orig_model = spy_wrapper.model
                                spy_wrapper.model = ref_model
                                ref_vel_fn = spy_wrapper.make_velocity_fn(**vfn_kwargs)
                                spy_wrapper.model = orig_model

                                vel_fn = spy_wrapper.make_velocity_fn(**vfn_kwargs)
                                kl = flow_grpo.compute_trajectory_kl(
                                    vel_fn, ref_vel_fn, traj,
                                    step_indices=step_indices,
                                )
                                all_kl_flat.append(kl)

                            flat_idx += 1

                    current_logprobs = torch.cat(all_current_lps, dim=0)  # [N*G]
                    old_logprobs_flat = torch.cat(all_old_lps_flat, dim=0)  # [N*G]
                    kl_tensor = torch.cat(all_kl_flat, dim=0) if all_kl_flat else None

                    grpo_result = flow_grpo.compute_grpo_loss(
                        current_logprobs,
                        old_logprobs_flat.detach(),
                        all_advantages,
                        kl_tensor,
                    )
                    gen_loss = grpo_result['loss']
                    flow_grpo_metrics = grpo_result['metrics']

                else:
                    # ---- Fallback: reward-weighted flow matching ----
                    # H: Get target latents
                    if vz_adapter is not None:
                        target_latents = vz_adapter.get_target_latents(
                            game_data, accelerator.device
                        )
                    else:
                        # Self-play: use generated latents as targets (detached)
                        target_latents = generated_latents.detach()

                    # I: Compute reward-weighted flow matching loss
                    flow_result = spy_wrapper.compute_flow_loss(
                        prompts,
                        target_latents=target_latents,
                        reward_weights=rw_flow_loss.compute_weights(
                            reward_tensor, is_spy
                        ),
                    )
                    gen_loss = flow_result['loss']

            # =========================================
            # Step J: Combined loss and backward
            # =========================================
            total_loss = (
                config.training.gen_loss_coeff * gen_loss +
                config.training.vote_loss_coeff * vote_loss
            )

            accelerator.backward(
                total_loss / config.training.gradient_accumulation_steps
            )

            # Gradient clipping and optimizer step
            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), config.training.max_grad_norm
                )

            if (global_step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # =========================================
            # Step K: Logging & Checkpointing
            # =========================================
            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()

                # Logging
                if (global_step + 1) % config.experiment.log_every == 0:
                    lr = [group["lr"] for group in optimizer.param_groups]
                    phase_info = phase_controller.log_phase_info(global_step)
                    spy_rate = (spy_caught_count / max(total_games_played, 1))

                    logs = {
                        "step_loss_gen": gen_loss.item(),
                        "step_loss_vote": vote_loss.item(),
                        "step_loss_total": total_loss.item(),
                        "lr_ve": lr[0],
                        "lr_proj": lr[1],
                        "lr_showo": lr[2],
                        "spy_detection_rate": spy_rate,
                        "batch_time": batch_time_m.val,
                        "baseline_spy": rw_flow_loss.baseline_spy.item(),
                        "baseline_civ": rw_flow_loss.baseline_civ.item(),
                    }

                    if train_vote and 'metrics' in grpo_result:
                        logs["vote_clip_fraction"] = grpo_result['metrics']['clip_fraction']
                        logs["vote_mean_advantage"] = grpo_result['metrics']['mean_advantage']
                        logs["mean_vote_accuracy"] = (
                            sum(vote_result['accuracy_rewards'])
                            / len(vote_result['accuracy_rewards'])
                        )

                    if train_gen and use_flow_grpo and flow_grpo_metrics:
                        logs["flow_grpo_clip_fraction"] = flow_grpo_metrics.get('clip_fraction', 0)
                        logs["flow_grpo_mean_ratio"] = flow_grpo_metrics.get('mean_ratio', 1)
                        logs["flow_grpo_approx_kl"] = flow_grpo_metrics.get('approx_kl', 0)
                        logs["flow_grpo_mean_logprob"] = flow_grpo_metrics.get('mean_logprob', 0)

                    accelerator.log(logs, step=global_step + 1)
                    logger.info(
                        f"Epoch: {epoch} "
                        f"Step: {global_step + 1} "
                        f"Loss_Gen: {gen_loss.item():.4f} "
                        f"Loss_Vote: {vote_loss.item():.4f} "
                        f"SpyRate: {spy_rate:.2%} "
                        f"{phase_info} "
                        f"LR_ve: {lr[0]:.6f} "
                        f"LR_proj: {lr[1]:.6f} "
                        f"LR_showo: {lr[2]:.6f}"
                    )

                    batch_time_m.reset()

                # Checkpointing
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                # Sample image generation
                if (
                    (global_step + 1) % config.experiment.generate_every == 0
                    and accelerator.is_main_process
                ):
                    generate_sample_images(
                        spy_wrapper, game_generator, config,
                        global_step + 1, accelerator.device, weight_type,
                    )

                global_step += 1

        if global_step >= config.training.max_train_steps:
            break

    # =========================================
    # Final cleanup
    # =========================================
    accelerator.wait_for_everyone()
    save_checkpoint(model, config, accelerator, "final")

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            config.experiment.output_dir, safe_serialization=False
        )

    accelerator.end_training()
    logger.info("Training complete!")
    logger.info(f"Final spy detection rate: {spy_caught_count}/{total_games_played} "
                f"= {spy_caught_count / max(total_games_played, 1):.2%}")


@torch.no_grad()
def generate_sample_images(spy_wrapper, game_generator, config,
                           global_step, device, weight_type):
    """Generate sample images from a spy game for visualization."""
    logger.info("Generating sample images...")
    spy_wrapper.model.eval()

    game_data = game_generator.generate_game(epoch=9999, sample_idx=global_step)
    prompts = [
        game_generator.format_generation_prompt_simple(game_data, pid)
        for pid in range(1, game_data['num_players'] + 1)
    ]

    gen_result = spy_wrapper.generate_images(
        prompts,
        guidance_scale=config.transport.guidance_scale,
        num_steps=config.transport.num_inference_steps,
    )

    pil_images = gen_result['images']

    # Create captions
    captions = []
    for pid in range(1, game_data['num_players'] + 1):
        role = "SPY" if pid == game_data['spy_player'] else "CIV"
        captions.append(f"P{pid} ({role})")

    # Log to wandb
    wandb_images = [
        wandb.Image(img, caption=cap)
        for img, cap in zip(pil_images, captions)
    ]
    wandb.log({"spy_game_images": wandb_images}, step=global_step)

    # Save grid
    output_dir = config.experiment.output_dir
    grid_dir = os.path.join(output_dir, "sample_images")
    os.makedirs(grid_dir, exist_ok=True)
    grid = save_images_as_grid(
        pil_images, f"step_{global_step}", grid_dir,
        grid_size=(2, 2) if len(pil_images) == 4 else (1, len(pil_images))
    )

    spy_wrapper.model.train()


def save_checkpoint(model, config, accelerator, global_step):
    """Save model checkpoint."""
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            for removing in checkpoints[:num_to_remove]:
                shutil.rmtree(os.path.join(output_dir, removing))

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False,
        )
        json.dump(
            {"global_step": global_step},
            (save_path / "metadata.json").open("w+")
        )
        logger.info(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
