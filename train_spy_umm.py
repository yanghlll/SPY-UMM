"""
SPY-UMM Training Script — Flow-GRPO only.

Text-to-image post-training via spy game self-play using Show-o2.
Uses Flow-GRPO (ODE→SDE conversion + PPO-clip policy gradient).
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
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

# Add Show-o2 to path for model imports
SHOWO2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Show-o', 'show-o2'))
sys.path = [p for p in sys.path if os.path.abspath(p) != SHOWO2_DIR]
sys.path.insert(0, SHOWO2_DIR)

from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.lr_schedulers import get_scheduler
from models.my_logging import set_verbosity_info, set_verbosity_error
from models.misc import prepare_gen_input, get_text_tokenizer, get_weight_type
from utils import (get_config, flatten_omega_conf, AverageMeter, denorm,
                   path_to_llm_name, _freeze_params, save_images_as_grid)
from transport import Sampler, create_transport

# Swap module cache for SPY-UMM imports
_showo2_models_cache = {k: sys.modules[k] for k in list(sys.modules.keys())
                        if k == 'models' or k.startswith('models.')}
for k in _showo2_models_cache:
    del sys.modules[k]
if SHOWO2_DIR in sys.path:
    sys.path.remove(SHOWO2_DIR)

from data import SpyGameDataGenerator
from models.showo2_spy_wrapper import Showo2SpyWrapper
from training.flow_grpo import FlowGRPO, FlowGRPOConfig

sys.modules.update(_showo2_models_cache)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
logger = get_logger(__name__, log_level="INFO")


def main():
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

    total_batch_size_per_gpu = config.training.batch_size
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = total_batch_size_per_gpu

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id
        wandb_init_kwargs = dict(
            name=config.experiment.name, id=run_id, resume=resume_wandb_run,
            entity=config.wandb.get("entity", None), config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)
        accelerator.init_trackers(
            config.experiment.project, config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        OmegaConf.save(config, Path(config.experiment.output_dir) / "config.yaml")

    if config.training.seed is not None:
        set_seed(config.training.seed)

    # ==================== MODELS ====================
    logger.info("Loading models...")
    weight_type = get_weight_type(config)

    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(
            vae_pth=config.model.vae_model.pretrained_model_path,
            dtype=weight_type, device=accelerator.device,
        )
    else:
        raise NotImplementedError(f"VAE type {config.model.vae_model.type}")

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path, add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path, use_safetensors=False
        ).to(accelerator.device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(accelerator.device)

    _freeze_params(model, config.model.showo.frozen_params)

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
        config.dataset.preprocessing.num_mixed_modal_tokens += 1

    if config.model.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    logger.info(f"Model loaded. Frozen: {config.model.showo.frozen_params}")

    # ==================== TRANSPORT ====================
    preproc_config = config.dataset.preprocessing
    transport = create_transport(
        path_type=config.transport.path_type, prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight, train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps, snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift, seq_len=preproc_config.num_t2i_image_tokens,
    )
    sampler = Sampler(transport)

    # ==================== SPY WRAPPER ====================
    spy_wrapper = Showo2SpyWrapper(
        model=model, vae_model=vae_model, text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids, transport=transport,
        sampler=sampler, config=config,
    )

    # ==================== OPTIMIZER ====================
    optimizer_config = config.optimizer.params
    ve_params, proj_params, showo_params = [], [], []
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
        [{"params": ve_params, "lr": optimizer_config.learning_rate_ve},
         {"params": proj_params, "lr": optimizer_config.learning_rate_proj},
         {"params": showo_params, "lr": optimizer_config.learning_rate_showo}],
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay, eps=optimizer_config.epsilon,
    )

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {total_trainable:,}")

    # ==================== LR SCHEDULER ====================
    # Total optimizer steps = max_train_steps * num_inner_epochs
    num_inner_epochs_cfg = config.training.get("num_inner_epochs", 2)
    total_optimizer_steps = config.training.max_train_steps * num_inner_epochs_cfg
    warmup_steps = config.lr_scheduler.params.warmup_steps
    if warmup_steps is None:
        warmup_steps = int(config.lr_scheduler.params.get("warmup_ratio", 0.05)
                           * total_optimizer_steps)
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler, optimizer=optimizer,
        num_training_steps=total_optimizer_steps,
        num_warmup_steps=warmup_steps,
    )

    # ==================== FLOW-GRPO ====================
    flow_grpo_config = FlowGRPOConfig(
        sde_noise_scale=config.training.get("sde_noise_scale", 0.7),
        group_size=config.training.get("flow_grpo_group_size", 4),
        clip_range=config.training.get("clip_range", 0.2),
        beta=config.training.get("flow_grpo_beta", 0.0),
        adv_clip_max=config.training.get("adv_clip_max", 5.0),
        num_train_steps=config.training.get("flow_grpo_train_steps", 20),
        num_inference_steps=config.transport.num_inference_steps,
        sde_window_size=config.training.get("sde_window_size", 3),
        sde_window_range=tuple(config.training.get("sde_window_range", [0, -1])),
        do_shift=config.transport.do_shift,
        time_shifting_factor=config.transport.time_shifting_factor,
    )
    flow_grpo = FlowGRPO(flow_grpo_config).to(accelerator.device)
    logger.info(f"Flow-GRPO: G={flow_grpo_config.group_size}, "
                f"T={flow_grpo_config.num_train_steps}, "
                f"window={flow_grpo_config.sde_window_size}")

    # ==================== GAME SETUP ====================
    game_config = config.game
    game_generator = SpyGameDataGenerator(
        num_players=game_config.num_players,
        num_objects_min=game_config.num_objects_min,
        num_objects_max=game_config.num_objects_max,
        num_to_modify=game_config.num_objects_to_modify,
    )
    num_players = game_config.num_players

    # ==================== CHECKPOINT RESUME ====================
    global_step = 0
    first_epoch = 0
    if config.experiment.resume_from_checkpoint:
        output_dir = config.experiment.output_dir
        dirs = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint")],
                      key=lambda x: int(x.split("-")[1]))
        if dirs:
            path = os.path.join(output_dir, dirs[-1])
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // config.game.epoch_size
            state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
            if config.model.showo.params_not_load is not None:
                for k in [k for k in state_dict if any(n in k for n in config.model.showo.params_not_load)]:
                    del state_dict[k]
            model.load_state_dict(state_dict, strict=(config.model.showo.params_not_load is None))
            del state_dict
            logger.info(f"Resumed from step {global_step}")

    # ==================== ACCELERATOR PREPARE ====================
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    spy_wrapper.model = accelerator.unwrap_model(model)

    # ==================== TRAINING LOOP ====================
    logger.info("***** Running SPY-UMM Flow-GRPO Training *****")
    logger.info(f"  Players={num_players}, G={flow_grpo_config.group_size}, "
                f"T={flow_grpo_config.num_train_steps}, "
                f"inner_epochs={config.training.get('num_inner_epochs', 2)}")

    num_train_epochs = math.ceil(config.training.max_train_steps / config.game.epoch_size)
    batch_time_m = AverageMeter()
    end = time.time()
    spy_caught_count = 0
    total_games_played = 0

    G = flow_grpo_config.group_size
    T_train = flow_grpo_config.num_train_steps
    sde_guidance = config.transport.guidance_scale
    max_vote_tokens = config.training.get("max_vote_tokens", 128)
    num_inner_epochs = config.training.get("num_inner_epochs", 2)

    for epoch in range(first_epoch, num_train_epochs):
        model.train()

        for sample_idx in range(config.game.epoch_size):
            if global_step >= config.training.max_train_steps:
                break

            # Step A: Generate game data
            game_data = game_generator.generate_game(epoch, sample_idx)
            prompts = [
                game_generator.format_generation_prompt_simple(game_data, pid)
                for pid in range(1, num_players + 1)
            ]

            # =============================================
            # Phase 1: Generate G games via batched SDE
            # =============================================
            all_game_trajs = []
            all_game_vfn_kwargs = []
            all_game_rewards = []
            sde_window = flow_grpo.select_sde_window(T_train)

            for g in range(G):
                with torch.no_grad():
                    inputs = spy_wrapper.prepare_flow_grpo_inputs(
                        prompts, guidance_scale=sde_guidance
                    )
                    vfn_kwargs = inputs['velocity_fn_kwargs']
                    velocity_fn = spy_wrapper.make_velocity_fn(**vfn_kwargs)

                    traj = flow_grpo.generate_trajectory(
                        velocity_fn, inputs['z_init'],
                        num_steps=T_train, sde_window=sde_window,
                    )

                    # Voting: each player sees ALL images, votes with role-specific prompt
                    sde_latents = traj['final']
                    all_image_latents = [sde_latents[pid:pid+1] for pid in range(num_players)]

                    game_votes = []
                    for pid in range(1, num_players + 1):
                        vote_prompt = game_generator.format_voting_prompt(
                            game_data, player_id=pid
                        )
                        vote_resp = spy_wrapper.judge_vote(
                            all_image_latents, vote_prompt,
                            max_new_tokens=max_vote_tokens,
                        )
                        vote_info = game_generator.extract_vote(vote_resp)
                        game_votes.append(vote_info)

                    game_outcome = game_generator.calculate_game_rewards(game_data, game_votes)
                    gen_rewards_g = game_generator.compute_generation_rewards(game_outcome)

                    if game_outcome['spy_caught']:
                        spy_caught_count += 1
                    total_games_played += 1

                all_game_trajs.append(traj)
                all_game_vfn_kwargs.append(vfn_kwargs)
                all_game_rewards.append(gen_rewards_g)

            # =============================================
            # Phase 2: Group-relative advantages
            # =============================================
            flat_rewards = []
            for rw in all_game_rewards:
                flat_rewards.extend(rw)
            all_advantages = FlowGRPO.compute_advantages(
                torch.tensor(flat_rewards, dtype=torch.float32, device=accelerator.device)
            )

            # =============================================
            # Phase 3: Batched per-timestep backward + inner epochs
            # =============================================
            grpo_metrics_accum = []
            n_sde_window = len(all_game_trajs[0]['sde_steps'])

            # Tile vfn_kwargs for G games
            base_vfn = all_game_vfn_kwargs[0]
            if G > 1:
                tiled_kwargs = {}
                for k, v in base_vfn.items():
                    if isinstance(v, torch.Tensor):
                        tiled_kwargs[k] = v.repeat(G, *([1] * (v.dim() - 1)))
                    else:
                        tiled_kwargs[k] = v
            else:
                tiled_kwargs = base_vfn

            # Pre-stack SDE step data
            batched_sde_steps = []
            for step_idx in range(n_sde_window):
                batched_sde_steps.append({
                    'x_t': torch.cat([all_game_trajs[g]['sde_steps'][step_idx]['x_t']
                                      for g in range(G)], dim=0),
                    'x_next': torch.cat([all_game_trajs[g]['sde_steps'][step_idx]['x_next']
                                         for g in range(G)], dim=0),
                    'old_logprob': torch.cat([all_game_trajs[g]['sde_steps'][step_idx]['old_logprob']
                                              for g in range(G)], dim=0),
                    't': all_game_trajs[0]['sde_steps'][step_idx]['t'],
                    'dt': all_game_trajs[0]['sde_steps'][step_idx]['dt'],
                })

            for inner_epoch in range(num_inner_epochs):
                velocity_fn = spy_wrapper.make_velocity_fn(**tiled_kwargs)

                for step_idx in range(n_sde_window):
                    step_result = flow_grpo.compute_per_step_loss(
                        velocity_fn, batched_sde_steps[step_idx],
                        all_advantages, ref_model_fn=None,
                    )
                    step_loss = step_result['loss'] * config.training.gen_loss_coeff
                    accelerator.backward(step_loss / (num_inner_epochs * n_sde_window))
                    grpo_metrics_accum.append(step_result['metrics'])

                # Optimizer step per inner epoch
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.grad is not None],
                        config.training.max_grad_norm or 1e10,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Aggregate metrics
            flow_grpo_metrics = {}
            if grpo_metrics_accum:
                flow_grpo_metrics = {
                    k: sum(m[k] for m in grpo_metrics_accum) / len(grpo_metrics_accum)
                    for k in grpo_metrics_accum[0]
                }
                flow_grpo_metrics['abs_policy_loss'] = sum(
                    abs(m['policy_loss']) for m in grpo_metrics_accum
                ) / len(grpo_metrics_accum)
                flow_grpo_metrics['mean_reward'] = sum(flat_rewards) / len(flat_rewards)

            generated_latents = all_game_trajs[0]['final']

            # =============================================
            # Logging & Checkpointing
            # =============================================
            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()

                if (global_step + 1) % config.experiment.log_every == 0:
                    lr = [g["lr"] for g in optimizer.param_groups]
                    spy_rate = spy_caught_count / max(total_games_played, 1)

                    logs = {
                        "abs_policy_loss": flow_grpo_metrics.get('abs_policy_loss', 0),
                        "policy_loss": flow_grpo_metrics.get('policy_loss', 0),
                        "clip_fraction": flow_grpo_metrics.get('clip_fraction', 0),
                        "mean_ratio": flow_grpo_metrics.get('mean_ratio', 1),
                        "approx_kl": flow_grpo_metrics.get('approx_kl', 0),
                        "mean_reward": flow_grpo_metrics.get('mean_reward', 0),
                        "spy_detection_rate": spy_rate,
                        "lr": lr[1],
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)
                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss: {flow_grpo_metrics.get('abs_policy_loss', 0):.4f} "
                        f"Ratio: {flow_grpo_metrics.get('mean_ratio', 1):.4f} "
                        f"Clip: {flow_grpo_metrics.get('clip_fraction', 0):.3f} "
                        f"Reward: {flow_grpo_metrics.get('mean_reward', 0):.3f} "
                        f"SpyRate: {spy_rate:.2%} "
                        f"LR: {lr[1]:.6f}"
                    )
                    batch_time_m.reset()

                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                if ((global_step + 1) % config.experiment.generate_every == 0
                        and accelerator.is_main_process):
                    generate_sample_images(
                        spy_wrapper, game_generator, config,
                        global_step + 1, accelerator.device, weight_type,
                    )

                global_step += 1

        if global_step >= config.training.max_train_steps:
            break

    # Final save
    accelerator.wait_for_everyone()
    save_checkpoint(model, config, accelerator, "final")
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(
            config.experiment.output_dir, safe_serialization=False
        )
    accelerator.end_training()
    logger.info(f"Training complete! SpyRate: {spy_caught_count}/{total_games_played}")


@torch.no_grad()
def generate_sample_images(spy_wrapper, game_generator, config, global_step, device, weight_type):
    """Generate sample images for visualization."""
    logger.info("Generating sample images...")
    spy_wrapper.model.eval()
    game_data = game_generator.generate_game(epoch=9999, sample_idx=global_step)
    prompts = [game_generator.format_generation_prompt_simple(game_data, pid)
               for pid in range(1, game_data['num_players'] + 1)]
    gen_result = spy_wrapper.generate_images(
        prompts, guidance_scale=config.transport.guidance_scale,
        num_steps=config.transport.num_inference_steps,
    )
    pil_images = gen_result['images']
    captions = [f"P{pid} ({'SPY' if pid == game_data['spy_player'] else 'CIV'})"
                for pid in range(1, game_data['num_players'] + 1)]
    wandb.log({"spy_game_images": [wandb.Image(img, caption=cap)
               for img, cap in zip(pil_images, captions)]}, step=global_step)
    grid_dir = os.path.join(config.experiment.output_dir, "sample_images")
    os.makedirs(grid_dir, exist_ok=True)
    save_images_as_grid(pil_images, f"step_{global_step}", grid_dir,
                        grid_size=(2, 2) if len(pil_images) == 4 else (1, len(pil_images)))
    spy_wrapper.model.train()


def save_checkpoint(model, config, accelerator, global_step):
    """Save model checkpoint."""
    output_dir = config.experiment.output_dir
    limit = config.experiment.get("checkpoints_total_limit", None)
    if accelerator.is_main_process and limit is not None:
        checkpoints = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint")],
                             key=lambda x: int(x.split("-")[1]))
        for rm in checkpoints[:max(0, len(checkpoints) - limit + 1)]:
            shutil.rmtree(os.path.join(output_dir, rm))
    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(
            save_path / "unwrapped_model", save_function=accelerator.save,
            state_dict=state_dict, safe_serialization=False,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
