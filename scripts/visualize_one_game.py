"""
Visualize one Flow-GRPO game: generate SDE images, run voting, display results.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_one_game.py \
        config=configs/spy_umm_1.5b_flow_grpo.yaml \
        game.num_players=4 \
        transport.num_inference_steps=20 \
        training.flow_grpo_train_steps=10
"""

import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SHOWO2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Show-o', 'show-o2'))
sys.path = [p for p in sys.path if os.path.abspath(p) != SHOWO2_DIR]
sys.path.insert(0, SHOWO2_DIR)

from models import Showo2Qwen2_5
from models.misc import get_text_tokenizer, get_weight_type
from utils import get_config, denorm, path_to_llm_name
from transport import Sampler, create_transport

_showo2_cache = {k: sys.modules[k] for k in list(sys.modules.keys())
                 if k == 'models' or k.startswith('models.')}
for k in _showo2_cache:
    del sys.modules[k]
if SHOWO2_DIR in sys.path:
    sys.path.remove(SHOWO2_DIR)

from data import SpyGameDataGenerator
from models.showo2_spy_wrapper import Showo2SpyWrapper
from training.flow_grpo import FlowGRPO, FlowGRPOConfig

sys.modules.update(_showo2_cache)


def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_type = get_weight_type(config)

    print("Loading models...")
    # VAE
    from models import WanVAE
    vae = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_type, device=device
    )

    # Tokenizer
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True, return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path]
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    # Model
    model = Showo2Qwen2_5.from_pretrained(
        config.model.showo.pretrained_model_path, use_safetensors=False
    ).to(device)

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
        config.dataset.preprocessing.num_mixed_modal_tokens += 1

    # Transport
    preproc = config.dataset.preprocessing
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=preproc.num_t2i_image_tokens,
    )
    sampler = Sampler(transport)

    # Wrapper
    spy_wrapper = Showo2SpyWrapper(
        model=model, vae_model=vae, text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids, transport=transport,
        sampler=sampler, config=config,
    )

    # Flow-GRPO
    T = config.training.get("flow_grpo_train_steps", 10)
    flow_grpo = FlowGRPO(FlowGRPOConfig(
        sde_noise_scale=config.training.get("sde_noise_scale", 0.7),
        sde_window_size=config.training.get("sde_window_size", 2),
        sde_window_range=tuple(config.training.get("sde_window_range", [0, 5])),
        do_shift=config.transport.do_shift,
        time_shifting_factor=config.transport.time_shifting_factor,
    )).to(device)

    # Game data
    game_config = config.game
    game_gen = SpyGameDataGenerator(
        num_players=game_config.num_players,
        num_objects_min=game_config.num_objects_min,
        num_objects_max=game_config.num_objects_max,
        num_to_modify=game_config.num_objects_to_modify,
    )

    num_players = game_config.num_players
    game_data = game_gen.generate_game(epoch=0, sample_idx=42)
    spy_pid = game_data['spy_player']

    print(f"\n{'='*60}")
    print(f"GAME SETUP: {num_players} players, Spy = Player {spy_pid}")
    print(f"{'='*60}")
    print(f"\nOriginal description:\n  {game_data['original_description']}")
    print(f"\nModified description (spy gets this):\n  {game_data['modified_description']}")
    print(f"\nDifferences: {game_data['diff_metadata']}")

    # Generate prompts
    prompts = [
        game_gen.format_generation_prompt_simple(game_data, pid)
        for pid in range(1, num_players + 1)
    ]

    # Generate images via SDE
    print(f"\nGenerating {num_players} images via hybrid ODE/SDE (T={T})...")
    all_latents = []
    all_trajs = []

    with torch.no_grad():
        for pid in range(num_players):
            inputs = spy_wrapper.prepare_flow_grpo_inputs([prompts[pid]], guidance_scale=0.0)
            vfn_kwargs = inputs['velocity_fn_kwargs']
            velocity_fn = spy_wrapper.make_velocity_fn(**vfn_kwargs)
            traj = flow_grpo.generate_trajectory(velocity_fn, inputs['z_init'], num_steps=T)
            all_latents.append(traj['final'])
            all_trajs.append(traj)
            sde_begin, sde_end = traj['sde_window']
            print(f"  Player {pid+1} ({'SPY' if pid+1 == spy_pid else 'CIV'}): "
                  f"SDE window=[{sde_begin},{sde_end}), "
                  f"{len(traj['sde_steps'])} SDE steps")

    latents = torch.cat(all_latents, dim=0)

    # Decode images
    print("Decoding images via VAE...")
    with torch.no_grad():
        images_tensor = vae.batch_decode(latents.unsqueeze(2)).squeeze(2)
        images_np = denorm(images_tensor)
        pil_images = [Image.fromarray(img) for img in images_np]

    # Voting
    print(f"\nRunning voting ({num_players} voters)...")
    voting_prompt = game_gen.format_voting_prompt(game_data)
    vote_responses = []
    game_votes = []

    with torch.no_grad():
        for pid in range(num_players):
            resp = spy_wrapper.judge_vote(
                [latents[pid:pid+1]], voting_prompt,
                max_new_tokens=64, temperature=0.7,
            )
            vote_info = game_gen.extract_vote(resp)
            vote_responses.append(resp)
            game_votes.append(vote_info)

            voted = vote_info['voted_spy'] if vote_info else "INVALID"
            correct = "CORRECT" if vote_info and vote_info.get('voted_spy') == spy_pid else "WRONG"
            print(f"  Player {pid+1} voted: {voted} ({correct})")

    # Rewards
    game_outcome = game_gen.calculate_game_rewards(game_data, game_votes)
    gen_rewards = game_gen.compute_generation_rewards(game_outcome)

    print(f"\n{'='*60}")
    print(f"GAME RESULT: Spy {'CAUGHT' if game_outcome['spy_caught'] else 'ESCAPED'}")
    print(f"Vote counts: {game_outcome['vote_counts']}")
    print(f"Gen rewards: {gen_rewards}")
    print(f"{'='*60}")

    # Save outputs — use T in dir name for comparison
    out_dir = Path(f"output/visualize_game_T{T}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save individual images
    for pid in range(num_players):
        role = "SPY" if pid + 1 == spy_pid else "CIV"
        reward = gen_rewards[pid]
        fname = f"player{pid+1}_{role}_r{reward:.1f}.png"
        pil_images[pid].save(out_dir / fname)

    # Save grid with annotations
    grid = make_annotated_grid(
        pil_images, game_data, game_outcome,
        gen_rewards, vote_responses, game_votes
    )
    grid.save(out_dir / "game_grid.png")
    print(f"\nSaved to {out_dir}/")
    print(f"  game_grid.png — annotated grid of all players")
    for pid in range(num_players):
        role = "SPY" if pid + 1 == spy_pid else "CIV"
        print(f"  player{pid+1}_{role}_r{gen_rewards[pid]:.1f}.png")

    # Save full game data as JSON
    game_log = {
        "game_data": {k: v for k, v in game_data.items() if k != 'diff_metadata'},
        "spy_player": spy_pid,
        "vote_responses": vote_responses,
        "votes": [str(v) for v in game_votes],
        "spy_caught": game_outcome['spy_caught'],
        "vote_counts": game_outcome['vote_counts'],
        "gen_rewards": gen_rewards,
    }
    with open(out_dir / "game_log.json", "w") as f:
        json.dump(game_log, f, indent=2, ensure_ascii=False)
    print(f"  game_log.json — full game data")


def make_annotated_grid(images, game_data, outcome, rewards, responses, votes):
    """Create an annotated image grid showing game results."""
    N = len(images)
    spy_pid = game_data['spy_player']
    img_size = images[0].size[0]
    padding = 10
    header_h = 80
    footer_h = 60

    # Grid layout
    cols = min(N, 4)
    rows = (N + cols - 1) // cols
    grid_w = cols * img_size + (cols + 1) * padding
    grid_h = rows * (img_size + header_h + footer_h) + (rows + 1) * padding + 60

    grid = Image.new('RGB', (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    # Title
    result_text = f"Spy = Player {spy_pid} | Result: {'CAUGHT' if outcome['spy_caught'] else 'ESCAPED'}"
    draw.text((padding, 10), result_text, fill=(255, 255, 100), font=font)
    draw.text((padding, 35), f"Votes: {outcome['vote_counts']}", fill=(180, 180, 180), font=font_sm)

    for idx in range(N):
        pid = idx + 1
        role = "SPY" if pid == spy_pid else "CIV"
        reward = rewards[idx]

        col = idx % cols
        row = idx // cols
        x = padding + col * (img_size + padding)
        y = 60 + padding + row * (img_size + header_h + footer_h + padding)

        # Header
        role_color = (255, 80, 80) if role == "SPY" else (80, 200, 80)
        draw.text((x, y), f"Player {pid} ({role})", fill=role_color, font=font)
        r_color = (80, 255, 80) if reward > 0 else (255, 80, 80)
        draw.text((x, y + 25), f"Reward: {reward:+.1f}", fill=r_color, font=font_sm)

        # Image
        img_y = y + header_h
        # Add colored border
        border = 3
        border_color = (255, 60, 60) if role == "SPY" else (60, 180, 60)
        draw.rectangle([x - border, img_y - border,
                        x + img_size + border, img_y + img_size + border],
                       outline=border_color, width=border)
        grid.paste(images[idx].resize((img_size, img_size)), (x, img_y))

        # Footer: vote info
        vote = votes[idx]
        voted = vote['voted_spy'] if vote else "INVALID"
        correct = voted == spy_pid
        vote_color = (80, 255, 80) if correct else (255, 150, 80)
        draw.text((x, img_y + img_size + 5),
                  f"Voted: P{voted}" + (" ✓" if correct else " ✗"),
                  fill=vote_color, font=font_sm)

    return grid


if __name__ == "__main__":
    main()
