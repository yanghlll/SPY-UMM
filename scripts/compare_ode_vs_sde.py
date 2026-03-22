"""
Compare ODE vs SDE generation side-by-side.

For each player (SPY / CIV), generates one image via pure ODE (Show-o2 default)
and one via hybrid ODE/SDE (Flow-GRPO trajectory). Saves a comparison grid.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/compare_ode_vs_sde.py \
        config=configs/spy_umm_1.5b_flow_grpo.yaml
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
    device = torch.device("cuda")
    weight_type = get_weight_type(config)

    print("Loading models...")
    from models import WanVAE
    vae = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_type, device=device
    )

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True, return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path]
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    model = Showo2Qwen2_5.from_pretrained(
        config.model.showo.pretrained_model_path, use_safetensors=False
    ).to(device)

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
        config.dataset.preprocessing.num_mixed_modal_tokens += 1

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

    spy_wrapper = Showo2SpyWrapper(
        model=model, vae_model=vae, text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids, transport=transport,
        sampler=sampler, config=config,
    )

    T_sde = config.training.get("flow_grpo_train_steps", 10)
    flow_grpo = FlowGRPO(FlowGRPOConfig(
        sde_noise_scale=config.training.get("sde_noise_scale", 0.7),
        sde_window_size=config.training.get("sde_window_size", 2),
        sde_window_range=tuple(config.training.get("sde_window_range", [0, 5])),
        do_shift=config.transport.do_shift,
        time_shifting_factor=config.transport.time_shifting_factor,
    )).to(device)

    # Game setup
    game_gen = SpyGameDataGenerator(
        num_players=config.game.num_players,
        num_objects_min=config.game.num_objects_min,
        num_objects_max=config.game.num_objects_max,
        num_to_modify=config.game.num_objects_to_modify,
    )

    game_data = game_gen.generate_game(epoch=0, sample_idx=42)
    spy_pid = game_data['spy_player']
    num_players = game_data['num_players']

    print(f"\n{'='*70}")
    print(f"GAME: {num_players} players, Spy = Player {spy_pid}")
    print(f"{'='*70}")

    prompts = []
    for pid in range(1, num_players + 1):
        p = game_gen.format_generation_prompt_simple(game_data, pid)
        role = "SPY" if pid == spy_pid else "CIV"
        prompts.append(p)
        print(f"\n  Player {pid} ({role}) prompt:")
        print(f"    {p}")

    T_ode = config.transport.num_inference_steps
    guidance = config.transport.guidance_scale

    print(f"\n--- ODE generation: T={T_ode}, guidance={guidance} ---")
    with torch.no_grad():
        ode_result = spy_wrapper.generate_images(
            prompts, guidance_scale=guidance, num_steps=T_ode,
        )
    ode_images = ode_result['images']
    print(f"  ODE done. {len(ode_images)} images generated.")

    # SDE also uses CFG (like Bagel) — same guidance_scale as ODE
    print(f"\n--- SDE generation: T={T_sde}, cfg={guidance}, sde_window={flow_grpo.config.sde_window_size} ---")
    sde_images = []
    with torch.no_grad():
        for pid in range(num_players):
            inputs = spy_wrapper.prepare_flow_grpo_inputs([prompts[pid]], guidance_scale=guidance)
            vfn_kwargs = inputs['velocity_fn_kwargs']
            velocity_fn = spy_wrapper.make_velocity_fn(**vfn_kwargs)
            traj = flow_grpo.generate_trajectory(velocity_fn, inputs['z_init'], num_steps=T_sde)
            final = traj['final']
            img_tensor = vae.batch_decode(final.unsqueeze(2)).squeeze(2)
            img_np = denorm(img_tensor)
            sde_images.append(Image.fromarray(img_np[0]))
            sde_begin, sde_end = traj['sde_window']
            role = "SPY" if pid + 1 == spy_pid else "CIV"
            print(f"  Player {pid+1} ({role}): SDE window=[{sde_begin},{sde_end})")
    print(f"  SDE done. {len(sde_images)} images generated.")

    # Save
    out_dir = Path(f"output/compare_ode_vs_sde")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save individual images
    for pid in range(num_players):
        role = "SPY" if pid + 1 == spy_pid else "CIV"
        ode_images[pid].save(out_dir / f"p{pid+1}_{role}_ODE_T{T_ode}.png")
        sde_images[pid].save(out_dir / f"p{pid+1}_{role}_SDE_T{T_sde}.png")

    # Build comparison grid
    grid = make_comparison_grid(
        ode_images, sde_images, game_data, T_ode, T_sde, prompts
    )
    grid.save(out_dir / "comparison.png")

    # Save prompts
    info = {
        "spy_player": spy_pid,
        "T_ode": T_ode,
        "T_sde": T_sde,
        "guidance_ode": guidance,
        "guidance_sde": 0.0,
        "sde_noise_scale": flow_grpo.config.sde_noise_scale,
        "sde_window_size": flow_grpo.config.sde_window_size,
        "original_description": game_data['original_description'],
        "modified_description": game_data['modified_description'],
    }
    for pid in range(num_players):
        role = "SPY" if pid + 1 == spy_pid else "CIV"
        info[f"player{pid+1}_role"] = role
        info[f"player{pid+1}_prompt"] = prompts[pid]

    with open(out_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_dir}/")
    print(f"  comparison.png   — side-by-side ODE vs SDE grid")
    for pid in range(num_players):
        role = "SPY" if pid + 1 == spy_pid else "CIV"
        print(f"  p{pid+1}_{role}_ODE_T{T_ode}.png")
        print(f"  p{pid+1}_{role}_SDE_T{T_sde}.png")
    print(f"  info.json        — prompts and config")


def make_comparison_grid(ode_images, sde_images, game_data, T_ode, T_sde, prompts):
    """Grid: rows = players, cols = [ODE, SDE]."""
    N = len(ode_images)
    spy_pid = game_data['spy_player']
    img_size = 432
    padding = 12
    header_h = 90
    col_header_h = 40
    prompt_h = 50

    grid_w = 2 * img_size + 3 * padding
    grid_h = col_header_h + N * (header_h + img_size + prompt_h) + (N + 1) * padding

    grid = Image.new('RGB', (grid_w, grid_h), (25, 25, 25))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font
        font_md = font

    # Column headers
    x_ode = padding
    x_sde = padding * 2 + img_size
    draw.text((x_ode + img_size // 2 - 80, 8),
              f"ODE (T={T_ode}, cfg={5.0})", fill=(100, 200, 255), font=font)
    draw.text((x_sde + img_size // 2 - 80, 8),
              f"SDE (T={T_sde}, cfg={5.0})", fill=(255, 200, 100), font=font)

    for pid in range(N):
        role = "SPY" if pid + 1 == spy_pid else "CIV"
        role_color = (255, 80, 80) if role == "SPY" else (80, 200, 80)

        y_base = col_header_h + padding + pid * (header_h + img_size + prompt_h + padding)

        # Player header
        draw.text((padding, y_base), f"Player {pid+1} ({role})",
                  fill=role_color, font=font)

        # Prompt text (truncated)
        prompt_text = prompts[pid]
        if len(prompt_text) > 100:
            prompt_text = prompt_text[:97] + "..."
        draw.text((padding, y_base + 28), prompt_text,
                  fill=(180, 180, 180), font=font_sm)

        img_y = y_base + header_h

        # ODE image
        border = 3
        draw.rectangle([x_ode - border, img_y - border,
                        x_ode + img_size + border, img_y + img_size + border],
                       outline=(100, 200, 255), width=border)
        grid.paste(ode_images[pid].resize((img_size, img_size)), (x_ode, img_y))

        # SDE image
        draw.rectangle([x_sde - border, img_y - border,
                        x_sde + img_size + border, img_y + img_size + border],
                       outline=(255, 200, 100), width=border)
        grid.paste(sde_images[pid].resize((img_size, img_size)), (x_sde, img_y))

    return grid


if __name__ == "__main__":
    main()
