"""
SPY-UMM Inference and Evaluation.

Runs complete spy games with a trained model and evaluates:
1. Spy detection rate (how often spy is caught by majority vote)
2. Vote accuracy (mean reward of voting responses)
3. Format compliance of voting responses
4. Saves generated image grids for visualization
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add Show-o2 to path
SHOWO2_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Show-o', 'show-o2')
SHOWO2_DIR = os.path.normpath(SHOWO2_DIR)
if SHOWO2_DIR not in sys.path:
    sys.path.insert(0, SHOWO2_DIR)

from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.misc import get_text_tokenizer, get_weight_type
from utils import get_config, denorm, path_to_llm_name, load_state_dict
from transport import Sampler, create_transport

# SPY-UMM imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data import SpyGameDataGenerator
from models.showo2_spy_wrapper import Showo2SpyWrapper
from training.rewards import vote_accuracy_reward, vote_format_reward


def main():
    config = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_type = torch.bfloat16

    # Output directory
    output_dir = config.get("eval_output_dir", "eval_output")
    os.makedirs(output_dir, exist_ok=True)

    num_eval_games = config.get("eval_games", 100)

    print(f"Running {num_eval_games} evaluation games...")
    print(f"Output: {output_dir}")

    # ========================
    # Load models
    # ========================
    print("Loading models...")

    # VAE
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(
            vae_pth=config.model.vae_model.pretrained_model_path,
            dtype=weight_type, device=device
        )
    else:
        raise NotImplementedError

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
        ).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)

    # Load checkpoint if specified
    checkpoint_path = config.get("checkpoint_path", None)
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = load_state_dict(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        del state_dict

    model.to(weight_type)
    model.eval()

    # Time embedding adjustment
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_t2i_image_tokens += 1

    # Transport + Sampler
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

    # Spy wrapper
    spy_wrapper = Showo2SpyWrapper(
        model=model, vae_model=vae_model,
        text_tokenizer=text_tokenizer, showo_token_ids=showo_token_ids,
        transport=transport, sampler=sampler, config=config,
    )

    # Game generator
    game_config = config.game
    game_generator = SpyGameDataGenerator(
        num_players=game_config.num_players,
        num_objects_min=game_config.num_objects_min,
        num_objects_max=game_config.num_objects_max,
        num_to_modify=game_config.num_objects_to_modify,
    )

    # ========================
    # Run evaluation games
    # ========================
    results = {
        'spy_caught_count': 0,
        'total_games': 0,
        'vote_accuracy_rewards': [],
        'format_scores': [],
        'games': [],
    }

    for game_idx in range(num_eval_games):
        print(f"\n--- Game {game_idx + 1}/{num_eval_games} ---")

        game_data = game_generator.generate_game(epoch=9999, sample_idx=game_idx)

        # Generate images
        prompts = [
            game_generator.format_generation_prompt_simple(game_data, pid)
            for pid in range(1, game_data['num_players'] + 1)
        ]

        with torch.no_grad():
            gen_result = spy_wrapper.generate_images(
                prompts,
                guidance_scale=config.transport.guidance_scale,
                num_steps=config.transport.num_inference_steps,
            )

        # Run voting
        voting_prompt = game_generator.format_voting_prompt(game_data)
        with torch.no_grad():
            vote_text = spy_wrapper.judge_vote(
                [gen_result['latents'][i:i + 1] for i in range(game_data['num_players'])],
                voting_prompt,
                max_new_tokens=512,
                temperature=0.7,
                top_k=50,
            )

        # Score
        vote_info = game_generator.extract_vote(vote_text)
        acc_r = vote_accuracy_reward(vote_text, game_data['spy_player'])
        fmt_r = vote_format_reward(vote_text)

        spy_caught = (vote_info is not None and
                      vote_info.get('voted_spy') == game_data['spy_player'])

        results['vote_accuracy_rewards'].append(acc_r)
        results['format_scores'].append(fmt_r)
        results['total_games'] += 1
        if spy_caught:
            results['spy_caught_count'] += 1

        game_record = {
            'game_id': game_data['game_id'],
            'spy_player': game_data['spy_player'],
            'vote_text': vote_text,
            'voted_spy': vote_info.get('voted_spy') if vote_info else None,
            'spy_caught': spy_caught,
            'accuracy_reward': acc_r,
            'format_score': fmt_r,
        }
        results['games'].append(game_record)

        print(f"  Spy: Player {game_data['spy_player']}")
        print(f"  Vote: {vote_info}")
        print(f"  Caught: {spy_caught} | Acc: {acc_r:.1f} | Fmt: {fmt_r:.2f}")

        # Save images for first N games
        if game_idx < 20:
            save_game_visualization(
                gen_result['images'], game_data,
                vote_text, output_dir, game_idx,
            )

    # ========================
    # Summary
    # ========================
    detection_rate = results['spy_caught_count'] / max(results['total_games'], 1)
    mean_acc = sum(results['vote_accuracy_rewards']) / max(len(results['vote_accuracy_rewards']), 1)
    mean_fmt = sum(results['format_scores']) / max(len(results['format_scores']), 1)

    summary = {
        'total_games': results['total_games'],
        'spy_caught_count': results['spy_caught_count'],
        'spy_detection_rate': detection_rate,
        'mean_vote_accuracy_reward': mean_acc,
        'mean_format_score': mean_fmt,
    }

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Games:              {summary['total_games']}")
    print(f"  Spy Detection Rate: {detection_rate:.2%}")
    print(f"  Mean Vote Accuracy: {mean_acc:.3f}")
    print(f"  Mean Format Score:  {mean_fmt:.3f}")

    # Save results
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump({**summary, 'games': results['games']}, f, indent=2)
    print(f"\nResults saved to {results_path}")


def save_game_visualization(images: list, game_data: dict,
                            vote_text: str, output_dir: str, game_idx: int):
    """Save a 2x2 grid of player images with spy annotation."""
    num_players = len(images)
    spy_player = game_data['spy_player']

    # Create grid
    img_size = images[0].size[0]
    cols = 2
    rows = (num_players + cols - 1) // cols
    grid_w = cols * img_size
    grid_h = rows * img_size
    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))

    for idx, img in enumerate(images):
        row, col = idx // cols, idx % cols
        grid.paste(img, (col * img_size, row * img_size))

    # Add border for spy
    draw = ImageDraw.Draw(grid)
    spy_idx = spy_player - 1
    spy_row, spy_col = spy_idx // cols, spy_idx % cols
    x0 = spy_col * img_size
    y0 = spy_row * img_size
    for offset in range(3):
        draw.rectangle(
            [x0 + offset, y0 + offset, x0 + img_size - offset, y0 + img_size - offset],
            outline='red',
        )

    # Add labels
    for idx in range(num_players):
        row, col = idx // cols, idx % cols
        role = "SPY" if idx + 1 == spy_player else "CIV"
        label = f"P{idx + 1} ({role})"
        draw.text((col * img_size + 5, row * img_size + 5), label, fill='white')

    # Save
    images_dir = os.path.join(output_dir, "game_images")
    os.makedirs(images_dir, exist_ok=True)
    grid.save(os.path.join(images_dir, f"game_{game_idx:04d}.png"))

    # Save vote text
    vote_dir = os.path.join(output_dir, "vote_texts")
    os.makedirs(vote_dir, exist_ok=True)
    with open(os.path.join(vote_dir, f"game_{game_idx:04d}.txt"), 'w') as f:
        f.write(f"Spy: Player {spy_player}\n\n")
        f.write(f"Vote Response:\n{vote_text}\n")


if __name__ == '__main__':
    main()
