#!/bin/bash
# Flow-GRPO Training: 4-GPU, 4 players, full config
#
# SDE images participate in real games (Bagel-style)
# Hybrid ODE/SDE trajectory with CFG enabled
# Per-timestep backward for memory efficiency

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate spy-umm

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="/adialab/usr/shadabk/MedUMM/Show-o/show-o2:${PYTHONPATH:-}"

# Use first 4 GPUs
export CUDA_VISIBLE_DEVICES=${1:-"0,1,2,3"}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

cd /adialab/usr/shadabk/MedUMM/SPY-UMM

echo "============================================"
echo "  Flow-GRPO Training"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} cards)"
echo "  Players: 4, Groups: 2, T: 10, Window: 3"
echo "  Inner epochs: 1, CFG: 4.0, noise: 1.3"
echo "============================================"

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --mixed_precision=bf16 \
    train_spy_umm.py \
    config=configs/spy_umm_1.5b_flow_grpo.yaml \
    experiment.name="flow-grpo-train" \
    experiment.output_dir="output/flow-grpo-train" \
    experiment.save_every=200 \
    experiment.generate_every=100 \
    experiment.log_every=5 \
    game.num_players=4 \
    game.num_objects_min=3 \
    game.num_objects_max=5 \
    game.num_objects_to_modify=1 \
    game.epoch_size=500 \
    training.max_train_steps=5000 \
    training.gradient_accumulation_steps=1 \
    training.flow_grpo_group_size=2 \
    training.flow_grpo_train_steps=10 \
    training.sde_window_size=3 \
    training.sde_window_range="[0,-1]" \
    training.sde_noise_scale=1.3 \
    training.clip_range=1e-5 \
    training.flow_grpo_beta=0.0 \
    training.num_inner_epochs=1 \
    training.gen_loss_coeff=1.0 \
    training.max_grad_norm=1.0 \
    training.max_vote_tokens=64 \
    transport.num_inference_steps=10 \
    transport.guidance_scale=4.0 \
    2>&1 | tee output/flow-grpo-train.log
