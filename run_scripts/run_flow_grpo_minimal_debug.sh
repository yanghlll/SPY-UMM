#!/bin/bash
# Flow-GRPO Minimal Debug: 所有参数调到最小，单卡测试
#
# num_players=2, group_size=1, train_steps=4, sde_window=1, inference_steps=4
# 目标：验证 SDE 图像参与真实游戏 + per-step backward 能跑通

set -euo pipefail

# Activate spy-umm conda environment
eval "$(conda shell.bash hook)"
conda activate spy-umm

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="/adialab/usr/shadabk/MedUMM/Show-o/show-o2:${PYTHONPATH:-}"

# 使用第一张空卡
export CUDA_VISIBLE_DEVICES=${1:-0}

cd /adialab/usr/shadabk/MedUMM/SPY-UMM

echo "=== Flow-GRPO Minimal Debug ==="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Players: 2, Groups: 1, TrainSteps: 4, SDEWindow: 1, InferSteps: 4"
echo "========================================="

python train_spy_umm.py \
    config=configs/spy_umm_1.5b_flow_grpo.yaml \
    experiment.name="flow-grpo-min-debug" \
    experiment.output_dir="output/flow-grpo-min-debug" \
    experiment.save_every=999 \
    experiment.generate_every=999 \
    experiment.log_every=1 \
    game.num_players=2 \
    game.num_objects_min=2 \
    game.num_objects_max=3 \
    game.num_objects_to_modify=1 \
    game.epoch_size=100 \
    training.max_train_steps=5 \
    training.gradient_accumulation_steps=1 \
    training.phase_mode="generation" \
    training.use_flow_grpo=True \
    training.flow_grpo_group_size=2 \
    training.flow_grpo_train_steps=4 \
    training.sde_window_size=1 \
    training.sde_window_range="[0,-1]" \
    training.flow_grpo_beta=0.0 \
    training.num_inner_epochs=2 \
    training.gen_loss_coeff=1.0 \
    training.vote_loss_coeff=0.0 \
    transport.num_inference_steps=10 \
    transport.guidance_scale=5.0 \
    2>&1 | tee output/flow-grpo-min-debug.log
