#!/bin/bash
# SPY-UMM + Flow-GRPO Debug Launch Script (Single GPU, fewer steps)
#
# 单 GPU 快速调试，验证 Flow-GRPO 能否跑通。
# 参数全部调到最小：G=2, T=3, ODE推理5步, 总共20步。

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Add Show-o2 to PYTHONPATH
export PYTHONPATH="/nfs-stor/haolin.yang/Code/UMM/Show-o/show-o2:${PYTHONPATH}"

cd /nfs-stor/haolin.yang/Code/UMM/SPY-UMM

# 可选参数：传入 "fast" 启用 Flow-GRPO-Fast 模式
MODE=${1:-"full"}

if [ "$MODE" = "fast" ]; then
    echo "=== Flow-GRPO-Fast Debug Mode ==="
    EXTRA_ARGS="training.flow_grpo_fast_mode=True training.sde_window_size=1"
    EXP_NAME="flow-grpo-fast-debug"
else
    echo "=== Flow-GRPO Full Debug Mode ==="
    EXTRA_ARGS="training.flow_grpo_fast_mode=False"
    EXP_NAME="flow-grpo-debug"
fi

python train_spy_umm.py \
    config=configs/spy_umm_1.5b_flow_grpo.yaml \
    training.max_train_steps=20 \
    training.gradient_accumulation_steps=1 \
    training.flow_grpo_group_size=2 \
    training.flow_grpo_train_steps=3 \
    training.flow_grpo_beta=0.0 \
    transport.num_inference_steps=5 \
    experiment.save_every=10 \
    experiment.generate_every=10 \
    experiment.log_every=1 \
    experiment.name="${EXP_NAME}" \
    experiment.output_dir="output/${EXP_NAME}" \
    ${EXTRA_ARGS}
