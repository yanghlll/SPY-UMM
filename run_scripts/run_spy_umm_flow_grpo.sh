#!/bin/bash
# SPY-UMM + Flow-GRPO Training Launch Script
#
# Uses ODE->SDE conversion for proper policy gradient optimization
# on Show-o2's flow matching head via GRPO.
#
# Config variants:
#   spy_umm_1.5b_flow_grpo.yaml      - Full Flow-GRPO (all SDE steps)
#   spy_umm_1.5b_flow_grpo_fast.yaml  - Flow-GRPO-Fast (1-2 SDE steps, 5-10x faster)

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Add Show-o2 to PYTHONPATH
export PYTHONPATH="/adialab/usr/shadabk/MedUMM/Show-o/show-o2:${PYTHONPATH}"

cd /adialab/usr/shadabk/MedUMM/SPY-UMM

# Default config: Flow-GRPO (full)
# Change to spy_umm_1.5b_flow_grpo_fast.yaml for Fast mode
CONFIG=${1:-"configs/spy_umm_1.5b_flow_grpo.yaml"}

accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    --main_process_port=9998 \
    train_spy_umm.py \
    config=${CONFIG}
