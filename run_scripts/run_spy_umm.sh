#!/bin/bash
# SPY-UMM Training Launch Script (Multi-GPU with DeepSpeed)

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Add Show-o2 to PYTHONPATH
export PYTHONPATH="/adialab/usr/shadabk/MedUMM/Show-o/show-o2:${PYTHONPATH}"

cd /adialab/usr/shadabk/MedUMM/SPY-UMM

accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    --main_process_port=9998 \
    train_spy_umm.py \
    config=configs/spy_umm_1.5b.yaml
