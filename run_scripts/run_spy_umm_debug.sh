#!/bin/bash
# SPY-UMM Debug Launch Script (Single GPU, fewer steps)

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Add Show-o2 to PYTHONPATH
export PYTHONPATH="/adialab/usr/shadabk/MedUMM/Show-o/show-o2:${PYTHONPATH}"

cd /adialab/usr/shadabk/MedUMM/SPY-UMM

python train_spy_umm.py \
    config=configs/spy_umm_1.5b.yaml \
    training.max_train_steps=50 \
    training.gradient_accumulation_steps=1 \
    experiment.save_every=25 \
    experiment.generate_every=10 \
    experiment.log_every=1 \
    experiment.name="spy-umm-1.5b-debug" \
    experiment.output_dir="output/spy-umm-1.5b-debug"
