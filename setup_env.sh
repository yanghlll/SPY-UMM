#!/bin/bash
# SPY-UMM Environment Setup Script
# Based on Show-o2 (build_env.sh) + Vision-Zero dependencies
#
# Usage:
#   conda activate spy-umm
#   bash setup_env.sh

set -e

echo "=== Installing SPY-UMM dependencies ==="

# Core: PyTorch (Show-o2 requires 2.5.1)
pip3 install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# Show-o2 core dependencies
pip3 install transformers==4.47.0
pip3 install diffusers==0.31.0
pip3 install einops==0.8.0
pip3 install accelerate==0.23.0
pip3 install deepspeed==0.15.3
pip3 install timm==1.0.12
pip3 install huggingface-hub==0.24.0
pip3 install omegaconf
pip3 install torchdiffeq
pip3 install wandb
pip3 install sentencepiece
pip3 install decord
pip3 install gpustat
pip3 install ipdb

# Show-o2 additional
pip3 install ftfy regex tqdm
pip3 install git+https://github.com/openai/CLIP.git
pip3 install lightning==2.4.0
pip3 install dill
pip3 install pandas
pip3 install pyarrow==11.0.0
pip3 install jsonlines
pip3 install safetensors

# Vision-Zero / open-r1 dependencies (compatible versions)
pip3 install datasets
pip3 install tensorboardx
pip3 install matplotlib
pip3 install pycocotools
pip3 install pillow

# Flash Attention (requires CUDA, build from source)
pip3 install flash-attn --no-build-isolation

echo "=== Environment setup complete ==="
echo "Verify with: python -c 'import torch; print(torch.__version__)'"
