"""
Vision-Zero CLEVR Data Adapter for SPY-UMM.

Loads Vision-Zero's pre-rendered CLEVR image pairs (original/modified) and
encodes them through WanVAE to serve as ground-truth target latents for
the flow matching training objective.
"""

import os
import re
import random
from functools import lru_cache
from typing import Dict, Any, List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class VisionZeroDataAdapter:
    """Loads Vision-Zero CLEVR images as ground-truth targets for flow matching.

    CLEVR images are pre-rendered 3D scenes with original/modified pairs.
    The modified version has some objects replaced with different shapes/colors.
    These pairs align with the spy game structure: civilians get original,
    spy gets modified.
    """

    def __init__(self, images_dir: str, vae_model, image_size: int = 432,
                 device: str = 'cuda', dtype=torch.bfloat16,
                 cache_size: int = 256):
        """
        Args:
            images_dir: Path to Vision-Zero CLEVR replacement_images directory.
            vae_model: WanVAE instance for encoding images to latents.
            image_size: Resolution to resize images to (432 matches config).
            device: Device for latent tensors.
            dtype: Weight type.
            cache_size: Max number of latents to cache in memory.
        """
        self.images_dir = images_dir
        self.vae = vae_model
        self.image_size = image_size
        self.device = device
        self.dtype = dtype
        self.cache_size = cache_size

        # Image preprocessing: resize + normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [0,1] -> [-1,1]
        ])

        # Load available pairs
        self.pairs = self._load_pairs()
        if len(self.pairs) == 0:
            raise ValueError(f"No CLEVR image pairs found in {images_dir}")

        # LRU cache for encoded latents
        self._latent_cache = {}
        self._cache_order = []

    def _load_pairs(self) -> List[Dict[str, str]]:
        """Scan directory for original/modified image pairs."""
        if not os.path.isdir(self.images_dir):
            print(f"WARNING: Vision-Zero images directory not found: {self.images_dir}")
            return []

        # Find all _original.png files and match with _modified.png
        files = os.listdir(self.images_dir)
        original_files = sorted([
            f for f in files
            if f.endswith('_original.png')
        ])

        pairs = []
        for orig_file in original_files:
            # CLEVR_REPLACEMENT_replacement_XXXXXX_original.png
            # -> CLEVR_REPLACEMENT_replacement_XXXXXX_modified.png
            mod_file = orig_file.replace('_original.png', '_modified.png')
            if mod_file in files:
                pairs.append({
                    'original': os.path.join(self.images_dir, orig_file),
                    'modified': os.path.join(self.images_dir, mod_file),
                    'name': orig_file.replace('_original.png', ''),
                })

        print(f"VisionZeroDataAdapter: Found {len(pairs)} CLEVR image pairs")
        return pairs

    def _encode_image(self, image_path: str) -> torch.Tensor:
        """Load, preprocess, and encode a single image through VAE.

        Returns:
            [1, C, H, W] latent tensor.
        """
        # Check cache
        if image_path in self._latent_cache:
            return self._latent_cache[image_path]

        # Load and preprocess
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img)  # [3, H, W], [-1, 1]
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
        img_tensor = img_tensor.to(self.dtype).to(self.device)

        # Encode through VAE
        with torch.no_grad():
            latent = self.vae.sample(img_tensor)  # [1, C, 1, h, w]
            latent = latent.squeeze(2)  # [1, C, h, w]

        # Cache with eviction
        if len(self._cache_order) >= self.cache_size:
            evict_key = self._cache_order.pop(0)
            self._latent_cache.pop(evict_key, None)
        self._latent_cache[image_path] = latent
        self._cache_order.append(image_path)

        return latent

    def get_target_latents(self, game_data: Dict[str, Any],
                           device: Optional[str] = None) -> torch.Tensor:
        """Get target latents for all players in a game.

        Civilian players get the 'original' CLEVR image latent,
        spy player gets the 'modified' CLEVR image latent.

        Args:
            game_data: Game data from SpyGameDataGenerator.generate_game().
            device: Target device (defaults to self.device).

        Returns:
            [num_players, C, H, W] tensor of target latents.
        """
        device = device or self.device

        # Select a CLEVR pair deterministically based on game_id
        pair_idx = hash(game_data['game_id']) % len(self.pairs)
        pair = self.pairs[pair_idx]

        original_latent = self._encode_image(pair['original'])
        modified_latent = self._encode_image(pair['modified'])

        targets = []
        for pid in range(1, game_data['num_players'] + 1):
            if pid == game_data['spy_player']:
                targets.append(modified_latent)
            else:
                targets.append(original_latent)

        return torch.cat(targets, dim=0).to(device)

    def get_random_pair(self, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Get a random CLEVR pair encoded as latents.

        Returns:
            Dict with 'original' and 'modified' latent tensors, each [1, C, H, W].
        """
        rng = random.Random(seed)
        pair = rng.choice(self.pairs)
        return {
            'original': self._encode_image(pair['original']),
            'modified': self._encode_image(pair['modified']),
            'name': pair['name'],
        }

    def __len__(self) -> int:
        return len(self.pairs)
