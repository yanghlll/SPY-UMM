"""
Reward-Weighted Flow Matching Loss.

Core innovation: weights the standard flow matching MSE loss by game rewards.
Since Show-o2 uses continuous flow matching (not discrete tokens), we cannot
use GRPO directly. Instead, we scale the velocity prediction loss by the
reward signal from the spy game.

Standard flow matching:
    loss = MSE(v_pred, v_target)

Reward-weighted flow matching:
    advantage = (reward - baseline) / std
    weight = max(advantage, 0)  # Only reinforce positive outcomes
    loss = weight * MSE(v_pred, v_target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class RewardWeightedFlowMatchingLoss(nn.Module):
    """Computes reward-weighted flow matching loss.

    Integrates game reward signals into the continuous diffusion training
    by weighting the velocity prediction MSE loss per sample.
    """

    def __init__(self, reward_baseline_ema: float = 0.9,
                 reward_clamp_min: float = 0.0):
        super().__init__()
        self.ema = reward_baseline_ema
        self.clamp_min = reward_clamp_min

        # EMA baselines for spy and civilian roles
        self.register_buffer('baseline_spy', torch.tensor(0.0))
        self.register_buffer('baseline_civ', torch.tensor(0.0))
        self.register_buffer('baseline_global', torch.tensor(0.0))
        self.update_count = 0

    def compute_weights(self, rewards: torch.Tensor,
                        is_spy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reward-based weights for loss scaling.

        Args:
            rewards: [B] reward values.
            is_spy: [B] boolean mask, True for spy players.

        Returns:
            [B] non-negative weights for loss scaling.
        """
        B = rewards.shape[0]

        if is_spy is not None and is_spy.any():
            # Use role-specific baselines
            weights = torch.zeros(B, device=rewards.device)

            spy_mask = is_spy.bool()
            civ_mask = ~spy_mask

            if spy_mask.any():
                spy_rewards = rewards[spy_mask]
                spy_adv = spy_rewards - self.baseline_spy
                weights[spy_mask] = spy_adv

                # Update spy baseline
                self.baseline_spy = (
                    self.ema * self.baseline_spy +
                    (1 - self.ema) * spy_rewards.mean()
                )

            if civ_mask.any():
                civ_rewards = rewards[civ_mask]
                civ_adv = civ_rewards - self.baseline_civ
                weights[civ_mask] = civ_adv

                # Update civilian baseline
                self.baseline_civ = (
                    self.ema * self.baseline_civ +
                    (1 - self.ema) * civ_rewards.mean()
                )
        else:
            # Global baseline
            advantages = rewards - self.baseline_global
            weights = advantages

            # Update global baseline
            self.baseline_global = (
                self.ema * self.baseline_global +
                (1 - self.ema) * rewards.mean()
            )

        # Normalize advantages
        if weights.numel() > 1:
            std = weights.std().clamp(min=1e-8)
            weights = weights / std

        # Clamp to non-negative (only reinforce good outcomes)
        weights = weights.clamp(min=self.clamp_min)

        self.update_count += 1
        return weights

    def forward(self, v_pred: torch.Tensor, v_target: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                rewards: Optional[torch.Tensor] = None,
                is_spy: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute reward-weighted flow matching loss.

        Args:
            v_pred: [B, L, D] predicted velocity.
            v_target: [B, L, D] target velocity.
            mask: [B, L] mask for valid positions.
            rewards: [B] reward values. If None, uses uniform weighting.
            is_spy: [B] boolean, True for spy players.

        Returns:
            Dict with 'loss', 'per_sample_loss', 'weights', 'metrics'.
        """
        # Per-element MSE
        mse = F.mse_loss(v_pred, v_target, reduction='none')  # [B, L, D]

        if mask is not None:
            # Apply mask and compute per-sample mean
            mask_expanded = mask.unsqueeze(-1).expand_as(mse)
            mse = mse * mask_expanded
            per_sample_loss = mse.sum(dim=[1, 2]) / mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        else:
            per_sample_loss = mse.mean(dim=[1, 2])  # [B]

        if rewards is not None:
            weights = self.compute_weights(rewards, is_spy)
            weighted_loss = (weights * per_sample_loss).sum() / weights.sum().clamp(min=1e-8)
        else:
            weights = torch.ones_like(per_sample_loss)
            weighted_loss = per_sample_loss.mean()

        return {
            'loss': weighted_loss,
            'per_sample_loss': per_sample_loss.detach(),
            'weights': weights.detach(),
            'metrics': {
                'baseline_spy': self.baseline_spy.item(),
                'baseline_civ': self.baseline_civ.item(),
                'baseline_global': self.baseline_global.item(),
                'mean_weight': weights.mean().item(),
                'mean_per_sample_loss': per_sample_loss.mean().item(),
                'update_count': self.update_count,
            }
        }


def velocity_prediction_per_sample(latents: torch.Tensor,
                                   labels: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-sample MSE velocity prediction loss.

    Drop-in replacement for Show-o2's velocity_prediction() that returns
    per-sample losses instead of a scalar mean.

    Args:
        latents: [B, L, D] predicted velocities.
        labels: [B, L, D] target velocities.
        mask: [B, L] mask for valid positions.

    Returns:
        [B] per-sample loss values.
    """
    mse = F.mse_loss(latents, labels, reduction='none')  # [B, L, D]

    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand_as(mse)
        mse = mse * mask_expanded
        per_sample = mse.sum(dim=[1, 2]) / mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
    else:
        per_sample = mse.mean(dim=[1, 2])

    return per_sample  # [B]
