"""
Flow-GRPO: Group Relative Policy Optimization for Flow Matching.

Implements the ODE→SDE conversion from Flow-GRPO (arXiv 2505.05470) to enable
proper policy gradient optimization on continuous flow matching models.

Architecture follows the Bagel Flow-GRPO reference implementation:
1. Hybrid ODE/SDE trajectory: most steps are deterministic ODE, only a random
   contiguous window uses stochastic SDE to create trainable "actions".
2. Per-timestep backward: each SDE step gets its own gradient update for
   memory efficiency (no need to store all intermediate activations).
3. Each generated image gets its own real reward (not borrowed from ODE).

Reference: Flow-GRPO (arXiv 2505.05470, NeurIPS 2025)
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class FlowGRPOConfig:
    """Configuration for Flow-GRPO training."""
    # SDE noise schedule: σ_t = a * sqrt(t / (1-t))
    sde_noise_scale: float = 0.7
    # GRPO hyperparameters
    group_size: int = 4          # G: number of games per training step
    epsilon: float = 0.2         # PPO clip parameter
    beta: float = 0.01           # KL penalty coefficient
    # Denoising steps
    num_train_steps: int = 10    # T: total denoising steps during training
    num_inference_steps: int = 50
    # SDE window: contiguous window of SDE steps within the trajectory
    sde_window_size: int = 2     # number of SDE steps (rest are ODE)
    sde_window_range: Tuple[int, int] = (0, 5)  # step index range for window start
    # Time shifting (from Show-o2's transport)
    do_shift: bool = True
    time_shifting_factor: float = 3.0


class FlowGRPO(nn.Module):
    """Flow-GRPO: GRPO for continuous flow matching models.

    Follows the Bagel reference implementation pattern:
    - Hybrid ODE/SDE trajectory generation
    - Per-timestep GRPO loss computation
    - Model-agnostic interface via velocity_fn callable
    """

    def __init__(self, config: FlowGRPOConfig):
        super().__init__()
        self.config = config
        self.a = config.sde_noise_scale

    # ==================== NOISE SCHEDULE ====================

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """σ_t = a * sqrt(t / (1-t))."""
        t_clamped = t.clamp(min=1e-4, max=1.0 - 1e-4)
        return self.a * torch.sqrt(t_clamped / (1.0 - t_clamped))

    def sigma_t_sq(self, t: torch.Tensor) -> torch.Tensor:
        """σ_t² = a² * t / (1-t)."""
        t_clamped = t.clamp(min=1e-4, max=1.0 - 1e-4)
        return self.a ** 2 * t_clamped / (1.0 - t_clamped)

    # ==================== ODE → SDE CONVERSION ====================

    def sde_drift(self, x_t: torch.Tensor, t: torch.Tensor,
                  v_theta: torch.Tensor) -> torch.Tensor:
        """Compute SDE drift: f(x,t) = v_θ + σ²/(2t)·(x + (1-t)·v_θ)."""
        t_expanded = t.view(-1, 1, 1, 1)
        sigma_sq = self.sigma_t_sq(t).view(-1, 1, 1, 1)
        correction = (sigma_sq / (2.0 * t_expanded.clamp(min=1e-4))) * (
            x_t + (1.0 - t_expanded) * v_theta
        )
        return v_theta + correction

    def sde_step(self, x_t: torch.Tensor, t: torch.Tensor,
                 dt: float, v_theta: torch.Tensor,
                 noise: Optional[torch.Tensor] = None,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Euler-Maruyama SDE step: x_{t+dt} = x_t + f·dt + σ·√dt·z."""
        if noise is None:
            noise = torch.randn_like(x_t)
        drift = self.sde_drift(x_t, t, v_theta)
        sigma = self.sigma_t(t).view(-1, 1, 1, 1)
        x_next = x_t + drift * dt + sigma * math.sqrt(abs(dt)) * noise
        return x_next, noise

    def ode_step(self, x_t: torch.Tensor, dt: float,
                 v_theta: torch.Tensor) -> torch.Tensor:
        """Deterministic ODE (Euler) step: x_{t+dt} = x_t + v_θ·dt."""
        return x_t + v_theta * dt

    # ==================== LOG PROBABILITY ====================

    def compute_step_logprob(self, x_t: torch.Tensor, x_next: torch.Tensor,
                             t: torch.Tensor, dt: float,
                             v_theta: torch.Tensor) -> torch.Tensor:
        """Compute log π_θ(x_{t+dt} | x_t) for one SDE step.

        Following Bagel: average over spatial dims, drop normalization constant
        (it cancels in the importance ratio).

        Returns:
            [B] per-sample log probability (averaged over dimensions).
        """
        drift = self.sde_drift(x_t, t, v_theta)
        sigma = self.sigma_t(t).view(-1, 1, 1, 1)

        mean = x_t + drift * dt
        std = sigma * math.sqrt(abs(dt))  # [B, 1, 1, 1]

        diff = x_next - mean
        # Average over spatial dims (like Bagel), not sum
        # Normalization constant cancels in importance ratio
        log_prob = -0.5 * (diff ** 2 / std.clamp(min=1e-8) ** 2).mean(dim=[1, 2, 3])

        return log_prob  # [B]

    def compute_step_kl(self, x_t: torch.Tensor, t: torch.Tensor, dt: float,
                        v_theta: torch.Tensor,
                        v_ref: torch.Tensor) -> torch.Tensor:
        """Compute per-step KL divergence (like Bagel's KL penalty).

        KL = ||mean_θ - mean_ref||² / (2·var)

        Returns:
            [B] per-step KL divergence.
        """
        sigma = self.sigma_t(t).view(-1, 1, 1, 1)
        var = (sigma ** 2) * abs(dt)  # [B, 1, 1, 1]

        # Compute mean difference
        drift_theta = self.sde_drift(x_t, t, v_theta)
        drift_ref = self.sde_drift(x_t, t, v_ref)
        mean_diff = (drift_theta - drift_ref) * dt

        kl = (mean_diff ** 2 / (2.0 * var.clamp(min=1e-8))).mean(dim=[1, 2, 3])
        return kl  # [B]

    # ==================== TIMESTEP SCHEDULE ====================

    def make_timestep_schedule(self, num_steps: int,
                               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create timestep schedule with optional time shifting.

        Returns:
            (timesteps [num_steps+1], dt_values [num_steps])
        """
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
        if self.config.do_shift and self.config.time_shifting_factor:
            factor = self.config.time_shifting_factor
            timesteps = timesteps / (timesteps + factor - factor * timesteps)
        dt_values = timesteps[1:] - timesteps[:-1]
        return timesteps, dt_values

    def select_sde_window(self, num_steps: int) -> Tuple[int, int]:
        """Select contiguous SDE window (like Bagel's sde_window).

        Returns:
            (sde_begin, sde_end) step indices. Steps in [begin, end) use SDE.
        """
        ws = min(self.config.sde_window_size, num_steps)
        lo, hi = self.config.sde_window_range
        lo = max(0, min(lo, num_steps - ws))
        hi = max(lo, min(hi, num_steps - ws))
        begin = random.randint(lo, hi) if hi > lo else lo
        end = begin + ws
        return begin, end

    # ==================== HYBRID ODE/SDE TRAJECTORY ====================

    @torch.no_grad()
    def generate_trajectory(
        self,
        model_fn: Callable,
        z: torch.Tensor,
        num_steps: int,
    ) -> Dict[str, any]:
        """Generate hybrid ODE/SDE trajectory (like Bagel).

        Most steps are deterministic ODE. A random contiguous window uses
        stochastic SDE. Only SDE window data is recorded for training.

        Args:
            model_fn: (x_t, t) -> v_theta velocity prediction.
            z: [B, C, H, W] initial noise.
            num_steps: Total denoising steps.

        Returns:
            Dict with:
                'final': [B, C, H, W] final denoised latent.
                'sde_steps': list of dicts, one per SDE window step, each with:
                    'x_t': [B,C,H,W] state before step (detached)
                    'x_next': [B,C,H,W] state after step (detached)
                    'old_logprob': [B] log prob under sampling policy (detached)
                    't': scalar timestep
                    'dt': scalar step size
                'sde_window': (begin, end) step indices
        """
        device = z.device
        B = z.shape[0]

        timesteps, dt_values = self.make_timestep_schedule(num_steps, device)
        sde_begin, sde_end = self.select_sde_window(num_steps)

        sde_steps = []
        x_t = z

        for i in range(num_steps):
            t_i = timesteps[i].expand(B)
            dt_i = dt_values[i].item()

            v_theta = model_fn(x_t, t_i)

            if sde_begin <= i < sde_end:
                # SDE step: stochastic, record for training
                noise = torch.randn_like(x_t)
                x_next, _ = self.sde_step(x_t, t_i, dt_i, v_theta, noise=noise)

                old_lp = self.compute_step_logprob(
                    x_t, x_next, t_i, dt_i, v_theta
                )

                sde_steps.append({
                    'x_t': x_t.detach(),
                    'x_next': x_next.detach(),
                    'old_logprob': old_lp.detach(),
                    't': timesteps[i].item(),
                    'dt': dt_i,
                })
            else:
                # ODE step: deterministic
                x_next = self.ode_step(x_t, dt_i, v_theta)

            x_t = x_next.detach()

        return {
            'final': x_t,
            'sde_steps': sde_steps,
            'sde_window': (sde_begin, sde_end),
        }

    # ==================== PER-STEP GRPO LOSS ====================

    def compute_per_step_loss(
        self,
        model_fn: Callable,
        sde_step_data: Dict,
        advantage: torch.Tensor,
        ref_model_fn: Optional[Callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute GRPO loss for a single SDE step (like Bagel per-timestep backward).

        Args:
            model_fn: (x_t, t) -> v_theta, current policy (with gradient).
            sde_step_data: One element from trajectory['sde_steps'].
            advantage: [B] advantage value for this sample.
            ref_model_fn: Optional reference model for KL penalty.

        Returns:
            Dict with 'loss', 'metrics'.
        """
        x_t = sde_step_data['x_t']        # detached
        x_next = sde_step_data['x_next']  # detached
        old_lp = sde_step_data['old_logprob']  # detached
        device = x_t.device
        B = x_t.shape[0]

        t_val = sde_step_data['t']
        dt_val = sde_step_data['dt']
        t_tensor = torch.full((B,), t_val, device=device)

        # Recompute velocity with gradient
        v_theta = model_fn(x_t, t_tensor)
        new_lp = self.compute_step_logprob(x_t, x_next, t_tensor, dt_val, v_theta)

        # PPO-clip importance ratio
        log_ratio = (new_lp - old_lp).clamp(-10.0, 10.0)
        ratio = torch.exp(log_ratio)

        clipped_ratio = ratio.clamp(
            1.0 - self.config.epsilon, 1.0 + self.config.epsilon
        )

        # Pessimistic bound (like Bagel: max of unclipped/clipped negative loss)
        surr1 = -advantage * ratio
        surr2 = -advantage * clipped_ratio
        policy_loss = torch.max(surr1, surr2).mean()

        # KL penalty
        kl_loss = torch.tensor(0.0, device=device)
        if ref_model_fn is not None and self.config.beta > 0:
            with torch.no_grad():
                v_ref = ref_model_fn(x_t, t_tensor)
            kl = self.compute_step_kl(x_t, t_tensor, dt_val, v_theta, v_ref)
            kl_loss = self.config.beta * kl.mean()

        total_loss = policy_loss + kl_loss

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.config.epsilon).float().mean()

        return {
            'loss': total_loss,
            'metrics': {
                'clip_fraction': clip_frac.item(),
                'mean_ratio': ratio.mean().item(),
                'approx_kl': (log_ratio ** 2).mean().item() * 0.5,
                'policy_loss': policy_loss.item(),
                'kl_loss': kl_loss.item(),
            },
        }

    # ==================== ADVANTAGE COMPUTATION ====================

    @staticmethod
    def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
        """Group-relative advantages: Â = (R - mean) / std."""
        if rewards.numel() <= 1:
            return torch.zeros_like(rewards)
        mean_r = rewards.mean()
        std_r = rewards.std().clamp(min=1e-8)
        return (rewards - mean_r) / std_r
