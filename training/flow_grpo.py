"""
Flow-GRPO: Group Relative Policy Optimization for Flow Matching.

Follows the Bagel Flow-GRPO reference implementation (yifan123/flow_grpo):
1. Hybrid ODE/SDE trajectory: most steps ODE, a random contiguous window uses SDE.
2. Per-timestep backward for memory efficiency.
3. SDE step formula, log-prob, KL, and PPO-clip loss match Bagel exactly.
4. All SDE math in float32 to avoid bf16 overflow (following Bagel).

Reference: Flow-GRPO (arXiv 2505.05470)
"""

import math
import random
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class FlowGRPOConfig:
    """Configuration for Flow-GRPO training."""
    sde_noise_scale: float = 0.7      # a in σ_t = a * sqrt(t/(1-t))
    group_size: int = 4               # G: number of games per training step
    clip_range: float = 0.2           # PPO clip (Bagel uses 1e-5; 0.2 is standard PPO)
    beta: float = 0.0                 # KL penalty (Bagel default: 0)
    adv_clip_max: float = 5.0         # Advantage clipping
    num_train_steps: int = 20         # T: total denoising steps
    num_inference_steps: int = 50
    sde_window_size: int = 3          # contiguous SDE steps (Bagel: 3)
    sde_window_range: Tuple[int, int] = (0, -1)  # window start range; -1 = T//2
    do_shift: bool = True
    time_shifting_factor: float = 3.0


class FlowGRPO(nn.Module):
    """Flow-GRPO for continuous flow matching models.

    All SDE computations done in float32 (Bagel: "bf16 can overflow here").
    """

    def __init__(self, config: FlowGRPOConfig):
        super().__init__()
        self.config = config
        self.a = config.sde_noise_scale

    # ==================== SDE MATH (all float32) ====================

    def _sde_step_with_logprob(
        self,
        v_theta: torch.Tensor,   # [B, C, H, W] velocity prediction
        x_t: torch.Tensor,       # [B, C, H, W] current state
        t: torch.Tensor,         # [B] timestep
        dt: float,               # scalar step size
        x_next: Optional[torch.Tensor] = None,  # if provided, compute log_prob of this transition
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """SDE step + log-prob, matching Bagel's _sde_step_with_logprob exactly.

        Bagel formula (adapted from reverse-time t:1→0 to forward-time t:0→1):
            std_dev_t = sqrt(t / (1-t)) * noise_level
            mean = x_t * (1 + std² / (2t) * dt) + v_t * (1 + std²*(1-t)/(2t)) * dt
            x_next = mean + std_dev_t * sqrt(|dt|) * noise
            log_prob = -(x_next - mean)² / (2 * (std_dev_t * sqrt(|dt|))²)

        All math in float32 to avoid bf16 overflow.

        Returns:
            (x_next, log_prob, mean, std_dev_t)
        """
        # Cast to float32 (Bagel: "bf16 can overflow here")
        v_t = v_theta.float()
        x = x_t.float()

        t_exp = t.float().view(-1, 1, 1, 1)
        t_safe = t_exp.clamp(min=1e-4, max=1.0 - 1e-4)

        # σ_t = a * sqrt(t / (1-t))
        std_dev_t = self.a * torch.sqrt(t_safe / (1.0 - t_safe))  # [B,1,1,1]

        # SDE mean (matches Bagel exactly)
        mean = (x * (1.0 + std_dev_t**2 / (2.0 * t_safe) * dt)
                + v_t * (1.0 + std_dev_t**2 * (1.0 - t_safe) / (2.0 * t_safe)) * dt)

        if x_next is None:
            # Sampling: generate new x_next
            if noise is None:
                noise = torch.randn_like(x)
            x_next_f = mean + std_dev_t * math.sqrt(abs(dt)) * noise.float()
        else:
            x_next_f = x_next.float()

        # Log-prob: Gaussian log-likelihood (drop normalization constant)
        # Bagel: log_prob = -((prev_sample.detach() - mean)**2) / (2 * (std*sqrt(|dt|))**2)
        # then .mean() over all dims
        variance = (std_dev_t * math.sqrt(abs(dt))) ** 2
        log_prob = -((x_next_f.detach() - mean) ** 2) / (2.0 * variance.clamp(min=1e-8))
        log_prob = log_prob.mean(dim=[1, 2, 3])  # [B] (per-sample, not scalar like Bagel's B=1)

        return x_next_f.to(x_t.dtype), log_prob, mean, std_dev_t

    def ode_step(self, x_t: torch.Tensor, dt: float,
                 v_theta: torch.Tensor) -> torch.Tensor:
        """Deterministic ODE (Euler) step."""
        return x_t + v_theta * dt

    # ==================== TIMESTEP SCHEDULE ====================

    def make_timestep_schedule(self, num_steps: int,
                               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Timestep schedule 0→1 with optional time shifting."""
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
        if self.config.do_shift and self.config.time_shifting_factor:
            f = self.config.time_shifting_factor
            timesteps = timesteps / (timesteps + f - f * timesteps)
        dt_values = timesteps[1:] - timesteps[:-1]
        return timesteps, dt_values

    def select_sde_window(self, num_steps: int) -> Tuple[int, int]:
        """Select contiguous SDE window. Bagel: start in [lo, hi)."""
        ws = min(self.config.sde_window_size, num_steps)
        lo, hi = self.config.sde_window_range
        if hi < 0:
            hi = num_steps // 2
        lo = max(0, min(lo, num_steps - ws))
        hi = max(lo, min(hi, num_steps - ws))
        begin = random.randint(lo, hi) if hi > lo else lo
        return begin, begin + ws

    # ==================== HYBRID ODE/SDE TRAJECTORY ====================

    @torch.no_grad()
    def generate_trajectory(
        self,
        model_fn: Callable,
        z: torch.Tensor,
        num_steps: int,
        sde_window: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, any]:
        """Generate hybrid ODE/SDE trajectory (like Bagel).

        Returns dict with 'final', 'sde_steps', 'sde_window'.
        """
        device = z.device
        B = z.shape[0]
        timesteps, dt_values = self.make_timestep_schedule(num_steps, device)

        if sde_window is not None:
            sde_begin, sde_end = sde_window
        else:
            sde_begin, sde_end = self.select_sde_window(num_steps)

        sde_steps = []
        x_t = z

        for i in range(num_steps):
            t_i = timesteps[i].expand(B)
            dt_i = dt_values[i].item()
            v_theta = model_fn(x_t, t_i)

            if sde_begin <= i < sde_end:
                x_next, old_lp, _, _ = self._sde_step_with_logprob(
                    v_theta, x_t, t_i, dt_i
                )
                sde_steps.append({
                    'x_t': x_t.detach(),
                    'x_next': x_next.detach(),
                    'old_logprob': old_lp.detach(),
                    't': timesteps[i].item(),
                    'dt': dt_i,
                })
            else:
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
        advantages: torch.Tensor,
        ref_model_fn: Optional[Callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute GRPO loss for one SDE step, matching Bagel exactly.

        Bagel's per-timestep loss:
            ratio = exp(new_log_prob - old_log_prob)
            unclipped = -advantage * ratio
            clipped   = -advantage * clamp(ratio, 1-clip, 1+clip)
            policy_loss = mean(max(unclipped, clipped))
            kl_loss = mean((mean_θ - mean_ref)²) / (2 * std²)
            loss = policy_loss + beta * kl_loss
        """
        x_t = sde_step_data['x_t']
        x_next = sde_step_data['x_next']
        old_lp = sde_step_data['old_logprob']
        device = x_t.device
        B = x_t.shape[0]

        t_val = sde_step_data['t']
        dt_val = sde_step_data['dt']
        t_tensor = torch.full((B,), t_val, device=device)

        # Clamp advantages (Bagel: adv_clip_max)
        adv = advantages.clamp(-self.config.adv_clip_max, self.config.adv_clip_max)

        # Forward: recompute velocity with gradient, get new log_prob + mean
        v_theta = model_fn(x_t, t_tensor)
        _, new_lp, mean_theta, std_dev_t = self._sde_step_with_logprob(
            v_theta, x_t, t_tensor, dt_val, x_next=x_next
        )

        # PPO-clip (Bagel: torch.maximum for pessimistic bound)
        ratio = torch.exp((new_lp - old_lp).clamp(-10.0, 10.0))
        clip = self.config.clip_range
        unclipped_loss = -adv * ratio
        clipped_loss = -adv * ratio.clamp(1.0 - clip, 1.0 + clip)
        policy_loss = torch.maximum(unclipped_loss, clipped_loss).mean()

        # KL penalty (Bagel: ||mean_θ - mean_ref||² / (2 * std²))
        kl_loss = torch.tensor(0.0, device=device)
        if ref_model_fn is not None and self.config.beta > 0:
            with torch.no_grad():
                v_ref = ref_model_fn(x_t, t_tensor)
                _, _, mean_ref, _ = self._sde_step_with_logprob(
                    v_ref, x_t, t_tensor, dt_val, x_next=x_next
                )
            variance = (std_dev_t * math.sqrt(abs(dt_val))) ** 2
            kl = ((mean_theta - mean_ref) ** 2).mean() / (2.0 * variance.mean().clamp(min=1e-8))
            kl_loss = self.config.beta * kl

        total_loss = policy_loss + kl_loss

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > clip).float().mean()

        return {
            'loss': total_loss,
            'metrics': {
                'clip_fraction': clip_frac.item(),
                'mean_ratio': ratio.mean().item(),
                'approx_kl': ((new_lp - old_lp) ** 2).mean().item() * 0.5,
                'policy_loss': policy_loss.item(),
                'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            },
        }

    # ==================== ADVANTAGE COMPUTATION ====================

    @staticmethod
    def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
        """Group-relative advantages: Â = (R - mean) / std."""
        if rewards.numel() <= 1:
            return torch.zeros_like(rewards)
        return (rewards - rewards.mean()) / rewards.std().clamp(min=1e-8)
