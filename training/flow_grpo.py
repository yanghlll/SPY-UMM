"""
Flow-GRPO: Group Relative Policy Optimization for Flow Matching.

Implements the ODE→SDE conversion from Flow-GRPO (arXiv 2505.05470) to enable
proper policy gradient optimization on Show-o2's continuous flow matching head.

Key ideas:
1. Convert deterministic ODE (dx = v_θ dt) to stochastic SDE that preserves
   marginal distributions but enables log-probability computation.
2. The converted SDE transition π_θ(x_{t-1}|x_t, c) is an isotropic Gaussian,
   allowing closed-form importance ratios for PPO-clip.
3. Flow-GRPO-Fast: train on only 1-2 randomly selected SDE steps per trajectory
   for 5-10x speedup.

Reference: Flow-GRPO (arXiv 2505.05470, NeurIPS 2025)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FlowGRPOConfig:
    """Configuration for Flow-GRPO training."""
    # SDE noise schedule: σ_t = a * sqrt(t / (1-t))
    sde_noise_scale: float = 0.7
    # GRPO hyperparameters
    group_size: int = 4          # G: number of images per prompt
    epsilon: float = 0.2         # PPO clip parameter
    beta: float = 0.01           # KL penalty coefficient
    # Denoising steps
    num_train_steps: int = 10    # T: denoising steps for training (vs 50 for inference)
    num_inference_steps: int = 50
    # Flow-GRPO-Fast: use only a subset of SDE steps
    fast_mode: bool = False
    sde_window_size: int = 2     # number of SDE steps to train on in fast mode
    sde_window_range: Tuple[float, float] = (0.1, 0.9)  # t range for SDE window
    # Time shifting (from Show-o2's transport)
    do_shift: bool = True
    time_shifting_factor: float = 3.0


class FlowGRPO(nn.Module):
    """Flow-GRPO: GRPO for continuous flow matching models.

    Converts Show-o2's deterministic ODE sampling into a stochastic SDE,
    computes per-step log probabilities, and applies PPO-clip GRPO loss.
    """

    def __init__(self, config: FlowGRPOConfig):
        super().__init__()
        self.config = config
        self.a = config.sde_noise_scale

    # ==================== NOISE SCHEDULE ====================

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Compute SDE noise schedule: σ_t = a * sqrt(t / (1-t)).

        Args:
            t: [B] or scalar, timestep in [eps, 1-eps].

        Returns:
            σ_t values, same shape as t.
        """
        # Clamp to avoid division by zero
        t_clamped = t.clamp(min=1e-4, max=1.0 - 1e-4)
        return self.a * torch.sqrt(t_clamped / (1.0 - t_clamped))

    def sigma_t_sq(self, t: torch.Tensor) -> torch.Tensor:
        """σ_t² = a² * t / (1-t)."""
        t_clamped = t.clamp(min=1e-4, max=1.0 - 1e-4)
        return self.a ** 2 * t_clamped / (1.0 - t_clamped)

    # ==================== ODE → SDE CONVERSION ====================

    def sde_drift(self, x_t: torch.Tensor, t: torch.Tensor,
                  v_theta: torch.Tensor) -> torch.Tensor:
        """Compute the drift term of the converted SDE.

        The SDE is:
            dx_t = f(x_t, t) dt + σ_t dw
        where:
            f(x_t, t) = v_θ(x_t, t) + (σ_t² / (2t)) * (x_t + (1-t) * v_θ(x_t, t))

        Args:
            x_t: [B, C, H, W] current state.
            t: [B] timestep (scalar broadcast or per-sample).
            v_theta: [B, C, H, W] velocity prediction from model.

        Returns:
            [B, C, H, W] SDE drift.
        """
        t_expanded = t.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        sigma_sq = self.sigma_t_sq(t).view(-1, 1, 1, 1)

        # Extra drift from SDE conversion
        correction = (sigma_sq / (2.0 * t_expanded.clamp(min=1e-4))) * (
            x_t + (1.0 - t_expanded) * v_theta
        )

        return v_theta + correction

    def sde_step(self, x_t: torch.Tensor, t: torch.Tensor,
                 dt: float, v_theta: torch.Tensor,
                 noise: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Euler-Maruyama SDE step.

        x_{t+dt} = x_t + f(x_t, t) * dt + σ_t * sqrt(dt) * z
        where z ~ N(0, I).

        Args:
            x_t: [B, C, H, W] current state.
            t: [B] current timestep.
            dt: scalar step size.
            v_theta: [B, C, H, W] velocity prediction.
            noise: [B, C, H, W] optional pre-sampled noise.

        Returns:
            (x_{t+dt}, noise_used): next state and the noise used.
        """
        if noise is None:
            noise = torch.randn_like(x_t)

        drift = self.sde_drift(x_t, t, v_theta)
        sigma = self.sigma_t(t).view(-1, 1, 1, 1)

        x_next = x_t + drift * dt + sigma * math.sqrt(abs(dt)) * noise

        return x_next, noise

    # ==================== LOG PROBABILITY ====================

    def compute_step_logprob(self, x_t: torch.Tensor, x_next: torch.Tensor,
                             t: torch.Tensor, dt: float,
                             v_theta: torch.Tensor) -> torch.Tensor:
        """Compute log π_θ(x_{t+dt} | x_t, c) for one SDE step.

        Since π_θ(x_{t+dt} | x_t) is Gaussian:
            mean = x_t + f(x_t, t) * dt
            std = σ_t * sqrt(dt)

        log p = -0.5 * ||x_{t+dt} - mean||² / var - 0.5 * D * log(2π * var)

        Args:
            x_t: [B, C, H, W] current state.
            x_next: [B, C, H, W] next state.
            t: [B] current timestep.
            dt: scalar step size.
            v_theta: [B, C, H, W] velocity prediction at (x_t, t).

        Returns:
            [B] log probability of the transition.
        """
        drift = self.sde_drift(x_t, t, v_theta)
        sigma = self.sigma_t(t).view(-1, 1, 1, 1)

        mean = x_t + drift * dt
        var = (sigma ** 2) * abs(dt)  # [B, 1, 1, 1]

        # Per-element log prob
        diff = x_next - mean
        D = diff[0].numel()  # total dimensions per sample
        log_prob = -0.5 * (diff ** 2 / var.clamp(min=1e-8)).sum(dim=[1, 2, 3])
        log_prob = log_prob - 0.5 * D * torch.log(2 * math.pi * var.squeeze().clamp(min=1e-8))

        return log_prob  # [B]

    def compute_step_kl(self, t: torch.Tensor, dt: float,
                        v_theta: torch.Tensor,
                        v_ref: torch.Tensor) -> torch.Tensor:
        """Compute per-step KL divergence D_KL(π_θ || π_ref).

        D_KL = (dt/2) * ((σ_t(1-t))/(2t) + 1/σ_t)² * ||v_θ - v_ref||²

        Args:
            t: [B] current timestep.
            dt: scalar step size.
            v_theta: [B, C, H, W] current policy velocity.
            v_ref: [B, C, H, W] reference policy velocity.

        Returns:
            [B] per-step KL divergence.
        """
        t_expanded = t.view(-1, 1, 1, 1)
        sigma = self.sigma_t(t).view(-1, 1, 1, 1)

        # KL coefficient
        coeff = (sigma * (1.0 - t_expanded)) / (2.0 * t_expanded.clamp(min=1e-4)) + 1.0 / sigma.clamp(min=1e-8)

        velocity_diff_sq = (v_theta - v_ref) ** 2
        kl = (abs(dt) / 2.0) * (coeff ** 2) * velocity_diff_sq

        return kl.sum(dim=[1, 2, 3])  # [B]

    # ==================== SDE TRAJECTORY GENERATION ====================

    def generate_sde_trajectory(
        self,
        model_fn,
        z: torch.Tensor,
        num_steps: int,
        model_kwargs: Optional[dict] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """Generate a complete SDE trajectory with stored states and noise.

        This is used during rollout to generate images while recording
        all intermediate states needed for log-prob computation.

        Args:
            model_fn: Callable (x_t, t) -> v_theta. The model's velocity prediction.
            z: [B, C, H, W] initial noise x_0 ~ N(0, I).
            num_steps: Number of denoising steps.
            model_kwargs: Additional kwargs passed to model_fn.

        Returns:
            Dict with:
                'states': List of [B, C, H, W] states at each timestep (len = num_steps + 1)
                'noises': List of [B, C, H, W] noise used at each step (len = num_steps)
                'velocities': List of [B, C, H, W] velocity predictions (len = num_steps)
                'timesteps': [num_steps] tensor of timestep values
                'final': [B, C, H, W] final denoised image
        """
        if model_kwargs is None:
            model_kwargs = {}

        device = z.device
        B = z.shape[0]

        # Create timestep schedule: t goes from 0 to 1
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

        # Apply time shifting if configured
        if self.config.do_shift and self.config.time_shifting_factor:
            factor = self.config.time_shifting_factor
            timesteps = timesteps / (timesteps + factor - factor * timesteps)

        dt_values = timesteps[1:] - timesteps[:-1]  # [num_steps]

        states = [z]
        noises = []
        velocities = []

        x_t = z
        for i in range(num_steps):
            t_i = timesteps[i].expand(B)
            dt_i = dt_values[i].item()

            # Get velocity prediction
            v_theta = model_fn(x_t, t_i, **model_kwargs)
            velocities.append(v_theta.detach())

            # SDE step
            noise_i = torch.randn_like(x_t)
            x_next, _ = self.sde_step(x_t, t_i, dt_i, v_theta, noise=noise_i)

            states.append(x_next.detach())
            noises.append(noise_i)
            x_t = x_next.detach()

        return {
            'states': states,
            'noises': noises,
            'velocities': velocities,
            'timesteps': timesteps,
            'final': states[-1],
        }

    # ==================== LOG PROB OVER TRAJECTORY ====================

    def compute_trajectory_logprob(
        self,
        model_fn,
        trajectory: Dict[str, List[torch.Tensor]],
        model_kwargs: Optional[dict] = None,
        step_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Compute total log probability over a trajectory (or subset of steps).

        Args:
            model_fn: Callable (x_t, t) -> v_theta.
            trajectory: Output from generate_sde_trajectory().
            model_kwargs: Additional kwargs for model_fn.
            step_indices: If given, only compute log prob for these steps
                         (for Flow-GRPO-Fast). If None, compute all steps.

        Returns:
            [B] total log probability summed over selected steps.
        """
        if model_kwargs is None:
            model_kwargs = {}

        states = trajectory['states']
        timesteps = trajectory['timesteps']
        dt_values = timesteps[1:] - timesteps[:-1]

        num_steps = len(states) - 1
        if step_indices is None:
            step_indices = list(range(num_steps))

        B = states[0].shape[0]
        total_logprob = torch.zeros(B, device=states[0].device)

        for i in step_indices:
            x_t = states[i]
            x_next = states[i + 1]
            t_i = timesteps[i].expand(B)
            dt_i = dt_values[i].item()

            # Recompute velocity (with gradient)
            v_theta = model_fn(x_t, t_i, **model_kwargs)
            step_lp = self.compute_step_logprob(x_t, x_next, t_i, dt_i, v_theta)
            total_logprob = total_logprob + step_lp

        return total_logprob

    def compute_trajectory_kl(
        self,
        model_fn,
        ref_model_fn,
        trajectory: Dict[str, List[torch.Tensor]],
        model_kwargs: Optional[dict] = None,
        step_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Compute total KL divergence over trajectory between current and ref policy.

        Args:
            model_fn: Current policy velocity prediction.
            ref_model_fn: Reference policy velocity prediction.
            trajectory: Output from generate_sde_trajectory().
            model_kwargs: Additional kwargs for model functions.
            step_indices: Subset of steps for Fast mode.

        Returns:
            [B] total KL divergence.
        """
        if model_kwargs is None:
            model_kwargs = {}

        states = trajectory['states']
        timesteps = trajectory['timesteps']
        dt_values = timesteps[1:] - timesteps[:-1]

        num_steps = len(states) - 1
        if step_indices is None:
            step_indices = list(range(num_steps))

        B = states[0].shape[0]
        total_kl = torch.zeros(B, device=states[0].device)

        for i in step_indices:
            x_t = states[i]
            t_i = timesteps[i].expand(B)
            dt_i = dt_values[i].item()

            v_theta = model_fn(x_t, t_i, **model_kwargs)
            with torch.no_grad():
                v_ref = ref_model_fn(x_t, t_i, **model_kwargs)

            step_kl = self.compute_step_kl(t_i, dt_i, v_theta, v_ref)
            total_kl = total_kl + step_kl

        return total_kl

    # ==================== GRPO LOSS ====================

    def select_fast_steps(self, num_steps: int,
                          device: torch.device) -> List[int]:
        """Select random SDE step indices for Flow-GRPO-Fast.

        Args:
            num_steps: Total number of denoising steps.
            device: Device for random number generation.

        Returns:
            List of step indices to train on.
        """
        t_min, t_max = self.config.sde_window_range
        # Map t range to step indices
        step_min = max(1, int(t_min * num_steps))
        step_max = min(num_steps - 1, int(t_max * num_steps))

        if step_max <= step_min:
            return [step_min]

        # Randomly select sde_window_size steps within the range
        n_select = min(self.config.sde_window_size, step_max - step_min)
        perm = torch.randperm(step_max - step_min, device=device)[:n_select]
        indices = (perm + step_min).sort().values.tolist()

        return indices

    def compute_grpo_loss(
        self,
        current_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        kl_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the Flow-GRPO loss.

        J = (1/G) Σᵢ [min(rᵢ·Âᵢ, clip(rᵢ, 1-ε, 1+ε)·Âᵢ) - β·D_KL]

        Note: advantage is identical across all timesteps for a given image
        (terminal reward only, normalized within the group).

        Args:
            current_logprobs: [G] total log prob from current policy.
            old_logprobs: [G] total log prob from behavior (old) policy.
            advantages: [G] group-relative advantages.
            kl_values: [G] optional KL divergence values.

        Returns:
            Dict with 'loss', 'policy_loss', 'kl_loss', 'metrics'.
        """
        # Importance ratio (in log space for stability, then exponentiate)
        log_ratio = current_logprobs - old_logprobs
        # Clamp for numerical stability
        log_ratio = log_ratio.clamp(min=-10.0, max=10.0)
        ratio = torch.exp(log_ratio)

        # PPO-clip
        clipped_ratio = ratio.clamp(1.0 - self.config.epsilon,
                                    1.0 + self.config.epsilon)

        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty
        kl_loss = torch.tensor(0.0, device=current_logprobs.device)
        if self.config.beta > 0 and kl_values is not None:
            kl_loss = self.config.beta * kl_values.mean()

        total_loss = policy_loss + kl_loss

        # Metrics
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.config.epsilon).float().mean()
            approx_kl = (log_ratio ** 2).mean() * 0.5  # approximate KL from ratio

        return {
            'loss': total_loss,
            'policy_loss': policy_loss.detach(),
            'kl_loss': kl_loss.detach(),
            'metrics': {
                'clip_fraction': clip_fraction.item(),
                'mean_ratio': ratio.mean().item(),
                'mean_advantage': advantages.mean().item(),
                'std_advantage': advantages.std().item() if advantages.numel() > 1 else 0.0,
                'approx_kl': approx_kl.item(),
                'mean_logprob': current_logprobs.mean().item(),
            }
        }

    @staticmethod
    def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-relative advantages from terminal rewards.

        Â_i = (R_i - mean(R)) / std(R)

        The advantage is identical across all timesteps for a given image
        (terminal reward only).

        Args:
            rewards: [G] reward values for each generated image.

        Returns:
            [G] normalized advantages.
        """
        if rewards.numel() <= 1:
            return torch.zeros_like(rewards)

        mean_r = rewards.mean()
        std_r = rewards.std().clamp(min=1e-8)
        return (rewards - mean_r) / std_r


class FlowGRPOTrainer:
    """High-level trainer that orchestrates Flow-GRPO training on Show-o2.

    Manages the full pipeline:
    1. Generate G images per prompt using SDE sampling
    2. Score images with reward model
    3. Compute advantages
    4. Recompute log probs with gradient and apply GRPO loss
    """

    def __init__(self, flow_grpo: FlowGRPO, config: FlowGRPOConfig):
        self.flow_grpo = flow_grpo
        self.config = config

    def train_step(
        self,
        model_fn,
        ref_model_fn,
        prompts: List[str],
        reward_fn,
        prepare_input_fn,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Execute one Flow-GRPO training step.

        Args:
            model_fn: Callable (x_t, t, **kwargs) -> v_theta, current policy.
            ref_model_fn: Callable for reference policy (or None to skip KL).
            prompts: List of text prompts for this batch.
            reward_fn: Callable (images: List[PIL.Image]) -> [G] rewards.
            prepare_input_fn: Callable (prompts) -> (z_init, model_kwargs)
                             that prepares the initial noise and model inputs.
            device: Training device.
            dtype: Training dtype.

        Returns:
            Dict with 'loss', 'rewards', 'metrics'.
        """
        G = self.config.group_size
        T = self.config.num_train_steps

        # ---- Phase 1: Generate G trajectories (no_grad) ----
        all_trajectories = []
        all_old_logprobs = []
        all_rewards = []

        for prompt in prompts:
            prompt_trajectories = []
            prompt_old_lps = []

            for g in range(G):
                z_init, model_kwargs = prepare_input_fn([prompt])
                z_init = z_init.to(device).to(dtype)

                with torch.no_grad():
                    trajectory = self.flow_grpo.generate_sde_trajectory(
                        model_fn, z_init, num_steps=T,
                        model_kwargs=model_kwargs,
                    )
                    prompt_trajectories.append(trajectory)

                    # Compute old log probs
                    if self.config.fast_mode:
                        step_indices = self.flow_grpo.select_fast_steps(T, device)
                    else:
                        step_indices = None

                    old_lp = self.flow_grpo.compute_trajectory_logprob(
                        model_fn, trajectory,
                        model_kwargs=model_kwargs,
                        step_indices=step_indices,
                    )
                    prompt_old_lps.append(old_lp)

            all_trajectories.append(prompt_trajectories)
            all_old_logprobs.append(torch.cat(prompt_old_lps, dim=0))  # [G]

            # Score the final images
            final_images = [traj['final'] for traj in prompt_trajectories]
            rewards = reward_fn(final_images, prompt)
            all_rewards.append(rewards)

        # ---- Phase 2: Compute GRPO loss (with gradient) ----
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        all_metrics = {}

        for p_idx, (prompt_trajs, old_lps, rewards) in enumerate(
            zip(all_trajectories, all_old_logprobs, all_rewards)
        ):
            rewards_t = rewards.to(device).float()
            advantages = FlowGRPO.compute_advantages(rewards_t)

            # Recompute log probs with gradient
            current_lps = []
            kl_values = []

            for g, traj in enumerate(prompt_trajs):
                _, model_kwargs = prepare_input_fn([prompts[p_idx]])

                if self.config.fast_mode:
                    step_indices = self.flow_grpo.select_fast_steps(T, device)
                else:
                    step_indices = None

                cur_lp = self.flow_grpo.compute_trajectory_logprob(
                    model_fn, traj,
                    model_kwargs=model_kwargs,
                    step_indices=step_indices,
                )
                current_lps.append(cur_lp)

                # KL divergence
                if ref_model_fn is not None and self.config.beta > 0:
                    kl = self.flow_grpo.compute_trajectory_kl(
                        model_fn, ref_model_fn, traj,
                        model_kwargs=model_kwargs,
                        step_indices=step_indices,
                    )
                    kl_values.append(kl)

            current_logprobs = torch.cat(current_lps, dim=0)  # [G]
            kl_tensor = torch.cat(kl_values, dim=0) if kl_values else None

            grpo_result = self.flow_grpo.compute_grpo_loss(
                current_logprobs, old_lps.detach(),
                advantages, kl_tensor,
            )

            total_loss = total_loss + grpo_result['loss']

            # Accumulate metrics
            for k, v in grpo_result['metrics'].items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)

        # Average over prompts
        total_loss = total_loss / len(prompts)

        # Average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
        avg_metrics['mean_reward'] = torch.cat(
            [r.float() for r in all_rewards]
        ).mean().item()

        return {
            'loss': total_loss,
            'rewards': all_rewards,
            'metrics': avg_metrics,
        }
