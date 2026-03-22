"""
GRPO for Voting Phase.

Applies Group Relative Policy Optimization to the voting/understanding
phase of the spy game. Since Show-o2's understanding mode is autoregressive
text generation (via mmu_generate()), standard GRPO works directly.

Adapted from Vision-Zero's GRPO trainer (grpo_trainer.py).
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class VotingGRPO:
    """GRPO optimizer for the voting phase.

    Generates multiple vote completions, scores them with rewards,
    and updates the policy using PPO-clip objective.
    """

    def __init__(self, beta: float = 0.0, epsilon: float = 0.2):
        """
        Args:
            beta: KL penalty coefficient (0 = no KL penalty).
            epsilon: PPO clipping parameter.
        """
        self.beta = beta
        self.epsilon = epsilon

    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """Compute group-relative advantages.

        Normalizes rewards within the generation group.
        """
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        if rewards_t.numel() <= 1:
            return torch.zeros_like(rewards_t)

        mean_r = rewards_t.mean()
        std_r = rewards_t.std().clamp(min=1e-8)
        return (rewards_t - mean_r) / std_r

    def compute_loss(self,
                     current_logprobs: torch.Tensor,
                     old_logprobs: torch.Tensor,
                     advantages: torch.Tensor,
                     completion_mask: torch.Tensor,
                     ref_logprobs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute GRPO loss for voting completions.

        Args:
            current_logprobs: [G, L] per-token log probs from current policy.
            old_logprobs: [G, L] per-token log probs from behavior policy.
            advantages: [G] advantage values for each generation.
            completion_mask: [G, L] mask for valid completion tokens.
            ref_logprobs: [G, L] per-token log probs from reference policy (for KL).

        Returns:
            Dict with 'loss', 'metrics'.
        """
        G = current_logprobs.shape[0]  # num_generations

        # Compute per-token importance ratios
        ratio = torch.exp(current_logprobs - old_logprobs)

        # Per-token clipped ratio
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

        # PPO-clip objective (per-token)
        advantages_expanded = advantages.unsqueeze(-1)  # [G, 1]
        surr1 = ratio * advantages_expanded
        surr2 = clipped_ratio * advantages_expanded

        # Take minimum (pessimistic bound)
        per_token_loss = -torch.min(surr1, surr2)

        # Apply mask and compute mean
        masked_loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
        policy_loss = masked_loss.mean()

        # KL penalty (optional)
        kl_loss = torch.tensor(0.0, device=current_logprobs.device)
        if self.beta > 0 and ref_logprobs is not None:
            # KL ≈ E[(log π - log π_ref)²] / 2 (always non-negative)
            kl_div = ((current_logprobs - ref_logprobs) ** 2) * completion_mask
            kl_per_sample = kl_div.sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1) * 0.5
            kl_loss = self.beta * kl_per_sample.mean()

        total_loss = policy_loss + kl_loss

        # Compute metrics
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.epsilon).float()
            clip_fraction = (clip_fraction * completion_mask).sum() / completion_mask.sum().clamp(min=1)

        return {
            'loss': total_loss,
            'policy_loss': policy_loss.detach(),
            'kl_loss': kl_loss.detach(),
            'metrics': {
                'clip_fraction': clip_fraction.item(),
                'mean_advantage': advantages.mean().item(),
                'mean_ratio': ratio.mean().item(),
            }
        }


def generate_and_score_votes(spy_wrapper, image_latents_list, voting_prompt,
                             correct_spy, num_generations=4,
                             max_tokens=512, temperature=1.0):
    """Generate multiple vote completions and score them.

    Args:
        spy_wrapper: Showo2SpyWrapper instance.
        image_latents_list: List of image latents per player.
        voting_prompt: Text prompt for voting.
        correct_spy: The actual spy player number.
        num_generations: Number of completions to generate (G).
        max_tokens: Max tokens per completion.
        temperature: Sampling temperature.

    Returns:
        Dict with 'responses', 'rewards', 'format_rewards'.
    """
    from .rewards import vote_accuracy_reward, vote_format_reward

    responses = []
    accuracy_rewards = []
    format_rewards = []

    for _ in range(num_generations):
        response = spy_wrapper.judge_vote(
            image_latents_list,
            voting_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        responses.append(response)

        acc_r = vote_accuracy_reward(response, correct_spy)
        fmt_r = vote_format_reward(response)
        accuracy_rewards.append(acc_r)
        format_rewards.append(fmt_r)

    total_rewards = [a + f for a, f in zip(accuracy_rewards, format_rewards)]

    return {
        'responses': responses,
        'accuracy_rewards': accuracy_rewards,
        'format_rewards': format_rewards,
        'total_rewards': total_rewards,
    }
