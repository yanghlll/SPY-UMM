"""
Reward Functions for SPY-UMM.

Provides reward computation for both generation and voting phases.
Adapted from Vision-Zero's reward functions in clevr_spotdiff_generator.py
and grpo_jsonl.py.
"""

import re
from typing import List, Dict, Any, Optional


def vote_accuracy_reward(response: str, correct_spy: int) -> float:
    """Check if the vote correctly identifies the spy.

    Args:
        response: Model's text response containing the vote.
        correct_spy: The actual spy player number (1-indexed).

    Returns:
        1.0 if correct, -1.0 if incorrect, -0.5 for N/A, -1.0 for invalid.
    """
    vote = _extract_vote(response)
    if vote is None:
        return -1.0  # Invalid format
    if vote == "N/A":
        return -0.5  # Uncertain
    if vote == correct_spy:
        return 1.0  # Correct
    return -1.0  # Wrong


def vote_format_reward(response: str) -> float:
    """Check if the response follows the required format.

    Expected format: <think>...</think><answer>...</answer>

    Returns:
        Score from 0.0 to 1.0 based on format compliance.
    """
    score = 0.0

    # Check for think tags
    has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
    if has_think:
        score += 0.4

        # Check think content quality
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            if len(think_content) > 20:
                score += 0.1  # Meaningful reasoning

    # Check for answer tags
    has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL))
    if has_answer:
        score += 0.4

        # Check answer content
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            # Check it contains a number or N/A
            if re.search(r'\b[1-9]\b', answer_content) or answer_content.upper() in ('N/A', 'NA'):
                score += 0.1

    return score


def game_outcome_reward(spy_caught: bool, spy_player: int,
                        num_players: int) -> Dict[str, float]:
    """Compute game outcome rewards for the generation phase.

    These rewards flow back to weight the flow matching loss.

    Args:
        spy_caught: Whether the spy was identified by majority vote.
        spy_player: The spy's player ID.
        num_players: Total number of players.

    Returns:
        Dict with 'spy_reward' and 'civilian_reward'.
    """
    if spy_caught:
        # Spy was caught: civilians did well, spy did poorly
        return {
            'spy_reward': -1.0,
            'civilian_reward': 1.0,
        }
    else:
        # Spy escaped: spy did well, civilians did poorly
        return {
            'spy_reward': 1.0,
            'civilian_reward': -0.5,
        }


def compute_grpo_advantages(rewards: List[float],
                            epsilon: float = 1e-8) -> List[float]:
    """Compute GRPO-style advantages from a group of rewards.

    Normalizes rewards within the group (zero mean, unit variance).

    Args:
        rewards: List of reward values from multiple generations.
        epsilon: Small constant for numerical stability.

    Returns:
        List of advantage values.
    """
    if len(rewards) == 0:
        return []
    if len(rewards) == 1:
        return [0.0]

    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = (var_r + epsilon) ** 0.5

    return [(r - mean_r) / std_r for r in rewards]


def _extract_vote(response: str) -> Optional[Any]:
    """Extract vote from response text.

    Returns:
        int (player number), "N/A", or None (invalid).
    """
    # Try <answer></answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
    else:
        # Try boxed format
        boxed_match = re.search(r'\\\\?boxed\{(.*?)\}', response, re.DOTALL)
        if boxed_match:
            content = boxed_match.group(1).strip()
        else:
            return None

    if content.upper() in ("N/A", "NA"):
        return "N/A"

    numbers = re.findall(r'\b([1-9])\b', content)
    if numbers:
        return int(numbers[0])

    return None
