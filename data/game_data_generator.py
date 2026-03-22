"""
Game Data Generator for SPY-UMM.

Orchestrates the "Who's the Odd One Out?" spy game with text-to-image generation.
Adapted from Vision-Zero's CLEVRSpotDiffGenerator but reversed:
  - Input: text descriptions (instead of images)
  - Output: generated images (instead of text clues)
"""

import random
import re
import json
from typing import List, Dict, Any, Tuple, Optional

from .scene_description_generator import SceneDescriptionGenerator


class SpyGameDataGenerator:
    """Generates complete spy game instances for text-to-image training.

    Game flow:
    1. Generate text description pair (original vs modified)
    2. Assign spy player (gets modified description)
    3. Each player generates an image from their description
    4. Judge sees all images and votes on who is the spy
    """

    def __init__(self, num_players: int = 4,
                 num_objects_min: int = 3, num_objects_max: int = 6,
                 num_to_modify: int = 2):
        self.num_players = num_players
        self.scene_gen = SceneDescriptionGenerator(
            num_objects_min=num_objects_min,
            num_objects_max=num_objects_max,
            num_to_modify=num_to_modify
        )

        # Role advantage baselines for adaptive reward shaping (from Vision-Zero)
        self.b_spy = 0.0
        self.b_civ = 0.0
        self.alpha = 0.9  # EMA decay
        self.baseline_update_count = 0

    def generate_game(self, epoch: int, sample_idx: int) -> Dict[str, Any]:
        """Generate a complete game instance.

        Args:
            epoch: Current training epoch.
            sample_idx: Sample index within the epoch.

        Returns:
            Game data dictionary with all necessary information.
        """
        seed = epoch * 10000 + sample_idx

        # Generate text description pair
        original_desc, modified_desc, diff_metadata = self.scene_gen.generate_pair(seed)

        # Randomly assign spy player (1-indexed)
        rng = random.Random(seed + 1)
        spy_player = rng.randint(1, self.num_players)

        # Assign descriptions to players
        player_descriptions = []
        for pid in range(1, self.num_players + 1):
            if pid == spy_player:
                player_descriptions.append(modified_desc)
            else:
                player_descriptions.append(original_desc)

        return {
            "game_id": f"spy_epoch_{epoch}_sample_{sample_idx}",
            "epoch": epoch,
            "sample_idx": sample_idx,
            "num_players": self.num_players,
            "spy_player": spy_player,
            "player_descriptions": player_descriptions,
            "original_description": original_desc,
            "modified_description": modified_desc,
            "diff_metadata": diff_metadata,
        }

    def format_generation_prompt(self, game_data: Dict[str, Any],
                                 player_id: int) -> str:
        """Create the text prompt for image generation for a given player.

        Args:
            game_data: Game data from generate_game().
            player_id: 1-indexed player ID.

        Returns:
            Text prompt for Show-o2's t2i_generate().
        """
        desc = game_data["player_descriptions"][player_id - 1]
        spy_player = game_data["spy_player"]

        if player_id == spy_player:
            role_hint = (
                "You are the SPY. Your scene description is slightly different "
                "from the other players'. Generate an image that looks as similar "
                "as possible to what the original scene might look like, to avoid detection."
            )
        else:
            role_hint = (
                "You are a CIVILIAN. Generate an image that faithfully represents "
                "the scene description. Your image should be consistent with "
                "other civilian players' images."
            )

        prompt = (
            f"Generate a high-quality image of the following 3D rendered scene. "
            f"The scene contains geometric objects (cubes, spheres, cylinders) "
            f"with different colors, sizes, and materials on a flat surface.\n\n"
            f"Scene: {desc}"
        )

        return prompt

    def format_generation_prompt_simple(self, game_data: Dict[str, Any],
                                        player_id: int) -> str:
        """Generation prompt — just the scene description, nothing else.

        Generation is parallel and independent; each player only sees their
        own text prompt. No need for anti-copying instructions since players
        cannot see each other's images during generation.
        """
        desc = game_data["player_descriptions"][player_id - 1]
        return (
            f"A 3D rendered scene with geometric objects on a flat surface: {desc}"
        )

    def format_voting_prompt(self, game_data: Dict[str, Any],
                             player_id: int = None) -> str:
        """Create the voting/judge prompt for identifying the spy.

        The judge sees all generated images and must identify which player
        generated the odd image (from the different description).

        Args:
            game_data: Game data from generate_game().
            player_id: If None, creates a "God judge" prompt.
                       If specified, creates a player-specific voting prompt.

        Returns:
            Text prompt for Show-o2's mmu_generate().
        """
        num_players = game_data["num_players"]

        if player_id is None:
            # God judge perspective - sees all images
            context = f"""You are analyzing a game of "Who's the Odd One Out?" where {num_players} players each generated an image from a scene description. One player (the spy) received a slightly different description, so their generated image should look subtly different from the others.

All {num_players} generated images are shown to you. The spy's image may have different objects (different shapes or colors) at certain positions compared to the majority.

Your task:
1. Compare all {num_players} images carefully
2. Look for objects that appear different in one image compared to the others
3. Identify which player generated the odd image

Required output format:
<think>
Your step-by-step reasoning analyzing each image and comparing them.
</think>
<answer>
PLAYER_NUMBER (just the number, e.g., 1, 2, 3, or 4)
</answer>"""
        else:
            # Player-specific voting perspective
            spy_player = game_data["spy_player"]
            if player_id == spy_player:
                role_info = "Your role: SPY (your description was different)"
                instruction = (
                    "You know your image might look different. "
                    "Try to vote for someone else to avoid suspicion."
                )
            else:
                role_info = "Your role: CIVILIAN (your description was the same as most)"
                instruction = (
                    "Look for the image that seems inconsistent with the others. "
                    "Vote for the player you think is the spy."
                )

            context = f"""You are PLAYER {player_id} in a game of "Who's the Odd One Out?". Each player generated an image from a scene description. One player (the spy) had a different description.

{role_info}

All {num_players} generated images are shown. {instruction}

Required output format:
<think>
Your step-by-step reasoning.
</think>
<answer>
PLAYER_NUMBER or N/A
</answer>"""

        return context

    def extract_vote(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract vote from a judge/player response.

        Adapted from Vision-Zero's extract_vote_from_decision().
        """
        # Try <answer></answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
        else:
            # Try boxed format
            boxed_match = re.search(r'\\\\?boxed\{(.*?)\}', response, re.DOTALL)
            if boxed_match:
                answer_content = boxed_match.group(1).strip()
            else:
                return None

        # Check for N/A
        if answer_content.strip().upper() in ("N/A", "NA"):
            return {"voted_spy": "N/A", "reasoning": "Uncertain"}

        # Extract number
        numbers = re.findall(r'\b([1-9])\b', answer_content)
        if numbers:
            return {"voted_spy": int(numbers[0]), "reasoning": "Direct vote"}

        return None

    def calculate_game_rewards(self, game_data: Dict[str, Any],
                               player_votes: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate rewards for all players based on voting results.

        Adapted from Vision-Zero's calculate_game_rewards().

        Args:
            game_data: Game data dictionary.
            player_votes: List of vote dicts for each player.

        Returns:
            Dict with per-player rewards and game outcome info.
        """
        num_players = game_data["num_players"]
        spy_player = game_data["spy_player"]

        alpha_penalty = 0.6
        beta_bonus = 0.5

        # Count votes for each player
        vote_counts = {i: 0 for i in range(1, num_players + 1)}
        valid_votes = []

        for pid, vote_info in enumerate(player_votes, 1):
            if vote_info and "voted_spy" in vote_info:
                voted = vote_info["voted_spy"]
                if voted == "N/A":
                    valid_votes.append("N/A")
                elif isinstance(voted, int) and 1 <= voted <= num_players:
                    vote_counts[voted] += 1
                    valid_votes.append(voted)
                else:
                    valid_votes.append(None)
            else:
                valid_votes.append(None)

        rewards = {}

        for pid in range(1, num_players + 1):
            if pid == spy_player:
                # Spy reward
                votes_received = vote_counts[pid]
                stealth_score = 1.0 - votes_received / max(num_players - 1, 1)

                mislead_count = sum(
                    1.0 if v is not None and v != spy_player and v != "N/A"
                    else 0.5 if v == "N/A"
                    else 0.0
                    for v in valid_votes
                )
                misleading_bonus = beta_bonus * mislead_count / max(num_players - 1, 1)

                rewards[pid] = max(-1.0, min(1.0, stealth_score + misleading_bonus))
            else:
                # Civilian/detective reward
                vote_info = player_votes[pid - 1]
                if not vote_info or "voted_spy" not in vote_info:
                    rewards[pid] = -1.0
                    continue

                voted = vote_info["voted_spy"]
                case_solving = 1.0 if voted == spy_player else -1.0

                votes_received = vote_counts[pid]
                suspicion_penalty = -alpha_penalty * votes_received / max(num_players - 1, 1)

                rewards[pid] = max(-1.0, min(1.0, case_solving + suspicion_penalty))

        # Determine if spy was caught (majority vote)
        spy_caught = vote_counts[spy_player] > num_players // 2

        return {
            "player_rewards": rewards,
            "spy_caught": spy_caught,
            "vote_counts": dict(vote_counts),
            "spy_player": spy_player,
        }

    def compute_generation_rewards(self, game_outcome: Dict[str, Any]) -> List[float]:
        """Convert game outcome to per-player generation rewards.

        These rewards are used to weight the flow matching loss.
        """
        num_players = len(game_outcome["player_rewards"])
        spy_player = game_outcome["spy_player"]
        spy_caught = game_outcome["spy_caught"]

        gen_rewards = []
        for pid in range(1, num_players + 1):
            if pid == spy_player:
                # Spy: rewarded for stealth (not caught)
                gen_rewards.append(1.0 if not spy_caught else -1.0)
            else:
                # Civilian: rewarded for faithful generation (if spy is caught)
                gen_rewards.append(1.0 if spy_caught else -0.5)

        return gen_rewards


def create_spy_game_data_generator(num_players: int = 4,
                                   num_objects_min: int = 3,
                                   num_objects_max: int = 6,
                                   num_to_modify: int = 2):
    """Factory function to create spy game data generator.

    Returns:
        Tuple of (data_generator_func, reward_funcs_tuple)
    """
    generator = SpyGameDataGenerator(
        num_players=num_players,
        num_objects_min=num_objects_min,
        num_objects_max=num_objects_max,
        num_to_modify=num_to_modify
    )

    def data_generator(epoch: int, sample_idx: int) -> Dict[str, Any]:
        game_data = generator.generate_game(epoch, sample_idx)
        return {
            "game_data": game_data,
            "generator": generator,
        }

    return data_generator, generator
