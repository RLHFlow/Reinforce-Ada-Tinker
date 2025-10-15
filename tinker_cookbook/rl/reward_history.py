"""
Track history rewards for global GRPO
"""

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)


class RewardHistory:
    """Tracks rewards for each prompt to compute global advantages."""

    def __init__(self):
        # Maps prompt hash to list of rewards
        self.prompt_rewards: Dict[str, List[float]] = defaultdict(list)

    def add_rewards(self, prompt_id: str, rewards: List[float]):
        """Add new rewards for a prompt."""
        self.prompt_rewards[prompt_id].extend(rewards)

    def get_mean(self, prompt_id: str) -> float:
        """Get the global mean reward for a prompt."""
        rewards = self.prompt_rewards[prompt_id]
        if not rewards:
            return 0.0
        return sum(rewards) / len(rewards)

    def get_count(self, prompt_id: str) -> int:
        """Get the number of rewards collected for a prompt."""
        return len(self.prompt_rewards[prompt_id])

    def clear(self):
        """Clear all history."""
        self.prompt_rewards.clear()

    def save(self, path: str) -> None:
        """
        Save reward history to disk.

        Args:
            path: Path to save the reward history JSON file
        """
        # Convert defaultdict to regular dict for JSON serialization
        data = {"prompt_rewards": dict(self.prompt_rewards)}

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved reward history to {path}")
        logger.info(f"  Total prompts tracked: {len(self.prompt_rewards)}")
        total_rewards = sum(len(rewards) for rewards in self.prompt_rewards.values())
        logger.info(f"  Total rewards stored: {total_rewards}")

    def load(self, path: str) -> None:
        """
        Load reward history from disk.

        Args:
            path: Path to the reward history JSON file
        """
        if not os.path.exists(path):
            logger.warning(f"Reward history file not found: {path}")
            return

        with open(path, "r") as f:
            data = json.load(f)

        # Restore the prompt_rewards dictionary
        self.prompt_rewards = defaultdict(list, data["prompt_rewards"])

        logger.info(f"Loaded reward history from {path}")
        logger.info(f"  Total prompts tracked: {len(self.prompt_rewards)}")
        total_rewards = sum(len(rewards) for rewards in self.prompt_rewards.values())
        logger.info(f"  Total rewards stored: {total_rewards}")

    def get_stats(self) -> Dict[str, float]:
        """
        Get statistics about the reward history.

        Returns:
            Dictionary with statistics about the reward history
        """
        if not self.prompt_rewards:
            return {
                "num_prompts": 0,
                "overall_mean": 0.0,
            }

        all_rewards = [r for rewards in self.prompt_rewards.values() for r in rewards]

        return {
            "num_prompts": len(self.prompt_rewards),
            "overall_mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        }
