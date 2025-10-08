"""
Multi-round adaptive sampling for RL training.

This module implements Reinforce-Ada style adaptive downsampling where generation
happens in multiple rounds with early stopping for prompts that have collected
enough positive and negative samples.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSamplingConfig:
    """Configuration for multi-round adaptive sampling."""

    enabled: bool = False
    """Whether to enable multi-round adaptive sampling."""

    strategy: str = "balanced"
    """Sampling strategy: 'balanced' or 'positive_focused'."""

    positive_threshold: float = 0.7
    """Threshold for classifying samples as positive (reward > threshold)."""

    max_rounds: int = 4
    """Maximum number of generation rounds."""

    samples_per_round: int = 8
    """Number of samples to generate per active prompt in each round."""

    final_samples_per_prompt: int = 4
    """Final number of samples to keep per prompt."""

    use_global_stats: bool = False
    """Whether to track global positive/negative counts for GRPO."""


@dataclass
class PromptState:
    """Tracks the state of a single prompt across rounds."""

    finished: bool = False
    """Whether this prompt has collected enough samples."""

    seen: int = 0
    """Total samples generated for this prompt."""

    pos: int = 0
    """Count of positive samples (reward > threshold)."""

    neg: int = 0
    """Count of negative samples (reward <= threshold)."""


@dataclass
class RoundStats:
    """Statistics for a single generation round."""

    round_num: int
    active_prompts: int
    completed_prompts: int
    finished_prompts: int
    reward_mean: float
    duration_sec: float


class AdaptiveSampler:
    """
    Implements multi-round adaptive sampling with early stopping.

    This class manages the iterative generation process where prompts can finish
    early once they've collected enough positive and negative samples.
    """

    def __init__(
        self,
        config: AdaptiveSamplingConfig,
        sampling_client,
        max_tokens: int,
    ):
        self.config = config
        self.sampling_client = sampling_client
        self.max_tokens = max_tokens
        self.policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)

    async def multi_round_adaptive_downsampling(
        self,
        env_group_builders: list[EnvGroupBuilder],
    ) -> tuple[list[TrajectoryGroup], dict[str, Any]]:
        """
        Generate trajectories with multi-round adaptive downsampling.

        Args:
            env_group_builders: List of environment group builders, one per unique prompt.

        Returns:
            Tuple of (trajectory_groups, metadata) where:
                - trajectory_groups: List of TrajectoryGroup objects with selected samples
                - metadata: Dictionary with round statistics and metrics
        """
        if not self.config.enabled:
            # Fallback to standard sampling
            return await self._standard_sampling(env_group_builders)

        logger.info(
            f"Starting adaptive sampling: max_rounds={self.config.max_rounds}, "
            f"samples_per_round={self.config.samples_per_round}, "
            f"strategy={self.config.strategy}"
        )

        num_prompts = len(env_group_builders)

        # Initialize state tracking
        state = {i: PromptState() for i in range(num_prompts)}
        pos_cache: dict[int, list[TrajectoryGroup]] = defaultdict(list)
        neg_cache: dict[int, list[TrajectoryGroup]] = defaultdict(list)
        selected_groups: list[TrajectoryGroup] = []

        # Global statistics for GRPO
        global_stats = {i: {"total_pos": 0, "total_neg": 0} for i in range(num_prompts)}

        # Track which prompts are still active
        active_indices = set(range(num_prompts))

        # Round statistics
        round_stats: list[RoundStats] = []

        # Main generation loop
        for round_num in range(self.config.max_rounds):
            if not active_indices:
                logger.info(f"All prompts finished at round {round_num}")
                break

            round_start = time.time()

            # Generate for active prompts only
            active_builders = [env_group_builders[i] for i in active_indices]

            # Generate trajectories for active prompts
            round_groups = await self._generate_round(
                active_builders, self.config.samples_per_round
            )

            # Map back to original indices
            active_list = sorted(active_indices)
            round_groups_by_prompt = {}
            for i, builder_idx in enumerate(active_list):
                # Each group has samples_per_round trajectories
                start = i * self.config.samples_per_round
                end = start + self.config.samples_per_round
                trajectories = []
                for j in range(start, end):
                    if j < len(round_groups):
                        trajectories.extend(round_groups[j].trajectories_G)

                if trajectories:
                    # Create a TrajectoryGroup for this prompt's samples
                    rewards = [sum(t.reward for t in traj.transitions) for traj in trajectories]
                    metrics = [{} for _ in trajectories]
                    round_groups_by_prompt[builder_idx] = TrajectoryGroup(
                        trajectories_G=trajectories,
                        final_rewards_G=[0.0] * len(trajectories),
                        metrics_G=metrics,
                    )

            # Process results and update caches
            completed_this_round = 0
            all_rewards = []

            for prompt_idx in list(active_indices):
                if prompt_idx not in round_groups_by_prompt:
                    continue

                traj_group = round_groups_by_prompt[prompt_idx]
                rewards = traj_group.get_total_rewards()
                all_rewards.extend(rewards)

                prompt_state = state[prompt_idx]

                # Cache positive and negative samples
                for i, reward in enumerate(rewards):
                    if prompt_state.finished:
                        break

                    prompt_state.seen += 1
                    is_positive = reward > self.config.positive_threshold

                    # Create a single-trajectory group
                    single_traj_group = TrajectoryGroup(
                        trajectories_G=[traj_group.trajectories_G[i]],
                        final_rewards_G=[traj_group.final_rewards_G[i]],
                        metrics_G=[traj_group.metrics_G[i]],
                    )

                    if is_positive:
                        prompt_state.pos += 1
                        pos_cache[prompt_idx].append(single_traj_group)
                        global_stats[prompt_idx]["total_pos"] += 1
                    else:
                        prompt_state.neg += 1
                        neg_cache[prompt_idx].append(single_traj_group)
                        global_stats[prompt_idx]["total_neg"] += 1

                # Check if this prompt can finish
                if self._should_finish_prompt(prompt_idx, pos_cache, neg_cache):
                    selected = self._downsample_prompt_cache(prompt_idx, pos_cache, neg_cache)
                    selected_groups.append(selected)
                    prompt_state.finished = True
                    completed_this_round += 1

            # Record round statistics
            round_duration = time.time() - round_start
            round_stats.append(
                RoundStats(
                    round_num=round_num,
                    active_prompts=len(round_groups_by_prompt),
                    completed_prompts=completed_this_round,
                    finished_prompts=sum(1 for s in state.values() if s.finished),
                    reward_mean=float(np.mean(all_rewards)) if all_rewards else 0.0,
                    duration_sec=round_duration,
                )
            )

            logger.info(
                f"[Round {round_num}] active_prompts={len(active_indices)} "
                f"completed={completed_this_round} "
                f"duration={round_duration:.2f}s "
                f"reward_mean={round_stats[-1].reward_mean:.4f}"
            )

            # Update active set
            active_indices = {i for i in active_indices if not state[i].finished}

        # Handle fallback for unfinished prompts
        unfinished = [i for i in range(num_prompts) if not state[i].finished]
        if unfinished:
            logger.warning(f"{len(unfinished)} prompts did not finish, using fallback strategy")
            for prompt_idx in unfinished:
                if prompt_idx in pos_cache or prompt_idx in neg_cache:
                    selected = self._downsample_prompt_cache(
                        prompt_idx, pos_cache, neg_cache, fallback=True
                    )
                    selected_groups.append(selected)

        # Compile metadata
        metadata = {
            "adaptive_sampling/enabled": True,
            "adaptive_sampling/total_prompts": num_prompts,
            "adaptive_sampling/finished_prompts": sum(1 for s in state.values() if s.finished),
            "adaptive_sampling/total_rounds": len(round_stats),
            "adaptive_sampling/total_samples_generated": sum(
                stat.active_prompts * self.config.samples_per_round for stat in round_stats
            ),
            "adaptive_sampling/final_samples_kept": sum(
                len(group.trajectories_G) for group in selected_groups
            ),
        }

        # Add per-round statistics
        for stat in round_stats:
            metadata[f"adaptive_sampling/round_{stat.round_num}/active"] = stat.active_prompts
            metadata[f"adaptive_sampling/round_{stat.round_num}/completed"] = stat.completed_prompts
            metadata[f"adaptive_sampling/round_{stat.round_num}/reward_mean"] = stat.reward_mean
            metadata[f"adaptive_sampling/round_{stat.round_num}/duration_sec"] = stat.duration_sec

        # Add global stats if enabled
        if self.config.use_global_stats:
            metadata["adaptive_sampling/global_stats"] = global_stats

        return selected_groups, metadata

    async def _generate_round(
        self,
        builders: list[EnvGroupBuilder],
        samples_per_prompt: int,
    ) -> list[TrajectoryGroup]:
        """Generate trajectories for one round."""
        # Expand each builder to generate multiple samples
        expanded_builders = []
        for builder in builders:
            for _ in range(samples_per_prompt):
                expanded_builders.append(builder)

        # Generate all trajectories in parallel
        trajectory_groups = await asyncio.gather(
            *[do_group_rollout(builder, self.policy) for builder in expanded_builders]
        )

        return trajectory_groups

    def _should_finish_prompt(
        self,
        prompt_idx: int,
        pos_cache: dict[int, list],
        neg_cache: dict[int, list],
    ) -> bool:
        """Determine if a prompt has collected enough samples to finish."""
        if self.config.strategy == "balanced":
            # Need balanced positive and negative samples
            target_pos = self.config.final_samples_per_prompt // 2
            target_neg = self.config.final_samples_per_prompt - target_pos
            return (
                len(pos_cache.get(prompt_idx, [])) >= target_pos
                and len(neg_cache.get(prompt_idx, [])) >= target_neg
            )
        elif self.config.strategy == "positive_focused":
            # Just need at least one positive sample
            return len(pos_cache.get(prompt_idx, [])) >= 1
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def _downsample_prompt_cache(
        self,
        prompt_idx: int,
        pos_cache: dict[int, list[TrajectoryGroup]],
        neg_cache: dict[int, list[TrajectoryGroup]],
        fallback: bool = False,
    ) -> TrajectoryGroup:
        """
        Downsample cached samples to reach the target count.

        Args:
            prompt_idx: Index of the prompt
            pos_cache: Cache of positive trajectory groups
            neg_cache: Cache of negative trajectory groups
            fallback: If True, use whatever samples are available

        Returns:
            A merged TrajectoryGroup with selected samples
        """
        pos_groups = pos_cache.get(prompt_idx, [])
        neg_groups = neg_cache.get(prompt_idx, [])

        pos_count = len(pos_groups)
        neg_count = len(neg_groups)
        total_available = pos_count + neg_count

        if total_available == 0:
            raise ValueError(f"No samples available for prompt {prompt_idx}")

        # Determine how many to keep
        target_total = min(self.config.final_samples_per_prompt, total_available)

        if self.config.strategy == "balanced":
            # Try to keep equal positive and negative
            target_pos = min(target_total // 2, pos_count)
            target_neg = min(target_total - target_pos, neg_count)

            # Adjust if one type is insufficient
            if target_pos + target_neg < target_total:
                if pos_count > target_pos:
                    target_pos = min(pos_count, target_total - target_neg)
                elif neg_count > target_neg:
                    target_neg = min(neg_count, target_total - target_pos)
        else:
            # positive_focused: prioritize positives
            if fallback and total_available > 0:
                ratio = pos_count / total_available
                target_pos = max(1, min(int(ratio * target_total), pos_count))
            else:
                target_pos = min(pos_count, target_total)
            target_neg = min(neg_count, target_total - target_pos)

        # Select samples
        selected_pos = pos_groups[:target_pos]
        selected_neg = neg_groups[:target_neg]
        selected_all = selected_pos + selected_neg

        if not selected_all:
            raise ValueError(f"No samples selected for prompt {prompt_idx}")

        # Merge trajectory groups
        all_trajectories = []
        all_final_rewards = []
        all_metrics = []

        for group in selected_all:
            all_trajectories.extend(group.trajectories_G)
            all_final_rewards.extend(group.final_rewards_G)
            all_metrics.extend(group.metrics_G)

        return TrajectoryGroup(
            trajectories_G=all_trajectories,
            final_rewards_G=all_final_rewards,
            metrics_G=all_metrics,
        )

    async def _standard_sampling(
        self,
        env_group_builders: list[EnvGroupBuilder],
    ) -> tuple[list[TrajectoryGroup], dict[str, Any]]:
        """Fallback to standard sampling without adaptive downsampling."""
        trajectory_groups = await asyncio.gather(
            *[do_group_rollout(builder, self.policy) for builder in env_group_builders]
        )

        metadata = {
            "adaptive_sampling/enabled": False,
        }

        return trajectory_groups, metadata
