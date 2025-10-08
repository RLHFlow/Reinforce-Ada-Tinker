"""
Implements RL on general MDPs

"""

import asyncio
import io
import logging
import os
import time
from typing import Any, List, Sequence

import chz
import numpy as np
import tinker
import torch
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.metrics import (
    compute_kl_sample_train,
    compute_post_kl,
    incorporate_kl_penalty,
)
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import safezip, split_list, timed
from tinker_cookbook.rl.reward_history import RewardHistory
from tinker_cookbook.rl.adaptive_sampling import AdaptiveSamplingConfig

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    learning_rate: float
    dataset_builder: RLDatasetBuilder  # also determines batch size
    model_name: str
    max_tokens: int
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None

    remove_constant_reward_groups: bool = False
    eval_every: int = 20
    save_every: int = 20
    load_checkpoint_path: str | None = None

    # Global GRPO statistics
    global_stat_est: bool = False

    # Adaptive Sampling Configuration
    adaptive_sampling: AdaptiveSamplingConfig = chz.field(
        default_factory=lambda: AdaptiveSamplingConfig()
    )


def _select_representative_inds(scores: list[float], num_inds: int) -> list[int]:
    assert num_inds <= len(scores)
    sorted_inds = np.argsort(scores)
    uniform_inds = np.linspace(0, len(sorted_inds) - 1, num_inds).astype(int)
    return [int(sorted_inds[i]) for i in uniform_inds]


def print_group(traj_group: TrajectoryGroup, tokenizer: Tokenizer):
    # Cut down the number of trajectories to print
    max_trajs_to_print = 1
    if len(traj_group.trajectories_G) > max_trajs_to_print:
        inds = _select_representative_inds(traj_group.get_total_rewards(), max_trajs_to_print)
        traj_group = TrajectoryGroup(
            trajectories_G=[traj_group.trajectories_G[i] for i in inds],
            final_rewards_G=[traj_group.final_rewards_G[i] for i in inds],
            metrics_G=[traj_group.metrics_G[i] for i in inds],
        )

    rewards = traj_group.get_total_rewards()
    advantages_G = compute_advantages([traj_group])
    data_D, metadata_D = assemble_training_data([traj_group], advantages_G)

    buf = io.StringIO()

    def bprint(s: str):
        print(s, file=buf)

    bprint("\n====== Trajectory Group ======")
    last_metadata = None
    for datum, metadata in safezip(data_D, metadata_D):
        idx = metadata["traj_idx"]
        if metadata != last_metadata:
            bprint(f"****** trajectory idx={idx}, reward={rewards[idx]:.3g} ******")
            # Print trajectory-level metrics
            if traj_group.metrics_G[idx]:
                bprint("Trajectory metrics:")
                for key, value in traj_group.metrics_G[idx].items():
                    bprint(f"  {key}: {value}")
            # Print per-transition metrics
            transition_metrics = [
                transition.metrics
                for transition in traj_group.trajectories_G[idx].transitions
                if transition.metrics
            ]
            if transition_metrics:
                bprint("Per-step metrics:")
                for i, metrics in enumerate(transition_metrics):
                    bprint(f"  Step {i}:")
                    for key, value in metrics.items():
                        bprint(f"    {key}: {value}")
        bprint("---- datum ----")
        bprint(colorize_example(datum, tokenizer, key="advantages"))
        last_metadata = metadata
    bprint("====== End Trajectory Group ======")
    logger.info(buf.getvalue().rstrip())


async def optim_step(
    training_client: tinker.TrainingClient,
    learning_rate: float,
) -> None:
    """Apply the accumulated gradients to update the model weights"""
    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    optim_step_future = await training_client.optim_step_async(adam_params)
    await optim_step_future.result_async()


def remove_mask(datum: tinker.Datum) -> tinker.Datum:
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


async def forward_backward(
    training_client: tinker.TrainingClient,
    batch_d: List[tinker.Datum],
) -> List[torch.Tensor]:
    """Accumulate gradients on a minibatch of data"""
    fwd_bwd_future = await training_client.forward_backward_async(
        list(map(remove_mask, batch_d)), loss_fn="ppo"
    )
    fwd_bwd_result = await fwd_bwd_future.result_async()

    # Extract training logprobs from loss_fn_outputs
    training_logprobs_D: list[torch.Tensor] = []
    for output in fwd_bwd_result.loss_fn_outputs:
        training_logprobs = output["logprobs"].to_torch()
        training_logprobs_D.append(training_logprobs)

    # We dont display fwd_bwd_result.metrics to avoid spam
    return training_logprobs_D


async def train_step(
    data_D: List[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
) -> List[torch.Tensor]:
    """Train the model on collected trajectories."""
    batches_md = split_list(data_D, min(num_substeps, len(data_D)))
    training_logprobs_D: list[torch.Tensor] = []
    for batch_d in batches_md:
        training_logprobs = await forward_backward(training_client, batch_d)
        training_logprobs_D.extend(training_logprobs)
        await optim_step(training_client, learning_rate)
    return training_logprobs_D


async def do_group_rollout_and_filter_constant_reward(
    cfg: Config,
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
) -> TrajectoryGroup | None:
    policy = TinkerTokenCompleter(sampling_client, max_tokens=cfg.max_tokens)
    trajectory_group = await do_group_rollout(env_group_builder, policy)

    # Remove if all trajectories have the same reward
    trajectory_groups = [trajectory_group]
    if cfg.remove_constant_reward_groups:
        trajectory_groups = remove_constant_reward_groups(trajectory_groups)
    if len(trajectory_groups) == 0:
        return None
    return trajectory_groups[0]


async def save_checkpoint_and_get_sampling_client(
    cfg: Config,
    training_client: tinker.TrainingClient,
    i_batch: int,
    reward_history: RewardHistory | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    metrics = {}
    with timed("save_checkpoint", metrics):
        path_dict = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"{i_batch:06d}",
            log_path=cfg.log_path,
            loop_state={"batch": i_batch},
            kind="both" if (i_batch > 0 and i_batch % cfg.save_every == 0) else "sampler",
        )

        # Save reward history if it exists
        if reward_history is not None:
            reward_history_path = os.path.join(cfg.log_path, f"reward_history_{i_batch:06d}.json")
            reward_history.save(reward_history_path)

            # Also save as "latest" for easy resumption
            latest_path = os.path.join(cfg.log_path, "reward_history_latest.json")
            reward_history.save(latest_path)

            # Add reward history stats to metrics
            stats = reward_history.get_stats()
            metrics.update({f"reward_history/{k}": v for k, v in stats.items()})

        return training_client.create_sampling_client(path_dict["sampler_path"]), metrics


async def prepare_minibatch(
    cfg: Config,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    service_client: tinker.ServiceClient,
    reward_history: RewardHistory | None = None,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Print one trajectory
    for traj_group in trajectory_groups_P[:2]:
        print_group(traj_group, tokenizer)

    # Assemble training data
    with timed("assemble_training_data", metrics):
        if cfg.global_stat_est:
            if reward_history is None:
                raise ValueError("reward_history must be provided when global_stat_est=True")
            advantages_P = compute_advantages(
                trajectory_groups_P,
                env_group_builders_P=list(env_group_builders_P),
                reward_history=reward_history,
                use_global=True,
            )
        else:
            advantages_P = compute_advantages(trajectory_groups_P, use_global=False)

        data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Incorporate KL penalty if configured
    if cfg.kl_penalty_coef > 0:
        with timed("kl_vs_base", metrics):
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                service_client.create_sampling_client(base_model=cfg.model_name),
                # ^^^ TODO: replace with the model we load, if relevant
                cfg.kl_penalty_coef,
                cfg.kl_discount_factor,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


async def compute_full_batch_metrics_and_get_sampling_client(
    cfg: Config,
    training_client: tinker.TrainingClient,
    i_batch: int,
    data_D: list[tinker.Datum],
    training_logprobs_D: list[torch.Tensor],
    reward_history: RewardHistory | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    At the end of the iteration, this will compute metrics for the full batch
    and return the latest sampling client.
    """
    metrics = {}

    # Compute KL metrics
    with timed("compute_kl_sample_train", metrics):
        kl_sample_train_metrics = compute_kl_sample_train(data_D, training_logprobs_D)
        metrics.update(kl_sample_train_metrics)

    # Get a sampling client using the new weights
    sampling_client, checkpoint_metrics = await save_checkpoint_and_get_sampling_client(
        cfg,
        training_client,
        i_batch,
        reward_history=reward_history,
    )
    metrics.update(checkpoint_metrics)

    # Compute post-KL metrics if configured
    if cfg.compute_post_kl:
        with timed("compute_post_kl", metrics):
            post_kl_metrics = await compute_post_kl(data_D, sampling_client)
            metrics.update(post_kl_metrics)

    return sampling_client, metrics


async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    reward_history: RewardHistory | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        cfg,
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        service_client,
        reward_history=reward_history,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
        )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        cfg,
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        data_D,
        training_logprobs_D,
        reward_history=reward_history,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    reward_history: RewardHistory | None = None,
):
    """Implements fully synchronous on-policy training with adaptive sampling."""

    from tinker_cookbook.rl.adaptive_sampling import AdaptiveSampler

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        cfg, training_client, start_batch, reward_history=reward_history
    )

    # Create adaptive sampler
    adaptive_sampler = None
    if cfg.adaptive_sampling.enabled:
        adaptive_sampler = AdaptiveSampler(
            config=cfg.adaptive_sampling,
            sampling_client=sampling_client,
            max_tokens=cfg.max_tokens,
        )

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                for evaluator in evaluators:
                    eval_metrics = await evaluator(sampling_client)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # Get batch and sample trajectories
        env_group_builders_P = dataset.get_batch(i_batch)
        with timed("sample", metrics):
            if adaptive_sampler:
                # Multi-round adaptive sampling
                (
                    trajectory_groups_P,
                    adaptive_metrics,
                ) = await adaptive_sampler.multi_round_adaptive_downsampling(env_group_builders_P)
                # metrics.update(adaptive_metrics)

                logger.info(
                    f"Adaptive sampling completed: "
                    f"{len(trajectory_groups_P)} groups, "
                    f"{sum(len(g.trajectories_G) for g in trajectory_groups_P)} trajectories"
                )
            else:
                # Standard sampling: generate all samples at once
                policy = TinkerTokenCompleter(sampling_client, max_tokens=cfg.max_tokens)
                trajectory_groups_P = await asyncio.gather(
                    *[do_group_rollout(builder, policy) for builder in env_group_builders_P]
                )

        trajectory_groups_P = [
            trajectory_group
            for trajectory_group in trajectory_groups_P
            if trajectory_group is not None
        ]

        # Train step
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
            reward_history=reward_history,
        )

        # Update adaptive sampler with new sampling client
        if cfg.adaptive_sampling.enabled:
            adaptive_sampler.sampling_client = sampling_client
            adaptive_sampler.policy = TinkerTokenCompleter(
                sampling_client, max_tokens=cfg.max_tokens
            )

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


async def main(
    cfg: Config,
):
    """Main training loop for MDP RL."""
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    load_state_path: str | None = (
        resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    )
    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create dataset from thunk
    dataset, maybe_test_dataset = await cfg.dataset_builder()
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens))

    num_batches = len(dataset)
    logger.info(f"Will train on {num_batches} batches")

    # Initialize reward history tracker if using global statistics
    reward_history = None
    if cfg.global_stat_est:
        reward_history = RewardHistory()
        logger.info("Using global statistics estimation for advantages")

        # Try to load reward history if resuming
        if resume_info:
            # Try to load from the specific batch checkpoint
            reward_history_path = os.path.join(
                cfg.log_path, f"reward_history_{start_batch:06d}.json"
            )
            if os.path.exists(reward_history_path):
                reward_history.load(reward_history_path)
            else:
                # Fall back to latest
                latest_path = os.path.join(cfg.log_path, "reward_history_latest.json")
                if os.path.exists(latest_path):
                    reward_history.load(latest_path)
                else:
                    logger.warning("No reward history found, starting fresh")

    # Training loop
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=dataset,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
        reward_history=reward_history,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )

        # Save final reward history
        if reward_history is not None:
            final_reward_history_path = os.path.join(cfg.log_path, "reward_history_final.json")
            reward_history.save(final_reward_history_path)
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
