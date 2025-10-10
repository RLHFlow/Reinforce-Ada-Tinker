"""
Implements RL on general MDPs

"""

import asyncio
import io
import logging
import os
import time
from typing import Any, List, Sequence, Callable

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
    compute_sampling_client_metrics,
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
class AsyncConfig:
    """Configuration for async RL training"""

    # If samples are generated from a sample more than this many steps ago,
    # we will skip training on them.
    max_steps_off_policy: int
    # We will ensure all batches have at least this many groups, even
    # as we discard stale samples
    groups_per_batch: int
    

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

    # Multi-epoch training parameters
    num_epochs: int = 1
    total_steps: int | None = None  # If set, overrides num_epochs

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None

    remove_constant_reward_groups: bool = False
    eval_every: int = 50
    save_every: int = 50
    load_checkpoint_path: str | None = None

    async_config: AsyncConfig | None = None

    # Global GRPO statistics
    global_stat_est: bool = False

    # Adaptive Sampling Configuration
    adaptive_sampling: AdaptiveSamplingConfig | None = None


@chz.chz
class WrappedTrajectoryGroup:
    """
    A wrapper around a trajectory group that includes metadata about how it was generated.
    Used when we need to overlap sampling and training.
    """

    trajectory_group: TrajectoryGroup
    # The env group builder that produced the trajectory group.
    # Pass this along in case the sampler is too stale, and we need to
    # requeue this group.
    env_group_builder: EnvGroupBuilder
    # The step that produced this trajectory group.
    sampling_client_step: int
    metrics: dict[str, Any] = chz.field(default_factory=dict)


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


async def save_checkpoint_and_get_sampling_client(
    cfg: Config,
    training_client: tinker.TrainingClient,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    reward_history: RewardHistory | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    metrics = {}
    with timed("save_checkpoint", metrics):
        path_dict = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"{step:06d}",
            log_path=cfg.log_path,
            loop_state={"step": step, "epoch": epoch, "batch": batch_in_epoch},
            kind="both" if (step > 0 and step % cfg.save_every == 0) else "sampler",
        )

        # Save reward history if it exists
        if reward_history is not None and step > 0 and step % cfg.save_every == 0:
            reward_history_path = os.path.join(cfg.log_path, f"reward_history_{step:06d}.json")
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
    for traj_group in trajectory_groups_P[:1]:
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
    step: int,
    epoch: int,
    batch_in_epoch: int,
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
        step,
        epoch,
        batch_in_epoch,
        reward_history=reward_history,
    )
    metrics.update(checkpoint_metrics)

    # Compute post-KL metrics if configured
    if cfg.compute_post_kl:
        with timed("compute_post_kl", metrics):
            post_kl_metrics = await compute_post_kl(data_D, sampling_client)
            metrics.update(post_kl_metrics)

    return sampling_client, metrics


async def do_train_step_streaming_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    trajectory_groups_queue: asyncio.Queue[WrappedTrajectoryGroup | None],
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    trajectory_group_filter: Callable[[WrappedTrajectoryGroup | None], bool] = lambda _: True,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    As soon as we have enough trajectories for a minibatch, we will train on them.
    This allows us to overlap sampling and training.
    """
    assert cfg.stream_minibatch_config is not None
    assert cfg.stream_minibatch_config.groups_per_batch % cfg.num_substeps == 0, (
        f"{cfg.stream_minibatch_config.groups_per_batch=} must be divisible by {cfg.num_substeps=}"
    )
    # Number of groups across all minibatches in each optimizer substep
    groups_per_substep = cfg.stream_minibatch_config.groups_per_batch // cfg.num_substeps
    assert groups_per_substep % cfg.stream_minibatch_config.num_minibatches == 0, (
        f"{groups_per_substep} must be divisible by {cfg.stream_minibatch_config.num_minibatches=}"
    )
    # Number of groups per minibatch in each optimizer substep
    groups_per_minibatch = groups_per_substep // cfg.stream_minibatch_config.num_minibatches

    metrics = {}

    # Run multiple optimizer substeps per training iteration
    all_data_D = []
    all_training_logprobs_D = []
    all_wrapped_trajectory_groups = []
    for i_substep in range(cfg.num_substeps):
        # Run multiple minibatches per substep
        # Once we have enough trajectories for a minibatch, train on them
        wrapped_trajectory_groups = []
        i_minibatch = 0
        while i_minibatch < cfg.stream_minibatch_config.num_minibatches:
            wrapped_trajectory_group = await trajectory_groups_queue.get()
            if not trajectory_group_filter(wrapped_trajectory_group):
                continue
            wrapped_trajectory_groups.append(wrapped_trajectory_group)

            if len(wrapped_trajectory_groups) < groups_per_minibatch:
                continue
            logger.info(
                f"[stream_minibatch] Step {i_batch}, Substep {i_substep}/{cfg.num_substeps}, Minibatch {i_minibatch}/{cfg.stream_minibatch_config.num_minibatches}: Will train on minibatch, num groups: {len(wrapped_trajectory_groups)}"
            )

            # Note: we may have removed trajectory groups that have the same reward.
            # To have the same results as the sync implementation, we will
            # remove these and train on a smaller batch.
            wrapped_trajectory_groups = [g for g in wrapped_trajectory_groups if g is not None]
            data_D, prepare_minibatch_metrics = await prepare_minibatch(
                cfg,
                [g.env_group_builder for g in wrapped_trajectory_groups],
                [g.trajectory_group for g in wrapped_trajectory_groups],
                tokenizer,
                service_client,
            )
            metrics.update(prepare_minibatch_metrics)

            # Accumulate gradients across multiple minibatches
            with timed(
                f"train/forward_backward_substep_{i_substep}_minibatch_{i_minibatch}", metrics
            ):
                training_logprobs_D = await forward_backward(
                    training_client,
                    data_D,
                )
            all_data_D.extend(data_D)
            all_training_logprobs_D.extend(training_logprobs_D)
            all_wrapped_trajectory_groups.extend(wrapped_trajectory_groups)
            i_minibatch += 1
            wrapped_trajectory_groups = []

        # Run optimizer step only once after all minibatches
        with timed(f"train/optim_substep_{i_substep}", metrics):
            await optim_step(training_client, cfg.learning_rate)

    # Aggregate metrics across the entire batch
    metrics.update(compute_sampling_client_metrics(all_wrapped_trajectory_groups))
    metrics.update(
        compute_trajectory_metrics(
            [g.trajectory_group for g in all_wrapped_trajectory_groups],
            [g.env_group_builder.logging_tags() for g in all_wrapped_trajectory_groups],
        )
    )
    (
        sampling_client,
        full_batch_metrics,
    ) = await compute_full_batch_metrics_and_get_sampling_client(
        cfg,
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        all_data_D,
        all_training_logprobs_D,
    )
    metrics.update(full_batch_metrics)
    return sampling_client, metrics


async def do_train_step_and_get_sampling_client(
    cfg: Config,
    step: int,
    epoch: int,
    batch_in_epoch: int,
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
        step + 1,
        epoch,
        batch_in_epoch,
        data_D,
        training_logprobs_D,
        reward_history=reward_history,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


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


async def do_async_training(
    start_step: int,
    start_epoch: int,
    start_batch: int,
    total_steps: int,
    batches_per_epoch: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    reward_history: RewardHistory | None = None,
):
    """Implements async off-policy training with adaptive sampling support."""
    assert cfg.async_config is not None
    
    from tinker_cookbook.rl.adaptive_sampling import AdaptiveSampler

    shutdown_event = asyncio.Event()
    env_group_builders_queue = asyncio.Queue[EnvGroupBuilder | None](
        maxsize=cfg.async_config.groups_per_batch
    )
    trajectory_groups_queue = asyncio.Queue[WrappedTrajectoryGroup | None]()

    # Initial sampling client to use
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        cfg, training_client, start_step, start_epoch, start_batch, reward_history=reward_history
    )
    
    # Create adaptive sampler if enabled
    adaptive_sampler = None
    if cfg.adaptive_sampling and cfg.adaptive_sampling.enabled:
        adaptive_sampler = AdaptiveSampler(
            config=cfg.adaptive_sampling,
            sampling_client=sampling_client,
            max_tokens=cfg.max_tokens,
        )
    
    sampling_client_step = start_step
    sampling_client_updated_event = asyncio.Event()
    sampling_client_updated_event.set()

    def shutdown_loops():
        """Trigger all loops to shutdown"""
        shutdown_event.set()
        assert cfg.async_config is not None
        for _ in range(cfg.async_config.groups_per_batch):
            env_group_builders_queue.put_nowait(None)
        sampling_client_updated_event.set()

    async def dataloader_loop():
        """Gets the next set of env builders to run"""
        current_batch = start_batch
        current_epoch = start_epoch
        i_step = start_step
        
        while not shutdown_event.is_set() and i_step < total_steps:
            # Shuffle dataset at the beginning of each epoch
            if current_batch == 0:
                logger.info(f"Starting epoch {current_epoch} (step {i_step}/{total_steps})")
                dataset.set_epoch(seed=current_epoch)
                logger.info(f"Dataset shuffled with seed={current_epoch}")
            
            env_group_builders_P = dataset.get_batch(current_batch)
            for env_group_builder in env_group_builders_P:
                await env_group_builders_queue.put(env_group_builder)
            
            i_step += 1
            current_batch += 1
            
            # Move to next epoch if we've completed the current one
            if current_batch >= batches_per_epoch:
                current_batch = 0
                current_epoch += 1

    async def trajectory_group_worker_loop():
        """Generates trajectories for a single env builder"""
        while not shutdown_event.is_set():
            env_group_builder = await env_group_builders_queue.get()
            if env_group_builder is None:
                break

            metrics = {}
            t_start = time.time()
            # Save a reference to the sampling client step in case it changes
            # while we're running the rollout
            sampling_client_step_copy = sampling_client_step
            
            if adaptive_sampler:
                # Multi-round adaptive sampling
                trajectory_groups, adaptive_metrics = await adaptive_sampler.multi_round_adaptive_downsampling(
                    [env_group_builder],
                    reward_history=reward_history,
                )
                metrics.update(adaptive_metrics)
                
                # Adaptive sampling returns a list of trajectory groups
                if trajectory_groups:
                    trajectory_group = trajectory_groups[0]
                else:
                    trajectory_group = None
            else:
                # Standard sampling
                trajectory_group = await do_group_rollout_and_filter_constant_reward(
                    cfg,
                    sampling_client,
                    env_group_builder,
                )
            
            if trajectory_group is None:
                trajectory_groups_queue.put_nowait(None)
            else:
                metrics["time/trajectory_group_worker_loop/total"] = time.time() - t_start
                trajectory_groups_queue.put_nowait(
                    WrappedTrajectoryGroup(
                        trajectory_group=trajectory_group,
                        env_group_builder=env_group_builder,
                        sampling_client_step=sampling_client_step_copy,
                        metrics=metrics,
                    )
                )

    async def training_loop():
        """
        Waits for a sufficient number of valid trajectories to be accumulated and trains on them.
        Will discard trajectories that are too stale.
        """
        assert cfg.async_config is not None

        current_step = start_step
        current_epoch = start_epoch
        current_batch = start_batch
        wrapped_trajectory_groups = []
        
        while current_step < total_steps:
            wrapped_trajectory_group = await trajectory_groups_queue.get()
            if wrapped_trajectory_group is None:
                continue

            def filter_stale_trajectory_group(
                wrapped_trajectory_group: WrappedTrajectoryGroup | None,
            ) -> bool:
                """Returns False if the trajectory group is too stale or not valid"""
                if wrapped_trajectory_group is None:
                    return False

                # If the samples are too stale, requeue the data so that it will be used eventually.
                # Requeue on a separate coroutine to avoid blocking the training loop
                assert cfg.async_config is not None
                if (
                    current_step - wrapped_trajectory_group.sampling_client_step
                    > cfg.async_config.max_steps_off_policy
                ):
                    logger.info(f"[training_loop] Step {current_step}: Samples are too stale, skipping")
                    asyncio.create_task(
                        env_group_builders_queue.put(wrapped_trajectory_group.env_group_builder)
                    )
                    return False
                return True

            if not filter_stale_trajectory_group(wrapped_trajectory_group):
                continue

            metrics = {
                "progress/step": current_step,
                "progress/epoch": current_epoch,
                "progress/batch_in_epoch": current_batch,
                "progress/done_frac": current_step / total_steps,
            }
            t_start = time.time()

            # Dynamic sampling: Wait for enough trajectories to accumulate to
            # ensure all batch sizes are the same size. This avoids needing to adjust
            # the learning rate for different batch sizes.
            wrapped_trajectory_groups.append(wrapped_trajectory_group)
            if len(wrapped_trajectory_groups) < cfg.async_config.groups_per_batch:
                continue
            logger.info(
                f"[training_loop] Step {current_step}: Will train on batch, num groups: {len(wrapped_trajectory_groups)}"
            )

            # Compute sampling client metrics, as samples may have been generated with
            # different sampler versions
            metrics.update(compute_sampling_client_metrics(wrapped_trajectory_groups))

            #  Aggregate adaptive sampling metrics from trajectory workers
            adaptive_enabled_values = []
            first_round_reward_values = []
            
            for wrapped_traj in wrapped_trajectory_groups:
                if "adaptive_sampling/enabled" in wrapped_traj.metrics:
                    adaptive_enabled_values.append(wrapped_traj.metrics["adaptive_sampling/enabled"])
                if "adaptive_sampling/first_round_reward" in wrapped_traj.metrics:
                    first_round_reward_values.append(wrapped_traj.metrics["adaptive_sampling/first_round_reward"])
            
            # Log aggregated adaptive sampling metrics
            if adaptive_enabled_values:
                metrics["adaptive_sampling/enabled"] = all(adaptive_enabled_values)
            
            if first_round_reward_values:
                metrics["adaptive_sampling/real_reward"] = float(np.mean(first_round_reward_values))

            nonlocal sampling_client
            nonlocal sampling_client_step
            nonlocal adaptive_sampler
            
            sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
                cfg,
                current_step,
                current_epoch,
                current_batch,
                training_client,
                service_client,
                tokenizer,
                [g.env_group_builder for g in wrapped_trajectory_groups],
                [g.trajectory_group for g in wrapped_trajectory_groups],
                reward_history=reward_history,
            )
            
            sampling_client_step = current_step + 1
            sampling_client_updated_event.set()
            
            # Update adaptive sampler with new sampling client
            if adaptive_sampler:
                adaptive_sampler.sampling_client = sampling_client
                adaptive_sampler.policy = TinkerTokenCompleter(
                    sampling_client, max_tokens=cfg.max_tokens
                )

            # Log metrics
            metrics.update(train_step_metrics)
            metrics["time/total"] = time.time() - t_start
            ml_logger.log_metrics(metrics, step=current_step)
            
            # Update counters
            current_step += 1
            current_batch += 1
            wrapped_trajectory_groups = []
            
            # Move to next epoch if we've completed the current one
            if current_batch >= batches_per_epoch:
                current_batch = 0
                current_epoch += 1

        shutdown_loops()

    async def evaluation_loop():
        """Runs evals periodically"""
        if len(evaluators) == 0 or cfg.eval_every == 0:
            return

        while not shutdown_event.is_set():
            await sampling_client_updated_event.wait()
            sampling_client_updated_event.clear()

            metrics = {}
            t_start = time.time()
            # Save a reference to the original values in case it changes
            # while we're running the evals
            sampling_client_eval_step = sampling_client_step
            sampling_client_eval = sampling_client
            if cfg.eval_every > 0 and sampling_client_eval_step % cfg.eval_every == 0:
                with timed("run_evals", metrics):
                    for evaluator in evaluators:
                        eval_metrics = await evaluator(sampling_client_eval)
                        metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})
                metrics["time/evaluation_loop/total"] = time.time() - t_start
                ml_logger.log_metrics(metrics, step=sampling_client_eval_step)

    await asyncio.gather(
        dataloader_loop(),
        *[trajectory_group_worker_loop() for _ in range(cfg.async_config.groups_per_batch)],
        training_loop(),
        evaluation_loop(),
    )


async def do_sync_training(
    start_step: int,
    start_epoch: int,
    start_batch: int,
    total_steps: int,
    batches_per_epoch: int,
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
        cfg, training_client, start_step, start_epoch, start_batch, reward_history=reward_history
    )

    # Create adaptive sampler
    adaptive_sampler = None
    if cfg.adaptive_sampling and cfg.adaptive_sampling.enabled:
        adaptive_sampler = AdaptiveSampler(
            config=cfg.adaptive_sampling,
            sampling_client=sampling_client,
            max_tokens=cfg.max_tokens,
        )

    current_step = start_step
    current_epoch = start_epoch
    current_batch = start_batch

    while current_step < total_steps:
        # Shuffle dataset at the beginning of each epoch
        if current_batch == 0:
            logger.info(f"Starting epoch {current_epoch} (step {current_step}/{total_steps})")
            dataset.set_epoch(seed=current_epoch)
            logger.info(f"Dataset shuffled with seed={current_epoch}")

        metrics = {
            "progress/step": current_step,
            "progress/epoch": current_epoch,
            "progress/batch_in_epoch": current_batch,
            "progress/done_frac": current_step / total_steps,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and current_step % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                for evaluator in evaluators:
                    eval_metrics = await evaluator(sampling_client)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # Get batch and sample trajectories
        env_group_builders_P = dataset.get_batch(current_batch)
        with timed("sample", metrics):
            if adaptive_sampler:
                # Multi-round adaptive sampling
                (
                    trajectory_groups_P,
                    adaptive_metrics,
                ) = await adaptive_sampler.multi_round_adaptive_downsampling(
                    env_group_builders_P,
                    reward_history=reward_history
                )
                metrics.update(adaptive_metrics)

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
            current_step,
            current_epoch,
            current_batch,
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
        ml_logger.log_metrics(metrics, step=current_step)

        # Update counters
        current_step += 1
        current_batch += 1

        # Move to next epoch if we've completed the current one
        if current_batch >= batches_per_epoch:
            current_batch = 0
            current_epoch += 1


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
        start_step = resume_info.get("step", resume_info.get("batch", 0))
        start_epoch = resume_info.get("epoch", 0)
        start_batch = resume_info.get("batch", 0)
    else:
        start_step = 0
        start_epoch = 0
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

    batches_per_epoch = len(dataset)

    # Calculate total steps
    if cfg.total_steps is not None:
        total_steps = cfg.total_steps
        logger.info(f"Training for {total_steps} steps (overriding num_epochs)")
    else:
        total_steps = batches_per_epoch * cfg.num_epochs
        logger.info(
            f"Training for {batches_per_epoch} batches x {cfg.num_epochs} epochs = {total_steps} steps"
        )

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
    if cfg.async_config is not None:
        training_func = do_async_training
    else:
        training_func = do_sync_training
    await training_func(
        start_step=start_step,
        start_epoch=start_epoch,
        start_batch=start_batch,
        total_steps=total_steps,
        batches_per_epoch=batches_per_epoch,
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
    if start_batch < total_steps:
        final_epoch = (total_steps - 1) // batches_per_epoch
        final_batch = (total_steps - 1) % batches_per_epoch
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"step": total_steps, "epoch": final_epoch, "batch": final_batch},
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
