import asyncio
import logging
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import AsyncConfig, Config, main
from tinker_cookbook.recipes.reinforce_ada import math_env

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Simple command-line configuration for RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Training set
    dataset_name: str = "RLHFlow/reinforce_ada_hard_prompt"

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 512
    learning_rate: float = 1e-6
    max_tokens: int = 2048
    kl_penalty_coef: float = 0.0

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals
    eval_every: int = 50

    # Checkpointing
    save_every: int = 50

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None

    ## Reinforce-Ada specific hyperparameters
    multiround_adaptive_downsampling: bool = False
    reinforce_ada_choice: str | None = None  # "balanced" or "positive-focused"
    global_stat_est: bool = False

    ## TODO: clip
    # clip_ratio_low=0.2
    # clip_ratio_high=0.28


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get tokenizer for stop sequences
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_name = cli_config.model_name.replace("/", "-")
    run_name = f"{model_name}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.group_size}group-{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    # create log path if it doesn't exist
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/math_rl/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    # dataset builder
    dataset_builder = math_env.ReinforceAdaDatasetBuilder(
        dataset_name=cli_config.dataset_name,
        batch_size=cli_config.groups_per_batch,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        convo_prefix="standard",
    )

    # Create full config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        multiround_adaptive_downsampling=cli_config.multiround_adaptive_downsampling,
        reinforce_ada_choice=cli_config.reinforce_ada_choice,
        global_stat_est=cli_config.global_stat_est,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
