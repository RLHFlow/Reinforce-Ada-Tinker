from typing import List
import torch
import tinker


def compute_ppo_loss(
    target_tokens_D: List[torch.Tensor],
    sampled_logprobs_D: List[torch.Tensor],
    training_logprobs_D: List[torch.Tensor],
    advantages_D: List[torch.Tensor],
    masks_D: List[torch.Tensor],
    clip_ratio: float = 0.2,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    clip_ratio_c: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute PPO loss with dual-clip mechanism.

    Args:
        target_tokens_D: List of target token tensors
        sampled_logprobs_D: Log probabilities from sampling (old policy)
        training_logprobs_D: Log probabilities from current policy
        advantages_D: Advantage estimates for each token
        masks_D: Masks indicating which tokens to include in loss
        clip_ratio: Standard PPO clipping parameter (epsilon)
        clip_ratio_low: Lower clip range (defaults to clip_ratio)
        clip_ratio_high: Upper clip range (defaults to clip_ratio)
        clip_ratio_c: Lower bound for dual-clip PPO

    Returns:
        Tuple of (loss tensor, metrics dictionary)
    """
    if clip_ratio_low is None:
        clip_ratio_low = clip_ratio
    if clip_ratio_high is None:
        clip_ratio_high = clip_ratio

    assert clip_ratio_c > 1.0, f"clip_ratio_c must be greater than 1.0, got {clip_ratio_c}"

    # Concatenate all sequences
    all_sampled_logprobs = torch.cat(sampled_logprobs_D)
    all_training_logprobs = torch.cat(training_logprobs_D)
    all_advantages = torch.cat(advantages_D)
    all_masks = torch.cat(masks_D)

    # Compute log probability ratio: log(π_new/π_old) = log(π_new) - log(π_old)
    log_ratio = all_training_logprobs - all_sampled_logprobs

    # Clamp for numerical stability
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)

    # Compute importance sampling ratio
    ratio = torch.exp(log_ratio)

    # Standard PPO loss components
    pg_losses1 = -all_advantages * ratio
    pg_losses2 = -all_advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)

    # Take the max (more conservative) of unclipped and clipped
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    # Dual-clip: additional lower bound when advantages are negative
    pg_losses3 = -all_advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    # Apply dual-clip only when advantages < 0
    pg_losses = torch.where(all_advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply mask and compute mean loss
    masked_losses = pg_losses * all_masks
    loss = masked_losses.sum() / all_masks.sum().clamp(min=1.0)

    # Compute metrics
    with torch.no_grad():
        # Approximate KL divergence: KL(old||new) ≈ -log_ratio
        approx_kl = (-log_ratio * all_masks).sum() / all_masks.sum().clamp(min=1.0)

        # Clip fraction: how often clipping was active
        clip_fraction = (
            torch.gt(pg_losses2, pg_losses1).float() * all_masks
        ).sum() / all_masks.sum().clamp(min=1.0)

        # Lower clip fraction: how often lower bound was active (for negative advantages)
        lower_clip_fraction = (
            torch.gt(clip_pg_losses1, pg_losses3).float() * (all_advantages < 0).float() * all_masks
        ).sum() / all_masks.sum().clamp(min=1.0)

    metrics = {
        "ppo_loss": loss.item(),
        "approx_kl": approx_kl.item(),
        "clip_fraction": clip_fraction.item(),
        "lower_clip_fraction": lower_clip_fraction.item(),
        "mean_ratio": (ratio * all_masks).sum().item() / all_masks.sum().item(),
    }

    return loss, metrics


async def forward_backward_ppo(
    training_client: tinker.TrainingClient,
    data_D: List[tinker.Datum],
    clip_ratio: float = 0.2,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    clip_ratio_c: float = 3.0,
) -> tuple[List[torch.Tensor], dict[str, float]]:
    """
    Perform forward-backward pass with explicit PPO loss.

    This is the replacement for:
        training_client.forward_backward(data_D, loss_fn="ppo")

    Args:
        training_client: The Tinker training client
        data_D: List of training data
        clip_ratio: PPO clipping parameter
        clip_ratio_low: Lower clip bound
        clip_ratio_high: Upper clip bound
        clip_ratio_c: Dual-clip lower bound

    Returns:
        Tuple of (training logprobs, metrics)
    """

    # Extract the custom PPO fields from loss_fn_inputs BEFORE passing to Tinker
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    advantages_D = [datum.loss_fn_inputs["advantages"].to_torch() for datum in data_D]
    masks_D = [datum.loss_fn_inputs["mask"].to_torch() for datum in data_D]
    target_tokens_D = [torch.tensor(datum.loss_fn_inputs["target_tokens"].data) for datum in data_D]

    # Create cleaned data WITHOUT the custom fields
    # Tinker only needs model_input and target_tokens for forward pass
    cleaned_data_D = []
    for datum in data_D:
        cleaned_datum = tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                "target_tokens": datum.loss_fn_inputs["target_tokens"],
                # Only include target_tokens - Tinker will compute logprobs
            },
        )
        cleaned_data_D.append(cleaned_datum)

    def ppo_loss_fn(
        data: List[tinker.Datum], training_logprobs_list: List[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Custom loss function for PPO.

        Note: The 'data' parameter here is the cleaned_data_D we passed in,
        but we use the pre-extracted values (sampled_logprobs_D, advantages_D, etc.)
        from the outer scope.
        """

        # Compute PPO loss using pre-extracted values
        return compute_ppo_loss(
            target_tokens_D=target_tokens_D,
            sampled_logprobs_D=sampled_logprobs_D,
            training_logprobs_D=training_logprobs_list,
            advantages_D=advantages_D,
            masks_D=masks_D,
            clip_ratio=clip_ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            clip_ratio_c=clip_ratio_c,
        )

    # Use forward_backward_custom like in DPO
    fwd_bwd_future = await training_client.forward_backward_custom_async(
        cleaned_data_D,  # Pass cleaned data without custom fields
        ppo_loss_fn,
    )
    backward_result = await fwd_bwd_future.result_async()

    # Extract training logprobs from the result
    training_logprobs_D = [
        output["logprobs"].to_torch() for output in backward_result.loss_fn_outputs
    ]

    return training_logprobs_D, backward_result.metrics


async def train_step_explicit_ppo(
    data_D: List[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
    clip_ratio: float = 0.2,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    clip_ratio_c: float = 3.0,
) -> List[torch.Tensor]:
    """
    Train the model on collected trajectories with explicit PPO loss.

    This replaces the train_step function in rl/train.py
    """
    from tinker_cookbook.utils.misc_utils import split_list
    from tinker_cookbook.rl.train import optim_step

    batches_md = split_list(data_D, min(num_substeps, len(data_D)))
    training_logprobs_D: list[torch.Tensor] = []

    for batch_d in batches_md:
        # Forward-backward with explicit PPO loss
        training_logprobs, ppo_metrics = await forward_backward_ppo(
            training_client,
            batch_d,
            clip_ratio=clip_ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            clip_ratio_c=clip_ratio_c,
        )
        training_logprobs_D.extend(training_logprobs)

        # Optimizer step
        await optim_step(training_client, learning_rate)

    return training_logprobs_D
