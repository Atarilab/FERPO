# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import NamedTuple

import torch


class TargetSnisStats(NamedTuple):
    weights: torch.Tensor
    log_normalizer: torch.Tensor
    entropy: torch.Tensor
    kl_to_reference: torch.Tensor
    weight_ess_normalized: torch.Tensor
    log_ratio_mean: torch.Tensor
    log_ratio_max: torch.Tensor


class StatewiseEssTarget(NamedTuple):
    weights: torch.Tensor
    log_ratio: torch.Tensor
    lambda_lagrangian: torch.Tensor
    ess_normalized: torch.Tensor


def compute_maxent_target_log_ratio(
    q_values: torch.Tensor,
    reference_log_prob: torch.Tensor,
    older_reference_log_prob: torch.Tensor,
    temperature: torch.Tensor,
    lambda_lagrangian: torch.Tensor,
    momentum_factor: torch.Tensor | float,
    eps: float,
    detach_duals: bool = True,
    entropy_multiplier: float = 1.0,
) -> torch.Tensor:
    """Compute the sampled max-ent target log-density ratio."""
    if detach_duals:
        temperature = temperature.detach()
        lambda_lagrangian = lambda_lagrangian.detach()
        if isinstance(momentum_factor, torch.Tensor):
            momentum_factor = momentum_factor.detach()

    entropy_coefficient = entropy_multiplier * temperature
    normalizer = (entropy_coefficient + lambda_lagrangian).clamp_min(eps)
    return (
        q_values
        + momentum_factor * (reference_log_prob - older_reference_log_prob)
        - entropy_coefficient * reference_log_prob
    ) / normalizer


def compute_maxent_dual_value(
    temperature: torch.Tensor,
    lambda_lagrangian: torch.Tensor,
    q_values: torch.Tensor,
    reference_log_prob: torch.Tensor,
    older_reference_log_prob: torch.Tensor,
    momentum_factor: torch.Tensor | float,
    target_entropy: torch.Tensor,
    kl_bound: float | torch.Tensor,
    eps: float,
    entropy_multiplier: float = 1.0,
) -> torch.Tensor:
    log_ratio = compute_maxent_target_log_ratio(
        q_values,
        reference_log_prob,
        older_reference_log_prob,
        temperature,
        lambda_lagrangian,
        momentum_factor,
        eps,
        detach_duals=False,
        entropy_multiplier=entropy_multiplier,
    )
    log_z = torch.logsumexp(log_ratio, dim=0) - torch.log(
        torch.tensor(log_ratio.shape[0], dtype=log_ratio.dtype, device=log_ratio.device)
    )
    entropy_coefficient = entropy_multiplier * temperature
    normalizer = entropy_coefficient + lambda_lagrangian
    return entropy_coefficient * target_entropy - lambda_lagrangian * kl_bound - normalizer * log_z.mean()


def compute_maxent_dual_loss(
    temperature: torch.Tensor,
    lambda_lagrangian: torch.Tensor,
    q_values: torch.Tensor,
    reference_log_prob: torch.Tensor,
    older_reference_log_prob: torch.Tensor,
    momentum_factor: torch.Tensor | float,
    target_entropy: torch.Tensor,
    kl_bound: float | torch.Tensor,
    eps: float,
    update_temperature: bool = True,
    update_lagrangian: bool = True,
    entropy_multiplier: float = 1.0,
) -> torch.Tensor:
    if not update_temperature:
        temperature = temperature.detach()
    if not update_lagrangian:
        lambda_lagrangian = lambda_lagrangian.detach()
        if isinstance(momentum_factor, torch.Tensor):
            momentum_factor = momentum_factor.detach()
    return -compute_maxent_dual_value(
        temperature,
        lambda_lagrangian,
        q_values,
        reference_log_prob,
        older_reference_log_prob,
        momentum_factor,
        target_entropy,
        kl_bound,
        eps,
        entropy_multiplier,
    )


def compute_maxent_target_weights(
    q_values: torch.Tensor,
    reference_log_prob: torch.Tensor,
    older_reference_log_prob: torch.Tensor,
    temperature: torch.Tensor,
    lambda_lagrangian: torch.Tensor,
    momentum_factor: torch.Tensor | float,
    eps: float,
    detach_duals: bool = True,
    self_normalize: bool = True,
    entropy_multiplier: float = 1.0,
) -> torch.Tensor:
    """Compute sampled max-ent target weights over the leading action-sample axis."""
    log_ratio = compute_maxent_target_log_ratio(
        q_values,
        reference_log_prob,
        older_reference_log_prob,
        temperature,
        lambda_lagrangian,
        momentum_factor,
        eps,
        detach_duals=detach_duals,
        entropy_multiplier=entropy_multiplier,
    )
    if self_normalize:
        return torch.softmax(log_ratio, dim=0)
    return torch.exp(log_ratio)


def normalized_effective_sample_size_per_state(weights: torch.Tensor, eps: float) -> torch.Tensor:
    """Return normalized ESS for every state represented after the sample axis."""
    sample_count = weights.shape[0]
    return 1.0 / (sample_count * weights.square().sum(dim=0).clamp_min(eps))


def normalized_effective_sample_size(weights: torch.Tensor, eps: float) -> torch.Tensor:
    """Return mean normalized ESS over states."""
    return normalized_effective_sample_size_per_state(weights, eps).mean()


def compute_statewise_ess_target(
    q_values: torch.Tensor,
    reference_log_prob: torch.Tensor,
    older_reference_log_prob: torch.Tensor,
    temperature: torch.Tensor,
    momentum_factor: torch.Tensor | float,
    target_ess: torch.Tensor | float,
    eps: float,
    bisection_steps: int,
) -> StatewiseEssTarget:
    """Solve the minimum per-state KL multiplier meeting a normalized ESS floor.

    The solve bisects ``log(temperature + lambda)`` between the unconstrained
    denominator and an analytically feasible finite upper bound. All states
    are solved together with tensor operations, without reevaluating the
    policy or action-value critic.
    """
    if bisection_steps < 1:
        raise ValueError(f"`bisection_steps` has to be >= 1, got {bisection_steps}.")
    if q_values.shape != reference_log_prob.shape or q_values.shape != older_reference_log_prob.shape:
        raise ValueError(
            "Statewise ESS inputs must have identical shapes, got "
            f"{tuple(q_values.shape)}, {tuple(reference_log_prob.shape)}, and "
            f"{tuple(older_reference_log_prob.shape)}."
        )
    if q_values.shape[0] < 1:
        raise ValueError("Statewise ESS requires at least one action sample.")

    temperature = temperature.detach().clamp_min(eps)
    if isinstance(momentum_factor, torch.Tensor):
        momentum_factor = momentum_factor.detach()
    min_ess = 1.0 / q_values.shape[0]
    if not isinstance(target_ess, torch.Tensor) and not (min_ess <= target_ess <= 1.0):
        raise ValueError(
            "`target_ess` has to be in the normalized ESS range "
            f"[{min_ess}, 1], got {target_ess}."
        )
    target_ess = torch.as_tensor(target_ess, dtype=q_values.dtype, device=q_values.device)

    score = (
        q_values
        + momentum_factor * (reference_log_prob - older_reference_log_prob)
        - temperature * reference_log_prob
    )
    state_shape = score.shape[1:]
    unconstrained_denominator = temperature.expand(state_shape)
    unconstrained_log_ratio = score / unconstrained_denominator.unsqueeze(0)
    unconstrained_weights = torch.softmax(unconstrained_log_ratio, dim=0)
    unconstrained_ess = normalized_effective_sample_size_per_state(unconstrained_weights, eps)
    unconstrained_feasible = unconstrained_ess >= target_ess

    # If a state's logits span at most delta, then
    # max_i(w_i) <= exp(delta) / K and normalized ESS >= exp(-delta).
    # Using delta = -log(target_ess) / 2 therefore gives a strictly feasible
    # finite upper denominator for every target below one. The eps floor also
    # handles a target represented numerically as exactly one.
    score_max = score.amax(dim=0)
    score_min = score.amin(dim=0)
    score_span = (score_max - score_min).clamp_min(0.0)
    guaranteed_logit_span = (0.5 * -torch.log(target_ess.clamp_max(1.0))).clamp_min(eps)
    max_denominator = torch.tensor(
        torch.finfo(score.dtype).max / 16.0,
        dtype=score.dtype,
        device=score.device,
    )
    denominator_high = torch.maximum(
        unconstrained_denominator,
        score_span / guaranteed_logit_span,
    ).clamp_max(max_denominator)
    log_denominator_low = torch.log(unconstrained_denominator)
    log_denominator_high = torch.where(
        unconstrained_feasible,
        log_denominator_low,
        torch.log(denominator_high),
    )
    centered_score = score - score_max.unsqueeze(0)

    for _ in range(bisection_steps):
        log_denominator_mid = 0.5 * (log_denominator_low + log_denominator_high)
        inverse_denominator_mid = torch.exp(-log_denominator_mid)
        candidate_weights = torch.softmax(
            centered_score * inverse_denominator_mid.unsqueeze(0),
            dim=0,
        )
        candidate_ess = normalized_effective_sample_size_per_state(candidate_weights, eps)
        feasible = candidate_ess >= target_ess
        log_denominator_high = torch.where(
            feasible,
            log_denominator_mid,
            log_denominator_high,
        )
        log_denominator_low = torch.where(
            feasible,
            log_denominator_low,
            log_denominator_mid,
        )

    denominator = torch.exp(log_denominator_high).clamp_max(max_denominator)
    log_ratio = score / denominator.unsqueeze(0)
    weights = torch.softmax(log_ratio, dim=0)
    ess_normalized = normalized_effective_sample_size_per_state(weights, eps)
    lambda_lagrangian = torch.where(
        unconstrained_feasible,
        torch.zeros_like(denominator),
        (denominator - unconstrained_denominator).clamp_min(0.0),
    )
    return StatewiseEssTarget(
        weights=weights,
        log_ratio=log_ratio,
        lambda_lagrangian=lambda_lagrangian,
        ess_normalized=ess_normalized,
    )


def compute_target_snis_stats(
    log_ratio: torch.Tensor,
    reference_log_prob: torch.Tensor,
    eps: float,
) -> TargetSnisStats:
    sample_count = log_ratio.shape[0]
    weights = torch.softmax(log_ratio, dim=0)
    log_normalizer = torch.logsumexp(log_ratio, dim=0, keepdim=True) - torch.log(
        torch.tensor(sample_count, dtype=log_ratio.dtype, device=log_ratio.device)
    )
    log_density_ratio = log_ratio - log_normalizer
    target_log_prob = reference_log_prob + log_density_ratio

    entropy_per_state = -(weights * target_log_prob).sum(dim=0)
    kl_per_state = (weights * log_density_ratio).sum(dim=0)
    weight_ess_per_state = 1.0 / weights.square().sum(dim=0).clamp_min(eps) / sample_count

    return TargetSnisStats(
        weights=weights,
        log_normalizer=log_normalizer.squeeze(0),
        entropy=entropy_per_state.mean(),
        kl_to_reference=kl_per_state.mean(),
        weight_ess_normalized=weight_ess_per_state.mean(),
        log_ratio_mean=log_ratio.mean(),
        log_ratio_max=log_ratio.max(),
    )


def target_effective_sample_size(target_fraction: float, num_action_samples: int, device: torch.device) -> torch.Tensor:
    """Map a [0, 1] interpolation fraction to the normalized ESS range."""
    min_ess = 1.0 / num_action_samples
    target = min_ess + (1.0 - min_ess) * target_fraction
    return torch.tensor(target, dtype=torch.float32, device=device)
