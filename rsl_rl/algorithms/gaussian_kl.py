# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact KL utilities for diagonal Gaussian policies."""

from __future__ import annotations

import torch
from torch.distributions import Distribution


def diagonal_gaussian_kl_from_stats(
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_std: torch.Tensor,
    *,
    min_std: float = 1.0e-8,
) -> torch.Tensor:
    """Return ``KL(N_old || N_new)`` summed over the action dimension.

    The standard deviations are floored before entering the closed-form
    expression. No epsilon is added to the logarithmic ratio, so identical
    policies evaluate to exactly zero up to floating-point roundoff.
    """
    if min_std <= 0.0:
        raise ValueError(f"`min_std` must be positive, got {min_std}.")
    old_std = old_std.clamp_min(min_std)
    new_std = new_std.clamp_min(min_std)
    old_var = old_std.square()
    new_var = new_std.square()
    mean_delta_sq = (old_mean - new_mean).square()
    per_dimension = torch.log(new_std) - torch.log(old_std) + (old_var + mean_delta_sq) / (2.0 * new_var) - 0.5
    return per_dimension.sum(dim=-1)


def _base_diagonal_gaussian(distribution: Distribution) -> Distribution:
    """Unwrap a shared bijective transform to its diagonal Gaussian base."""
    base = getattr(distribution, "base_dist", distribution)
    while hasattr(base, "base_dist") and not (hasattr(base, "loc") and hasattr(base, "scale")):
        base = base.base_dist
    if not hasattr(base, "loc") or not hasattr(base, "scale"):
        raise TypeError(
            "Analytic policy KL requires a diagonal Gaussian distribution or "
            f"a transformed diagonal Gaussian, got {type(distribution).__name__}."
        )
    return base


def diagonal_gaussian_kl(
    old_distribution: Distribution,
    new_distribution: Distribution,
    *,
    min_std: float = 1.0e-8,
) -> torch.Tensor:
    """Return the exact ``KL(old_distribution || new_distribution)``.

    A KL between policies using the same invertible tanh/affine transform is
    equal to the KL between their Gaussian base distributions.
    """
    old_base = _base_diagonal_gaussian(old_distribution)
    new_base = _base_diagonal_gaussian(new_distribution)
    return diagonal_gaussian_kl_from_stats(
        old_base.loc,
        old_base.scale,
        new_base.loc,
        new_base.scale,
        min_std=min_std,
    )
