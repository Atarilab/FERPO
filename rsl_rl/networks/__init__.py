# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for components of modules."""

import torch

from .normalization import DiscountedRewardStdNormalizer, EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from .tanh_distribution import log_prob_from_tanh_normal, TanhNormal

HiddenState = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None

__all__ = [
    "DiscountedRewardStdNormalizer",
    "EmpiricalDiscountedVariationNormalization",
    "EmpiricalNormalization",
    "HiddenState",
    "log_prob_from_tanh_normal",
    "TanhNormal",
]
