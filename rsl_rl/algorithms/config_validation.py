# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-algorithm configuration invariants."""

from __future__ import annotations


def validate_critic_reward_normalization(
    critic_loss_type: str | None,
    normalize_rewards: bool,
) -> None:
    """Reject reward normalization for distributional HL-Gauss critics."""
    if critic_loss_type == "hl_gauss" and normalize_rewards:
        raise ValueError(
            "HL-Gauss critic loss cannot be combined with reward "
            "normalization. Set `normalize_rewards: false` when "
            "`critic_loss_type: hl_gauss`."
        )
