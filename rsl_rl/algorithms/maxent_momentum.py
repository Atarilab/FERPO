# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
from torch.distributions import kl_divergence

MOMENTUM_REFERENCE_MODES = {"previous", "ema"}
MOMENTUM_MARGIN_MODES = {"fixed", "relative_step"}


class MaxEntMomentumMixin:
    """Shared historical-policy momentum support for max-ent algorithms.

    The historical reference is either the preceding rollout policy or an EMA
    of actor parameters. Adaptive momentum uses the same positive log-parameter
    representation and policy optimizer as the temperature and KL duals.
    """

    def _init_maxent_momentum(
        self,
        *,
        momentum_factor: float,
        init_alpha_momentum: float,
        momentum_margin: float,
        momentum_margin_mode: str,
        momentum_step_fraction: float,
        update_momentum_lagrangian: bool,
        momentum_reference_mode: str,
        momentum_ema_beta: float,
    ) -> None:
        if momentum_factor < 0.0:
            raise ValueError(f"`momentum_factor` has to be >= 0, got {momentum_factor}.")
        if init_alpha_momentum <= 0.0:
            raise ValueError(
                f"`init_alpha_momentum` has to be > 0, got {init_alpha_momentum}."
            )
        if momentum_margin < 0.0:
            raise ValueError(f"`momentum_margin` has to be >= 0, got {momentum_margin}.")
        if momentum_margin_mode not in MOMENTUM_MARGIN_MODES:
            allowed_modes = ", ".join(sorted(MOMENTUM_MARGIN_MODES))
            raise ValueError(
                f"`momentum_margin_mode` has to be one of {allowed_modes}, "
                f"got {momentum_margin_mode!r}."
            )
        if momentum_step_fraction < 0.0:
            raise ValueError(
                "`momentum_step_fraction` has to be >= 0, got "
                f"{momentum_step_fraction}."
            )
        if momentum_reference_mode not in MOMENTUM_REFERENCE_MODES:
            allowed_modes = ", ".join(sorted(MOMENTUM_REFERENCE_MODES))
            raise ValueError(
                f"`momentum_reference_mode` has to be one of {allowed_modes}, "
                f"got {momentum_reference_mode!r}."
            )
        if not (0.0 <= momentum_ema_beta < 1.0):
            raise ValueError(
                f"`momentum_ema_beta` has to be in [0, 1), got {momentum_ema_beta}."
            )
        if momentum_margin_mode == "relative_step":
            if momentum_margin != 0.0:
                raise ValueError(
                    "`momentum_margin` must be 0 when "
                    "`momentum_margin_mode='relative_step'`; use "
                    "`momentum_step_fraction` to set the continuation target."
                )
            if not update_momentum_lagrangian:
                raise ValueError(
                    "`momentum_margin_mode='relative_step'` requires "
                    "`update_momentum_lagrangian=True`."
                )

        self.momentum_factor = float(momentum_factor)
        self.init_alpha_momentum = float(init_alpha_momentum)
        self.momentum_margin = float(momentum_margin)
        self.momentum_margin_mode = momentum_margin_mode
        self.momentum_step_fraction = float(momentum_step_fraction)
        self.update_momentum_lagrangian = bool(update_momentum_lagrangian)
        self.momentum_reference_mode = momentum_reference_mode
        self.momentum_ema_beta = float(momentum_ema_beta)

        self.previous_policy = None
        self._has_previous_rollout_policy = False
        explicit_actor_names = {
            name
            for name, _ in self.policy.named_parameters()
            if self._is_explicit_actor_parameter_name(name)
        }
        if explicit_actor_names:
            self._momentum_actor_parameter_names = explicit_actor_names
        else:
            # Small custom policies used by downstream projects may expose a
            # policy parameter without the standard ActorQ module prefixes.
            self._momentum_actor_parameter_names = {
                name
                for name, _ in self.policy.named_parameters()
                if not name.startswith(("critic.", "critic_embedding_layer.", "norm."))
                and not name.startswith("log_alpha_")
            }

        if self.update_momentum_lagrangian:
            initial_log_momentum = torch.log(
                torch.tensor(
                    self.init_alpha_momentum,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            log_alpha_momentum = torch.nn.Parameter(initial_log_momentum)
            self.policy.register_parameter("log_alpha_momentum", log_alpha_momentum)
            self.old_policy.register_parameter(
                "log_alpha_momentum",
                torch.nn.Parameter(initial_log_momentum.detach().clone(), requires_grad=False),
            )
            self.optimizer.add_param_group(
                {"params": [log_alpha_momentum], "lr": self.learning_rate}
            )

        if self._uses_momentum():
            self.previous_policy = copy.deepcopy(self.policy)
            self.previous_policy.to(self.device)
            self.previous_policy.eval()

    def _uses_momentum(self) -> bool:
        return bool(
            getattr(self, "update_momentum_lagrangian", False)
            or getattr(self, "momentum_factor", 0.0) > 0.0
        )

    def _momentum_coefficient(self) -> torch.Tensor:
        if getattr(self, "update_momentum_lagrangian", False):
            return self.policy.log_alpha_momentum.exp()
        return self.policy.alpha_temp.new_tensor(getattr(self, "momentum_factor", 0.0))

    @staticmethod
    def _is_explicit_actor_parameter_name(name: str) -> bool:
        return (
            name.startswith("actor.")
            or name.startswith("actor_cnns.")
            or name.startswith("memory_a.")
            or name in {"std", "log_std"}
        )

    @staticmethod
    def _is_actor_normalizer_state_name(name: str) -> bool:
        return name.startswith("actor_obs_normalizer.")

    def _copy_actor_normalizer_from_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.previous_policy is None:
            return
        reference_state = self.previous_policy.state_dict()
        with torch.no_grad():
            for name, value in state_dict.items():
                if self._is_actor_normalizer_state_name(name) and name in reference_state:
                    reference_state[name].copy_(value)

    def _update_momentum_reference(self, rollout_policy_state: dict[str, torch.Tensor]) -> None:
        if self.previous_policy is None:
            return
        if self.momentum_reference_mode == "previous":
            self.previous_policy.load_state_dict(rollout_policy_state)
            self.previous_policy.eval()
            return

        reference_state = self.previous_policy.state_dict()
        beta = self.momentum_ema_beta
        with torch.no_grad():
            for name, rollout_value in rollout_policy_state.items():
                if name not in reference_state:
                    continue
                reference_value = reference_state[name]
                if (
                    name in self._momentum_actor_parameter_names
                    and torch.is_floating_point(reference_value)
                ):
                    reference_value.mul_(beta).add_(rollout_value, alpha=1.0 - beta)
                elif self._is_actor_normalizer_state_name(name):
                    # Actor normalization defines the coordinate system in
                    # which the policy is evaluated. Share the rollout
                    # snapshot rather than averaging normalization statistics.
                    reference_value.copy_(rollout_value)
        self.previous_policy.eval()

    def _prepare_momentum_update(self) -> dict[str, torch.Tensor] | None:
        if not self._uses_momentum():
            return None

        rollout_policy_state = copy.deepcopy(self.policy.state_dict())
        if not self._has_previous_rollout_policy:
            self.previous_policy.load_state_dict(rollout_policy_state)
            self.previous_policy.eval()
        else:
            # Normalization statistics change while collecting a rollout.
            # Compare actor parameters using the current rollout coordinate
            # system, not stale historical normalization statistics.
            self._copy_actor_normalizer_from_state(rollout_policy_state)
        return rollout_policy_state

    def _finalize_momentum_update(
        self,
        rollout_policy_state: dict[str, torch.Tensor],
        metrics: dict[str, float],
    ) -> dict[str, float]:
        self._update_momentum_reference(rollout_policy_state)
        self._has_previous_rollout_policy = True
        return metrics

    def update(self) -> dict[str, float]:
        """Run an update while advancing the configured momentum reference."""
        if not self._uses_momentum():
            return super().update()

        rollout_policy_state_dict = self._prepare_momentum_update()
        metrics = super().update()
        return self._finalize_momentum_update(
            rollout_policy_state_dict,
            metrics,
        )

    def broadcast_parameters(self) -> None:
        """Broadcast trainable parameters and reset the local momentum reference."""
        super().broadcast_parameters()
        self.sync_previous_policy_to_current()

    def _estimate_momentum_kls(
        self,
        predicted_policy,
        old_policy_distribution,
        previous_policy_distribution,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # KL is invariant under the shared bijective tanh/affine transform, so
        # diagonal TanhNormal policies have the exact KL of their base Normals.
        current_base = getattr(predicted_policy, "base_dist", predicted_policy)
        old_base = getattr(old_policy_distribution, "base_dist", old_policy_distribution)
        previous_base = getattr(
            previous_policy_distribution,
            "base_dist",
            previous_policy_distribution,
        )
        with torch.no_grad():
            kl_roll = kl_divergence(current_base, old_base).sum(dim=-1)
            kl_previous = kl_divergence(current_base, previous_base).sum(dim=-1)
            kl_adjacent = kl_divergence(old_base, previous_base).sum(dim=-1)
        return kl_roll, kl_previous, kl_adjacent

    def _effective_momentum_margin(self, adjacent_kl: torch.Tensor) -> torch.Tensor:
        """Return the continuation margin relative to the historical reference.

        ``adjacent_kl`` compares the rollout policy with the configured
        historical reference.  For ``previous`` this is the consecutive-policy
        step used by the original relative-step formulation.  For ``ema`` it is
        the rollout-to-EMA gap, so the same fraction requests continuation away
        from the smoothed historical reference.
        """
        if self.momentum_margin_mode == "relative_step":
            return (1.0 + 2.0 * self.momentum_step_fraction) * adjacent_kl
        return adjacent_kl.new_tensor(self.momentum_margin)

    def _momentum_lagrange_loss_and_metrics(
        self,
        predicted_policy,
        old_policy_distribution,
        previous_policy_distribution,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        kl_roll, kl_previous, kl_adjacent = self._estimate_momentum_kls(
            predicted_policy,
            old_policy_distribution,
            previous_policy_distribution,
        )
        kl_roll_mean = kl_roll.mean()
        kl_previous_mean = kl_previous.mean()
        kl_adjacent_mean = kl_adjacent.mean()
        momentum_gap = kl_roll_mean - kl_previous_mean
        effective_margin = self._effective_momentum_margin(kl_adjacent_mean)
        momentum_violation = momentum_gap + effective_margin
        momentum_lagrange_loss = torch.zeros((), device=self.device)
        if self.update_momentum_lagrangian:
            momentum_lagrange_loss = (
                -self._momentum_coefficient() * momentum_violation.detach()
            )
        return momentum_lagrange_loss, {
            "momentum_kl_roll": kl_roll_mean,
            "momentum_kl_previous": kl_previous_mean,
            "momentum_kl_adjacent": kl_adjacent_mean,
            "momentum_gap": momentum_gap,
            "momentum_effective_margin": effective_margin,
            "momentum_step_fraction": effective_margin.new_tensor(
                self.momentum_step_fraction
            ),
            "momentum_relative_step_mode": effective_margin.new_tensor(
                float(self.momentum_margin_mode == "relative_step")
            ),
            "momentum_violation": momentum_violation,
        }

    def sync_previous_policy_to_current(self) -> None:
        if self.previous_policy is None:
            return
        self.previous_policy.load_state_dict(self.policy.state_dict())
        self.previous_policy.eval()
        self._has_previous_rollout_policy = False

    def policy_snapshot_state_dict(self) -> dict[str, object]:
        """Serialize momentum history needed for update-equivalent resume."""
        return {
            "previous_policy": (
                None
                if self.previous_policy is None
                else self.previous_policy.state_dict()
            ),
            "has_previous_rollout_policy": self._has_previous_rollout_policy,
        }

    def load_policy_snapshot_state_dict(
        self,
        state_dict: dict[str, object] | None,
    ) -> None:
        """Restore momentum history, falling back for legacy checkpoints."""
        if state_dict is None:
            self.sync_previous_policy_to_current()
            return
        previous_state = state_dict.get("previous_policy")
        if self.previous_policy is None:
            if previous_state is not None:
                raise ValueError(
                    "Checkpoint contains momentum history, but momentum is "
                    "disabled in the current algorithm configuration."
                )
        else:
            if not isinstance(previous_state, dict):
                raise ValueError(
                    "Checkpoint is missing the previous momentum policy state."
                )
            self.previous_policy.load_state_dict(previous_state)
            self.previous_policy.eval()
        self._has_previous_rollout_policy = bool(
            state_dict.get("has_previous_rollout_policy", False)
        )
