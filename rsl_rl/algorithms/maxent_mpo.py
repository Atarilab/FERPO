# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch.distributions import Normal

from rsl_rl.networks import TanhNormal

from .gaussian_kl import diagonal_gaussian_kl
from .maxent_utils import (
    compute_maxent_target_log_ratio,
    compute_statewise_ess_target,
    normalized_effective_sample_size_per_state,
    target_effective_sample_size,
)
from .reppo import REPPO


Q_WEIGHT_ESS_MODES = {"mean", "mean_log", "statewise"}


def _temperature_dual_loss(
    log_alpha_temp: torch.Tensor,
    entropy_gap: torch.Tensor,
    *,
    use_log_coordinate_loss: bool,
) -> torch.Tensor:
    """Return a detached-gap temperature loss in the selected dual coordinate."""
    coordinate = log_alpha_temp if use_log_coordinate_loss else log_alpha_temp.exp()
    return coordinate * entropy_gap.detach()


class MaxEntMPO(REPPO):
    """Maximum-entropy forward projection variant of REPPO.

    The actor minimizes a sampled forward-KL projection onto the max-ent target
    density induced by the critic and the frozen rollout policy.
    """

    def __init__(
        self,
        *args,
        num_action_samples: int = 16,
        action_proposal_std_scale: float = 1.0,
        target_q_weight_ess_fraction: float = 0.8,
        self_normalize_q_weights: bool = True,
        q_weight_ess_mode: str = "mean",
        statewise_ess_bisection_steps: int = 8,
        maxent_eps: float = 1.0e-8,
        update_entropy_lagrangian: bool = True,
        use_log_temperature_dual_loss: bool = False,
        update_kl_lagrangian: bool = True,
        lambda_constraint: str = "ess",
        forward_kl_bound: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if num_action_samples < 1:
            raise ValueError(f"`num_action_samples` has to be >= 1, got {num_action_samples}.")
        if not (1.0 <= action_proposal_std_scale < float("inf")):
            raise ValueError("`action_proposal_std_scale` must be finite and >= 1.")
        if action_proposal_std_scale != 1.0 and q_weight_ess_mode == "statewise":
            raise ValueError(
                "Wider action proposals cannot use the statewise ESS solver: "
                "proposal-corrected ESS is not monotone in lambda."
            )
        if not (0.0 <= target_q_weight_ess_fraction <= 1.0):
            raise ValueError(
                "`target_q_weight_ess_fraction` has to be in [0, 1], got "
                f"{target_q_weight_ess_fraction}."
            )
        if maxent_eps <= 0.0:
            raise ValueError(f"`maxent_eps` has to be > 0, got {maxent_eps}.")
        if q_weight_ess_mode not in Q_WEIGHT_ESS_MODES:
            allowed_modes = ", ".join(sorted(Q_WEIGHT_ESS_MODES))
            raise ValueError(
                f"`q_weight_ess_mode` has to be one of {allowed_modes}, "
                f"got {q_weight_ess_mode!r}."
            )
        if statewise_ess_bisection_steps < 1:
            raise ValueError(
                "`statewise_ess_bisection_steps` has to be >= 1, got "
                f"{statewise_ess_bisection_steps}."
            )
        if q_weight_ess_mode == "statewise" and not self_normalize_q_weights:
            raise ValueError("`q_weight_ess_mode='statewise'` requires self-normalized target weights.")
        if q_weight_ess_mode == "statewise" and update_kl_lagrangian:
            raise ValueError(
                "`q_weight_ess_mode='statewise'` solves lambda per state and requires "
                "`update_kl_lagrangian=False`."
            )

        if lambda_constraint not in {"ess", "forward_kl"}:
            raise ValueError("lambda_constraint must be 'ess' or 'forward_kl'.")
        if not (0.0 < forward_kl_bound < float("inf")):
            raise ValueError("forward_kl_bound must be finite and positive.")
        if lambda_constraint == "forward_kl" and q_weight_ess_mode == "statewise":
            raise ValueError("Forward KL control cannot use the statewise ESS solver.")
        if not getattr(self.policy, "entropy_regularization", True) and update_entropy_lagrangian:
            raise ValueError("tau=0 requires update_entropy_lagrangian=False.")
        self.lambda_constraint = lambda_constraint
        self.forward_kl_bound = float(forward_kl_bound)
        self.num_action_samples = num_action_samples
        self.action_proposal_std_scale = float(action_proposal_std_scale)
        self.target_q_weight_ess_fraction = target_q_weight_ess_fraction
        self.self_normalize_q_weights = self_normalize_q_weights
        self.q_weight_ess_mode = q_weight_ess_mode
        self.statewise_ess_bisection_steps = statewise_ess_bisection_steps
        self.maxent_eps = maxent_eps
        self.update_entropy_lagrangian = update_entropy_lagrangian
        self.use_log_temperature_dual_loss = bool(use_log_temperature_dual_loss)
        self.update_kl_lagrangian = update_kl_lagrangian
        self._last_statewise_lagrangian: torch.Tensor | None = None

    def _target_ess(self) -> torch.Tensor:
        return target_effective_sample_size(
            self.target_q_weight_ess_fraction,
            self.num_action_samples,
            self.policy.log_alpha_kl.device,
        )

    def _weighted_projection_loss(self, log_prob_new: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_log_prob = weights * log_prob_new
        if self.self_normalize_q_weights:
            return -weighted_log_prob.sum(dim=0).mean()
        return -weighted_log_prob.mean()

    def _ess_lagrange_loss_and_metrics(
        self,
        weights: torch.Tensor,
        target_ess: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.self_normalize_q_weights:
            diagnostic_weights = weights
        else:
            diagnostic_weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(self.maxent_eps)
        ess_per_state = normalized_effective_sample_size_per_state(diagnostic_weights, self.maxent_eps)
        ess_mean = ess_per_state.mean()
        negative_log_ess_mean = -torch.log(ess_per_state.clamp_min(self.maxent_eps)).mean()
        target_negative_log_ess = -torch.log(target_ess.clamp_min(self.maxent_eps))

        lambda_ess_loss = torch.zeros((), device=self.device)
        if self.update_kl_lagrangian and self.lambda_constraint == "ess":
            if self.q_weight_ess_mode == "mean":
                ess_gap = ess_mean - target_ess
            elif self.q_weight_ess_mode == "mean_log":
                ess_gap = target_negative_log_ess - negative_log_ess_mean
            else:
                raise RuntimeError("Statewise ESS mode must not update the global KL multiplier.")
            lambda_ess_loss = self.policy.alpha_kl * ess_gap.detach()

        statewise_lagrangian = self._last_statewise_lagrangian
        if statewise_lagrangian is None:
            statewise_lagrangian = torch.zeros_like(ess_per_state)
        return lambda_ess_loss, {
            "ess_normalized": ess_mean,
            "ess_normalized_min": ess_per_state.min(),
            "ess_negative_log_mean": negative_log_ess_mean,
            "target_ess_negative_log": target_negative_log_ess,
            "ess_violation_fraction": (ess_per_state < target_ess).float().mean(),
            "statewise_lambda_mean": statewise_lagrangian.mean(),
            "statewise_lambda_max": statewise_lagrangian.max(),
            "statewise_lambda_active_fraction": (statewise_lagrangian > self.maxent_eps).float().mean(),
        }

    def _maxent_weights(
        self,
        q_values: torch.Tensor,
        reference_log_prob: torch.Tensor,
        proposal_log_prob: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "q_weight_ess_mode", "mean") == "statewise":
            if proposal_log_prob is not None:
                raise ValueError("The statewise ESS solver requires rollout-policy action samples.")
            target = compute_statewise_ess_target(
                q_values,
                reference_log_prob,
                self.policy.alpha_temp,
                self._target_ess(),
                self.maxent_eps,
                self.statewise_ess_bisection_steps,
            )
            self._last_statewise_lagrangian = target.lambda_lagrangian.detach()
            return target.weights.detach(), target.log_ratio.detach()

        self._last_statewise_lagrangian = None
        log_ratio = compute_maxent_target_log_ratio(
            q_values,
            reference_log_prob,
            self.policy.alpha_temp,
            self.policy.alpha_kl,
            self.maxent_eps,
            detach_duals=True,
            entropy_multiplier=1.0,
        )
        if proposal_log_prob is not None:
            # The existing log ratio is log(q* / pi_roll). Change only the
            # sampling measure: log(q* / r) = log(q* / pi_roll) + log(pi_roll / r).
            log_ratio = log_ratio + reference_log_prob - proposal_log_prob
        if self.self_normalize_q_weights:
            weights = torch.softmax(log_ratio, dim=0)
        else:
            weights = torch.exp(log_ratio)
        return weights.detach(), log_ratio.detach()

    def _sample_wider_proposal(
        self, reference_distribution: Normal | TanhNormal,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sample a wider Gaussian, retaining tanh latents for every density."""
        if not isinstance(reference_distribution, (Normal, TanhNormal)):
            raise TypeError("Wider action proposals require a Normal or TanhNormal policy.")
        scale = reference_distribution.scale * self.action_proposal_std_scale
        sample_shape = torch.Size([self.num_action_samples])
        proposal = Normal(reference_distribution.loc, scale)
        if isinstance(reference_distribution, TanhNormal):
            # Widen in latent coordinates; preserve the rollout action transform.
            pre_tanh = proposal.sample(sample_shape)
            actions = reference_distribution.action_from_pre_tanh(pre_tanh)
            reference_log_prob = reference_distribution.log_prob_from_pre_tanh(pre_tanh).sum(dim=-1)
            # The shared tanh/affine Jacobian cancels in pi_roll / r. Evaluate
            # the correction in latent space to avoid atanh of saturated actions.
            log_correction = (
                reference_distribution.base_dist.log_prob(pre_tanh)
                - proposal.log_prob(pre_tanh)
            ).sum(dim=-1)
            return actions, reference_log_prob, reference_log_prob - log_correction, pre_tanh
        actions = proposal.sample(sample_shape)
        return (
            actions,
            reference_distribution.log_prob(actions).sum(dim=-1),
            proposal.log_prob(actions).sum(dim=-1),
            None,
        )

    def _sample_weighted_actions(
        self,
        obs_batch,
        hidden_states_batch,
        masks_batch,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        object,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        with torch.no_grad():
            self._policy_act(self.old_policy, obs_batch, hidden_states_batch, masks_batch)
            old_policy_distribution = self.old_policy.distribution
            proposal_log_prob = None
            pre_tanh = None
            if getattr(self, "action_proposal_std_scale", 1.0) != 1.0:
                old_policy_actions, reference_log_prob, proposal_log_prob, pre_tanh = (
                    self._sample_wider_proposal(old_policy_distribution)
                )
            elif hasattr(old_policy_distribution, "sample_and_log_prob"):
                old_policy_actions, reference_log_prob = (
                    old_policy_distribution.sample_and_log_prob(
                        torch.Size([self.num_action_samples])
                    )
                )
                reference_log_prob = reference_log_prob.sum(dim=-1)
            else:
                old_policy_actions = old_policy_distribution.sample(
                    (self.num_action_samples,)
                )
                reference_log_prob = old_policy_distribution.log_prob(
                    old_policy_actions
                ).sum(dim=-1)

            q_values = self._evaluate_sampled_action_values(
                obs_batch,
                old_policy_actions,
                hidden_states_batch,
                masks_batch,
            )
            if proposal_log_prob is None:
                weights, log_ratio = self._maxent_weights(q_values, reference_log_prob)
            else:
                weights, log_ratio = self._maxent_weights(
                    q_values, reference_log_prob, proposal_log_prob
                )
        return (
            old_policy_actions,
            q_values.detach(),
            weights,
            log_ratio,
            old_policy_distribution,
            pre_tanh,
            proposal_log_prob,
        )

    def _actor_samples(self, minibatch: dict) -> tuple:
        """Provide candidate scores and densities for the shared forward-KL fit."""
        return self._sample_weighted_actions(
            minibatch["obs_batch"], minibatch["hidden_states_batch"], minibatch["masks_batch"]
        )

    def update_actor(self, minibatch: dict) -> dict:
        obs_batch = minibatch["obs_batch"]
        hidden_states_batch = minibatch["hidden_states_batch"]
        masks_batch = minibatch["masks_batch"]

        self._policy_act(self.policy, obs_batch, hidden_states_batch, masks_batch)
        predicted_policy = self.policy.distribution
        entropy = self._estimate_policy_entropy(predicted_policy)

        (
            old_policy_actions,
            q_values,
            weights,
            log_ratio,
            old_policy_distribution,
            pre_tanh,
            proposal_log_prob,
        ) = self._actor_samples(minibatch)
        log_prob_new = (
            predicted_policy.log_prob_from_pre_tanh(pre_tanh)
            if pre_tanh is not None
            else predicted_policy.log_prob(old_policy_actions)
        ).sum(dim=-1)
        policy_loss = self._weighted_projection_loss(log_prob_new, weights)

        temp_target_loss = torch.zeros((), device=self.device)
        if self.update_entropy_lagrangian:
            temp_target_loss = _temperature_dual_loss(
                self.policy.log_alpha_temp,
                entropy.mean() - self.target_entropy,
                use_log_coordinate_loss=self.use_log_temperature_dual_loss,
            )

        target_ess = self._target_ess()
        lambda_ess_loss, ess_metrics = self._ess_lagrange_loss_and_metrics(weights, target_ess)
        ess_normalized = ess_metrics["ess_normalized"]
        # Old -> current policy KL, averaged over rollout minibatch states.
        # Exact for this ActorQ's shared tanh/affine Gaussian transform.
        # This replaces the lambda controller only, preserving the SNIS target.
        with torch.no_grad():
            rollout_forward_kl = diagonal_gaussian_kl(
                old_policy_distribution, predicted_policy
            ).mean()
        lambda_forward_kl_loss = torch.zeros((), device=self.device)
        if self.lambda_constraint == "forward_kl" and self.update_kl_lagrangian:
            # Minimization increases lambda when achieved KL exceeds its bound.
            lambda_forward_kl_loss = self.policy.alpha_kl * (
                self.forward_kl_bound - rollout_forward_kl
            ).detach()

        self._set_critic_grad(False)
        self.optimizer.zero_grad()
        actor_loss = (policy_loss + temp_target_loss + lambda_ess_loss
                      + lambda_forward_kl_loss)
        actor_loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._set_critic_grad(True)

        diagnostic_weights = weights
        if not self.self_normalize_q_weights:
            diagnostic_weights = torch.softmax(log_ratio, dim=0)
        diagnostic_log_prob = log_prob_new
        if proposal_log_prob is not None:
            diagnostic_log_prob = log_prob_new - proposal_log_prob
        normalized_new_log_prob = diagnostic_log_prob - torch.logsumexp(diagnostic_log_prob, dim=0, keepdim=True)
        forward_kl = (
            diagnostic_weights * (torch.log(diagnostic_weights + self.maxent_eps) - normalized_new_log_prob)
        ).sum(dim=0).mean()
        weighted_q = (diagnostic_weights * q_values).sum(dim=0).mean()
        weight_entropy = -(diagnostic_weights * torch.log(diagnostic_weights + self.maxent_eps)).sum(dim=0).mean()
        active_support = (diagnostic_weights > 0).float().sum(dim=0).mean()
        if self._last_statewise_lagrangian is None:
            normalizer = self.policy.alpha_temp + self.policy.alpha_kl
        else:
            normalizer = self.policy.alpha_temp + self._last_statewise_lagrangian
        metrics = {
            "actor_loss": actor_loss.item(),
            "entropy": entropy.mean().item(),
            "MaxEntMPO/policy_loss": policy_loss.item(),
            "MaxEntMPO/action_proposal_std_scale": self.action_proposal_std_scale,
            "MaxEntMPO/temperature_loss": temp_target_loss.item(),
            "MaxEntMPO/temperature_learning_rate": self.temperature_learning_rate,
            "MaxEntMPO/entropy_error": (entropy.mean() - self.target_entropy).item(),
            "MaxEntMPO/use_log_temperature_dual_loss": float(
                self.use_log_temperature_dual_loss
            ),
            "MaxEntMPO/lambda_ess_loss": lambda_ess_loss.item(),
            "MaxEntMPO/lambda_forward_kl_loss": lambda_forward_kl_loss.item(),
            "MaxEntMPO/rollout_forward_kl": rollout_forward_kl.item(),
            "MaxEntMPO/forward_kl_bound": self.forward_kl_bound,
            "MaxEntMPO/forward_kl_gap": rollout_forward_kl.item() - self.forward_kl_bound,
            "MaxEntMPO/forward_kl_controller": float(self.lambda_constraint == "forward_kl"),
            "MaxEntMPO/forward_kl": forward_kl.item(),
            "MaxEntMPO/weighted_q": weighted_q.item(),
            "MaxEntMPO/ess_normalized": ess_normalized.item(),
            "MaxEntMPO/target_ess_normalized": target_ess.item(),
            "MaxEntMPO/weight_entropy": weight_entropy.item(),
            "MaxEntMPO/weight_support": active_support.item(),
            "MaxEntMPO/maxent_normalizer": normalizer.mean().item(),
            "MaxEntMPO/log_ratio_mean": log_ratio.mean().item(),
            "MaxEntMPO/log_ratio_max": log_ratio.max().item(),
            **{f"MaxEntMPO/{key}": value.item() for key, value in ess_metrics.items()},
            **self._lagrange_metrics(),
        }
        return metrics
