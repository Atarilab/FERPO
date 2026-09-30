# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, TanhNormal
from rsl_rl.utils.hl_gauss import HLGaussLayer, embed_targets
from rsl_rl.utils.numerics import (
    finite_tensor_range,
    non_finite_tensor_description,
    tensor_issue_description,
)


def _reppo_fcnn(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    *,
    use_rms_norm: bool,
    activate_input: bool = False,
    output_shape: tuple[int, ...] | None = None,
) -> nn.Sequential:
    """Build the FCNN block used by the reference REPPO implementation."""
    if num_layers < 1:
        raise ValueError(f"`num_layers` must be at least 1, got {num_layers}.")
    layers: list[nn.Module] = []
    if activate_input:
        layers.append(nn.SiLU())
    in_features = input_dim
    for layer_index in range(num_layers):
        is_output = layer_index == num_layers - 1
        out_features = output_dim if is_output else hidden_dim
        layers.append(nn.Linear(in_features, out_features))
        if not is_output:
            if use_rms_norm:
                layers.append(nn.RMSNorm(out_features))
            layers.append(nn.SiLU())
        in_features = out_features
    if output_shape is not None:
        layers.append(nn.Unflatten(-1, output_shape))
    return nn.Sequential(*layers)


class REPPOCritic(nn.Module):
    """REPPO distributional Q critic plus its reward/dynamics predictor."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        vmin: float,
        vmax: float,
        *,
        encoder_layers: int,
        head_layers: int,
        predictor_layers: int,
        use_rms_norm: bool,
    ) -> None:
        super().__init__()
        if head_layers < 1:
            raise ValueError(f"`head_layers` must be at least 1, got {head_layers}.")
        self.feature_encoder = _reppo_fcnn(
            input_dim, hidden_dim, hidden_dim, encoder_layers, use_rms_norm=use_rms_norm
        )
        head: list[nn.Module] = [nn.SiLU()]
        for _ in range(head_layers - 1):
            head.append(nn.Linear(hidden_dim, hidden_dim))
            if use_rms_norm:
                head.append(nn.RMSNorm(hidden_dim))
            head.append(nn.SiLU())
        self.q_head = nn.Sequential(*head)
        self.q_output = HLGaussLayer(
            in_features=hidden_dim,
            min_value=vmin,
            max_value=vmax,
            num_bins=num_bins,
            sigma=0.75,
            offset_mult=40.9,
        )
        self.predictor = _reppo_fcnn(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            predictor_layers,
            use_rms_norm=use_rms_norm,
            activate_input=True,
        )

    def forward(self, critic_input: torch.Tensor) -> torch.Tensor:
        return self.feature_encoder(critic_input)

    def evaluate(
        self, features: torch.Tensor, *, return_logits: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.q_output(self.q_head(features), return_logits=return_logits)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.predictor(features)


class ActorQ(nn.Module):
    is_recurrent: bool = False
    critic_uses_actions: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        actor_hidden_dim: int | None = None,
        critic_hidden_dim: int | None = None,
        num_actor_layers: int = 3,
        num_critic_encoder_layers: int = 2,
        num_critic_head_layers: int = 2,
        num_critic_predictor_layers: int = 2,
        use_actor_norm: bool = True,
        use_critic_norm: bool = True,
        actor_min_std: float = 0.0,
        use_reppo_actor_initialization: bool = False,
        num_critic_bins: int = 151,
        vmin: float = -10.0,
        vmax: float = 10.0,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = True,
        zero_init_mean: bool = False,
        distribution_type: str = "tanh",
        entropy_regularization: bool = True,
        init_alpha_temp: float = 0.1,
        init_alpha_kl: float = 0.1,
        action_lower_bound: float = -1.0,
        action_upper_bound: float = 1.0,
        log_std_min: float | None = None,
        log_std_max: float | None = None,
        smooth_log_std_min: bool = False,
        smooth_log_std_bounds: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        critic_action_dim = num_actions if self.critic_uses_actions else 0
        num_critic_obs = critic_action_dim
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # REPPO uses SiLU throughout its learnable hidden blocks.
        self.activation = "swish"
        self.activation_fn = F.silu
        actor_hidden_dim = int(actor_hidden_dim or actor_hidden_dims[0])
        critic_hidden_dim = int(critic_hidden_dim or critic_hidden_dims[0])
        self.actor_min_std = float(actor_min_std)
        if self.actor_min_std < 0.0:
            raise ValueError(f"`actor_min_std` must be non-negative, got {actor_min_std}.")

        # Actor
        self.state_dependent_std = state_dependent_std
        self.zero_init_mean = bool(zero_init_mean)
        if self.state_dependent_std:
            self.actor = _reppo_fcnn(
                num_actor_obs,
                2 * num_actions,
                actor_hidden_dim,
                num_actor_layers,
                use_rms_norm=use_actor_norm,
                output_shape=(2, num_actions),
            )
        else:
            self.actor = _reppo_fcnn(
                num_actor_obs,
                num_actions,
                actor_hidden_dim,
                num_actor_layers,
                use_rms_norm=use_actor_norm,
            )
        if self.zero_init_mean:
            mean_output_layer = self.actor[-2] if self.state_dependent_std else self.actor[-1]
            if self.state_dependent_std:
                torch.nn.init.zeros_(mean_output_layer.weight[:num_actions])
                torch.nn.init.zeros_(mean_output_layer.bias[:num_actions])
            else:
                torch.nn.init.zeros_(mean_output_layer.weight)
                torch.nn.init.zeros_(mean_output_layer.bias)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # policy distribution
        self.distribution_type = distribution_type
        if self.distribution_type not in ["normal", "tanh"]:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}. Should be 'normal' or 'tanh'.")

        # Critic
        self.num_critic_bins = num_critic_bins
        self.vmin = vmin
        self.vmax = vmax
        self.critic = REPPOCritic(
            num_critic_obs,
            critic_hidden_dim,
            num_critic_bins,
            vmin,
            vmax,
            encoder_layers=num_critic_encoder_layers,
            head_layers=num_critic_head_layers,
            predictor_layers=num_critic_predictor_layers,
            use_rms_norm=use_critic_norm,
        )
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs - critic_action_dim)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.smooth_log_std_min = bool(smooth_log_std_min)
        self.smooth_log_std_bounds = bool(smooth_log_std_bounds)
        if self.log_std_min is not None and self.log_std_max is not None and self.log_std_min > self.log_std_max:
            raise ValueError(
                f"`log_std_min` has to be <= `log_std_max`, got {self.log_std_min} > {self.log_std_max}."
            )
        if self.smooth_log_std_min and self.smooth_log_std_bounds:
            raise ValueError(
                "`smooth_log_std_min` and `smooth_log_std_bounds` are mutually exclusive."
            )
        if self.smooth_log_std_bounds and (
            self.log_std_min is None or self.log_std_max is None
        ):
            raise ValueError(
                "`smooth_log_std_bounds=True` requires both `log_std_min` and "
                "`log_std_max`."
            )
        if self.smooth_log_std_bounds and self.log_std_min == self.log_std_max:
            raise ValueError(
                "`smooth_log_std_bounds=True` requires `log_std_min < log_std_max`."
            )
        if self.state_dependent_std:
            # The reference REPPO actor leaves both halves of its final linear
            # layer at PyTorch's default initialization. Keep the historical
            # RSL-RL initialization available for non-parity configurations.
            if not use_reppo_actor_initialization:
                torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
                if self.noise_std_type == "scalar":
                    torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
                elif self.noise_std_type == "log":
                    initial_log_std = torch.log(torch.tensor(init_noise_std + 1e-7))
                    if self.smooth_log_std_bounds:
                        initial_log_std = self._inverse_smooth_log_std_bounds(
                            initial_log_std,
                            self.log_std_min,
                            self.log_std_max,
                        )
                    elif self.smooth_log_std_min:
                        initial_log_std = self._inverse_smooth_log_std_min(
                            initial_log_std,
                            self.log_std_min if self.log_std_min is not None else -5.0,
                        )
                    torch.nn.init.constant_(
                        self.actor[-2].bias[num_actions:], initial_log_std
                    )
                elif self.noise_std_type == "sigmoid":
                    torch.nn.init.constant_(
                        self.actor[-2].bias[num_actions:],
                        -torch.log(torch.tensor((1.0 / init_noise_std) - 1.0) + 1e-7),
                    )
                else:
                    raise ValueError(
                        f"Unknown standard deviation type: {self.noise_std_type}. "
                        "Should be 'scalar' or 'log'"
                    )
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                initial_log_std = torch.log(init_noise_std * torch.ones(num_actions))
                if self.smooth_log_std_bounds:
                    initial_log_std = self._inverse_smooth_log_std_bounds(
                        initial_log_std,
                        self.log_std_min,
                        self.log_std_max,
                    )
                elif self.smooth_log_std_min and self.log_std_min is not None:
                    initial_log_std = self._inverse_smooth_log_std_min(initial_log_std, self.log_std_min)
                self.log_std = nn.Parameter(initial_log_std)
            elif self.noise_std_type == "sigmoid":
                self.log_std = nn.Parameter(torch.logit((init_noise_std - 1e-4) * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None
        self.num_actions = num_actions
        self.wpo_fisher_preconditioning_enabled = False

        # Trainable temperature parameters
        # Keep a finite parameter even when the effective temperature is zero.
        if init_alpha_temp <= 0.0:
            raise ValueError("init_alpha_temp must be positive; disable entropy_regularization for tau=0.")
        self.entropy_regularization = bool(entropy_regularization)
        self.log_alpha_temp = nn.Parameter(
            torch.log(torch.tensor(init_alpha_temp)),
            requires_grad=self.entropy_regularization,
        )
        self.log_alpha_kl = nn.Parameter(torch.log(torch.tensor(init_alpha_kl)))

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    @property
    def critic_embedding_layer(self) -> HLGaussLayer:
        """Compatibility alias for value-range and HL-Gauss helper code."""
        return self.critic.q_output

    def hlgauss_embed(self, targets: torch.Tensor) -> torch.Tensor:
        return embed_targets(
            targets,
            min_value=self.vmin,
            max_value=self.vmax,
            num_bins=self.num_critic_bins,
        )

    def hlgauss_decode(self, logits: torch.Tensor) -> torch.Tensor:
        return self.critic_embedding_layer.transform.decode(logits)

    def set_critic_value_range(self, vmin: float, vmax: float) -> None:
        """Update HL-Gauss critic support range without changing bin count."""
        if vmax <= vmin:
            raise ValueError(f"`vmax` must be greater than `vmin`, got vmin={vmin}, vmax={vmax}.")
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.critic_embedding_layer.update_value_range(self.vmin, self.vmax)

    @property
    def alpha_temp(self) -> torch.Tensor:
        if not self.entropy_regularization:
            return self.log_alpha_temp.new_zeros(())
        return self.log_alpha_temp.exp()

    @property
    def alpha_kl(self) -> torch.Tensor:
        return self.log_alpha_kl.exp()

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def set_wpo_fisher_preconditioning(self, enabled: bool) -> None:
        self.wpo_fisher_preconditioning_enabled = enabled

    def _precondition_wpo_mean(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if not self.wpo_fisher_preconditioning_enabled:
            return mean
        detached_mean = mean.detach()
        detached_variance = std.detach().square()
        return detached_mean + (mean - detached_mean) * detached_variance

    def _precondition_wpo_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        if not self.wpo_fisher_preconditioning_enabled:
            return log_std
        detached_log_std = log_std.detach()
        return detached_log_std + 0.5 * (log_std - detached_log_std)

    def _clamp_log_std(
        self,
        log_std: torch.Tensor,
        default_min: float | None = None,
        default_max: float | None = None,
    ) -> torch.Tensor:
        log_std_min = self.log_std_min if self.log_std_min is not None else default_min
        log_std_max = self.log_std_max if self.log_std_max is not None else default_max
        if self.smooth_log_std_bounds:
            return self._smooth_log_std_bounds(
                log_std,
                log_std_min,
                log_std_max,
            )
        if log_std_min is not None:
            if self.smooth_log_std_min:
                log_std = log_std_min + F.softplus(log_std - log_std_min)
            else:
                log_std = log_std.clamp_min(log_std_min)
        if log_std_max is not None:
            log_std = log_std.clamp_max(log_std_max)
        return log_std

    @staticmethod
    def _inverse_smooth_log_std_min(log_std: torch.Tensor, log_std_min: float) -> torch.Tensor:
        """Map an effective log std back to the raw smooth-floor parameterization."""
        distance = (log_std - log_std_min).clamp_min(torch.finfo(log_std.dtype).eps)
        inverse_softplus = distance + torch.log(-torch.expm1(-distance))
        return log_std_min + inverse_softplus

    @staticmethod
    def _smooth_log_std_bounds(
        raw_log_std: torch.Tensor,
        log_std_min: float,
        log_std_max: float,
    ) -> torch.Tensor:
        """Smoothly map an unconstrained value into the open log-std interval."""
        return (
            log_std_min
            + F.softplus(raw_log_std - log_std_min)
            - F.softplus(raw_log_std - log_std_max)
        )

    @staticmethod
    def _inverse_smooth_log_std_bounds(
        log_std: torch.Tensor,
        log_std_min: float,
        log_std_max: float,
    ) -> torch.Tensor:
        """Invert the smooth two-sided bound while preserving initialization."""
        if torch.any(log_std <= log_std_min) or torch.any(log_std >= log_std_max):
            raise ValueError(
                "Initial log standard deviation must lie strictly inside the "
                "smooth log-std bounds."
            )
        lower_distance = log_std - log_std_min
        upper_distance = log_std_max - log_std
        return (
            log_std_min
            + torch.log(torch.expm1(lower_distance))
            - torch.log(-torch.expm1(-upper_distance))
        )

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor, mean=None, std=None) -> None:
        if mean is None or std is None:
            if self.state_dependent_std:
                # Compute mean and standard deviation
                mean_and_std = self.actor(obs)
                if self.noise_std_type == "scalar":
                    mean, std = torch.unbind(mean_and_std, dim=-2)
                elif self.noise_std_type == "log":
                    mean, log_std = torch.unbind(mean_and_std, dim=-2)
                    log_std = self._clamp_log_std(log_std)
                    log_std = self._precondition_wpo_log_std(log_std)
                    std = torch.exp(log_std) + self.actor_min_std
                elif self.noise_std_type == "sigmoid":
                    mean, logit_std = torch.unbind(mean_and_std, dim=-2)
                    std = torch.sigmoid(logit_std) + 1e-4
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            else:
                # Compute mean
                mean = self.actor(obs)
                # Compute standard deviation
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    log_std = self._clamp_log_std(self.log_std)
                    std = (torch.exp(log_std) + self.actor_min_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        mean = self._precondition_wpo_mean(mean, std)
        if self.distribution_type == "normal":
            self.distribution = Normal(mean, std)
        elif self.distribution_type == "tanh":
            self.distribution = TanhNormal(mean, std, action_lower_bound=self.action_lower_bound, action_upper_bound=self.action_upper_bound)

    @torch.no_grad()
    def diagnose_action_sampling_failure(
        self,
        obs: TensorDict,
        *,
        context: str,
        error: BaseException,
    ) -> NoReturn:
        """Replace an opaque distribution error with failure-only tensor evidence."""
        tensors: list[tuple[str, torch.Tensor]] = []
        for obs_group in self.obs_groups["policy"]:
            tensors.append((f"policy.raw_observation[{obs_group}]", obs[obs_group]))

        actor_obs = self.get_actor_obs(obs)
        normalized_obs = self.actor_obs_normalizer(actor_obs)
        actor_output = self.actor(normalized_obs)
        tensors.extend(
            (
                ("policy.actor_observation", actor_obs),
                ("policy.normalized_observation", normalized_obs),
                ("policy.actor_output", actor_output),
            )
        )

        if self.state_dependent_std:
            mean, dispersion = torch.unbind(actor_output, dim=-2)
            if self.noise_std_type == "scalar":
                std = dispersion
                tensors.append(("policy.raw_std", dispersion))
            elif self.noise_std_type == "log":
                tensors.append(("policy.raw_log_std", dispersion))
                clamped_log_std = self._clamp_log_std(
                    dispersion,
                )
                tensors.append(("policy.clamped_log_std", clamped_log_std))
                preconditioned_log_std = self._precondition_wpo_log_std(clamped_log_std)
                tensors.append(("policy.preconditioned_log_std", preconditioned_log_std))
                std = torch.exp(preconditioned_log_std) + self.actor_min_std
            elif self.noise_std_type == "sigmoid":
                tensors.append(("policy.logit_std", dispersion))
                std = torch.sigmoid(dispersion) + 1.0e-4
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        else:
            mean = actor_output
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
                tensors.append(("policy.raw_std_parameter", self.std))
            elif self.noise_std_type == "log":
                clamped_log_std = self._clamp_log_std(self.log_std)
                tensors.extend(
                    (
                        ("policy.raw_log_std_parameter", self.log_std),
                        ("policy.clamped_log_std", clamped_log_std),
                    )
                )
                std = (torch.exp(clamped_log_std) + self.actor_min_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

        tensors.append(("policy.raw_mean", mean))
        mean = self._precondition_wpo_mean(mean, std)
        tensors.extend((("policy.mean", mean), ("policy.std", std)))
        distribution = self.distribution
        if distribution is not None:
            if isinstance(getattr(distribution, "loc", None), torch.Tensor):
                tensors.append(("policy.distribution_loc", distribution.loc))
            if isinstance(getattr(distribution, "scale", None), torch.Tensor):
                tensors.append(("policy.distribution_scale", distribution.scale))

        issues = [
            description
            for name, tensor in tensors
            if (
                description := non_finite_tensor_description(
                    name,
                    tensor,
                    environment_axis=0,
                )
            )
            is not None
        ]
        negative_std = tensor_issue_description(
            "policy.std",
            std,
            std < 0.0,
            environment_axis=0,
        )
        if negative_std is not None:
            issues.append(negative_std)
        ranges = ", ".join(
            finite_tensor_range(name, tensor)
            for name, tensor in tensors
            if name
            in {
                "policy.normalized_observation",
                "policy.mean",
                "policy.raw_log_std",
                "policy.clamped_log_std",
                "policy.std",
            }
        )
        evidence = " | ".join(issues) if issues else "all inspected tensors were finite and std was nonnegative"
        raise FloatingPointError(
            f"Policy action sampling failed during {context}: {type(error).__name__}: {error}. "
            f"Diagnostics: {evidence}. Finite ranges: {ranges}."
        ) from error

    def act(self, obs: TensorDict, *args, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        if self.state_dependent_std:
            mean = self.actor(obs)[..., 0, :]
        else:
            mean = self.actor(obs)
        if self.distribution_type == "tanh":
            mean_offset = (self.action_upper_bound + self.action_lower_bound) / 2.0
            scale = (self.action_upper_bound - self.action_lower_bound) / 2.0
            return mean_offset + scale * torch.tanh(mean)
        return mean

    def evaluate(self, obs, act, *args, return_logits=False):
        features = self.critic_features(obs, act)
        return self.critic.evaluate(
            features, return_logits=return_logits
        )

    def critic_features(self, obs: TensorDict, act: Tensor) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        inp = torch.cat([obs, act], dim=-1)
        return self.critic(inp)

    def evaluate_full(
        self, obs: TensorDict, act: Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Q prediction, logits, predicted next features, and features."""
        features = self.critic_features(obs, act)
        value, logits = self.critic.evaluate(features, return_logits=True)
        predicted_features = self.critic.predict(features)
        return value, logits, predicted_features, features

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
