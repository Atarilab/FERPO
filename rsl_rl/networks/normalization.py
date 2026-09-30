# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape: int | tuple[int] | list[int], eps: float = 1e-2, until: int | None = None) -> None:
        """Initialize EmpiricalNormalization module.

        .. note:: The normalization parameters are computed over the whole batch, not for each environment separately.

        Args:
            shape: Shape of input values except batch axis.
            eps: Small value for stability.
            until: If this arg is specified, the module learns input values until the sum of batch sizes exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self) -> torch.Tensor:
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize mean and variance of values based on empirical values."""
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Learn input values without computing the output values of them."""
        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        # Keep the registered buffer allocated outside inference mode. Replacing
        # it while rollout collection runs under ``torch.inference_mode`` turns
        # it into an inference tensor, which cannot later be restored by
        # ``load_state_dict`` outside inference mode.
        self._std.copy_(torch.sqrt(self._var))

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """De-normalize values based on empirical values."""
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize the scale of the
    rewards so that the value function can learn quickly. We did this by dividing the rewards by a running estimate of
    the standard deviation of the sum of discounted rewards.
    """

    def __init__(
        self, shape: int | tuple[int] | list[int], eps: float = 1e-2, gamma: float = 0.99, until: int | None = None
    ) -> None:
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Update discounted rewards
            avg = self.disc_avg.update(rew)
            # Update moments from discounted rewards
            self.emp_norm.update(avg)

        # Normalize rewards with the empirical std
        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


class _DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t
    """

    def __init__(self, gamma: float) -> None:
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg


class DiscountedRewardStdNormalizer(nn.Module):
    """Normalize rewards by the running standard deviation of discounted returns."""

    def __init__(
        self,
        num_envs: int,
        gamma: float = 0.99,
        eps: float = 1.0e-8,
        use_ema: bool = True,
        ema_alpha: float = 0.01,
        clip: float | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if num_envs < 1:
            raise ValueError(f"`num_envs` has to be >= 1, got {num_envs}.")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"`gamma` has to be in [0, 1], got {gamma}.")
        if eps <= 0.0:
            raise ValueError(f"`eps` has to be > 0, got {eps}.")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"`ema_alpha` has to be in (0, 1], got {ema_alpha}.")
        if clip is not None and clip <= 0.0:
            raise ValueError(f"`clip` has to be > 0 when provided, got {clip}.")

        self.num_envs = num_envs
        self.gamma = gamma
        self.eps = eps
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self.clip = clip

        self.register_buffer("discounted_returns", torch.zeros(num_envs, 1, device=device))
        self.register_buffer("_mean", torch.zeros(1, device=device))
        self.register_buffer("_square_mean", torch.zeros(1, device=device))
        self.register_buffer("_var", torch.ones(1, device=device))
        self.register_buffer("_std", torch.ones(1, device=device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=device))
        self.register_buffer("_last_raw_reward_mean", torch.zeros(1, device=device))
        self.register_buffer("_last_normalized_reward_mean", torch.zeros(1, device=device))
        self.register_buffer("_last_normalized_reward_std", torch.zeros(1, device=device))

    @property
    def std(self) -> torch.Tensor:
        return self._std.clone()

    def load_checkpoint_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Restore learned statistics while allowing a different vector-env size.

        ``discounted_returns`` is rollout state, not a learned statistic.  Its
        leading dimension therefore follows the number of environments used by
        the current runner and must be reset when a checkpoint is evaluated
        with a different vectorization level.
        """
        compatible_state = dict(state_dict)
        saved_returns = compatible_state.get("discounted_returns")
        if saved_returns is not None and saved_returns.shape != self.discounted_returns.shape:
            compatible_state["discounted_returns"] = torch.zeros_like(self.discounted_returns)
        self.load_state_dict(compatible_state)

    def forward(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        original_shape = rewards.shape
        rewards = rewards.to(device=self.discounted_returns.device, dtype=self.discounted_returns.dtype).view(-1, 1)
        dones = dones.to(device=self.discounted_returns.device).bool().view(-1, 1)
        if rewards.shape != self.discounted_returns.shape:
            raise ValueError(
                f"Expected rewards with shape compatible with {(self.num_envs,)}, got {tuple(original_shape)}."
            )
        if dones.shape != self.discounted_returns.shape:
            raise ValueError(f"Expected dones with shape compatible with {(self.num_envs,)}, got {tuple(dones.shape)}.")

        if self.training:
            self.discounted_returns.mul_(self.gamma).add_(rewards)
            self._update_statistics(self.discounted_returns)
            normalized_rewards = rewards / (self._std + self.eps)
            self.discounted_returns[dones] = 0.0
        else:
            normalized_rewards = rewards / (self._std + self.eps)

        if self.clip is not None:
            normalized_rewards = normalized_rewards.clamp(-self.clip, self.clip)
        if self.training:
            self._last_raw_reward_mean.copy_(rewards.mean().view_as(self._last_raw_reward_mean))
            self._last_normalized_reward_mean.copy_(
                normalized_rewards.mean().view_as(self._last_normalized_reward_mean)
            )
            self._last_normalized_reward_std.copy_(
                normalized_rewards.std(unbiased=False).view_as(self._last_normalized_reward_std)
            )
        return normalized_rewards.view(original_shape)

    def metrics(self) -> dict[str, float]:
        discounted_returns = self.discounted_returns.view(-1)
        return {
            "std": float(self._std.item()),
            "var": float(self._var.item()),
            "mean_discounted_return": float(self._mean.item()),
            "discounted_return_mean": float(discounted_returns.mean().item()),
            "discounted_return_std": float(discounted_returns.std(unbiased=False).item()),
            "discounted_return_abs_max": float(discounted_returns.abs().max().item()),
            "count": float(self.count.item()),
            "use_ema": 1.0 if self.use_ema else 0.0,
            "ema_alpha": float(self.ema_alpha),
            "gamma": float(self.gamma),
            "raw_reward_batch_mean": float(self._last_raw_reward_mean.item()),
            "normalized_reward_batch_mean": float(self._last_normalized_reward_mean.item()),
            "normalized_reward_batch_std": float(self._last_normalized_reward_std.item()),
        }

    def _update_statistics(self, discounted_returns: torch.Tensor) -> None:
        batch = discounted_returns.view(-1)
        batch_count = batch.numel()
        batch_mean = batch.mean()
        batch_square_mean = batch.square().mean()

        if self.count == 0:
            self._mean.copy_(batch_mean.view_as(self._mean))
            self._square_mean.copy_(batch_square_mean.view_as(self._square_mean))
            self.count += batch_count
        elif self.use_ema:
            alpha = self.ema_alpha
            self._mean.mul_(1.0 - alpha).add_(alpha * batch_mean)
            self._square_mean.mul_(1.0 - alpha).add_(alpha * batch_square_mean)
            self.count += batch_count
        else:
            total_count = self.count + batch_count
            old_weight = self.count.to(dtype=batch.dtype) / total_count.to(dtype=batch.dtype)
            new_weight = torch.as_tensor(batch_count, dtype=batch.dtype, device=batch.device) / total_count.to(
                dtype=batch.dtype
            )
            self._mean.mul_(old_weight).add_(new_weight * batch_mean)
            self._square_mean.mul_(old_weight).add_(new_weight * batch_square_mean)
            self.count.copy_(total_count)

        self._var.copy_((self._square_mean - self._mean.square()).clamp_min(0.0))
        self._std.copy_(torch.sqrt(self._var))
