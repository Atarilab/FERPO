"""Tanh-squashed Gaussian distribution for bounded continuous action spaces.

This module implements a Gaussian distribution transformed through tanh to bound
outputs to [-1, 1]. Commonly used in SAC and other algorithms requiring bounded actions.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import TanhTransform


class TanhNormal(TransformedDistribution):
    """Tanh-squashed Normal distribution.

    A Normal distribution passed through a tanh transform, resulting in samples
    bounded to (-1, 1). Uses numerically stable log probability computation.

    Args:
        loc: Mean of the underlying Normal distribution.
        scale: Standard deviation of the underlying Normal distribution.
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    has_rsample = True

    def __init__(self, loc: Tensor, scale: Tensor, validate_args: bool | None = None, action_lower_bound: Tensor | float | None = None, action_upper_bound: Tensor | float | None = None) -> None:
        self.loc = loc
        self.scale = scale
        self.action_bounds_lower = action_lower_bound
        self.action_bounds_upper = action_upper_bound
        # comute mean offset as intermediate between upper and lower bound
        if self.action_bounds_lower is not None and self.action_bounds_upper is not None:
            self._mean_offset = (self.action_bounds_upper + self.action_bounds_lower) / 2.0
            self._scale = (self.action_bounds_upper - self.action_bounds_lower) / 2.0
            # Convert to tensor if they're not already
            if not isinstance(self._mean_offset, Tensor):
                self._mean_offset = torch.tensor(self._mean_offset, dtype=loc.dtype, device=loc.device)
            if not isinstance(self._scale, Tensor):
                self._scale = torch.tensor(self._scale, dtype=loc.dtype, device=loc.device)
        else:
            self._mean_offset = torch.tensor(0.0, dtype=loc.dtype, device=loc.device)
            self._scale = torch.tensor(1.0, dtype=loc.dtype, device=loc.device)
        base_dist = Normal(loc, scale, validate_args=validate_args)
        self._cached_action: Tensor | None = None
        self._cached_pre_tanh: Tensor | None = None
        # cache_size=1 caches the transform for rsample -> log_prob pattern
        super().__init__(base_dist, TanhTransform(cache_size=1), validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        """Mean of the distribution (tanh of the underlying mean)."""
        return self._mean_offset + self._scale * torch.tanh(self.loc)

    @property
    def mode(self) -> Tensor:
        """Mode of the distribution (same as mean for symmetric distributions)."""
        return self.mean

    @property
    def stddev(self) -> Tensor:
        """Standard deviation of the underlying Normal distribution."""
        return self.scale * (1 - torch.tanh(self.loc) ** 2)

    def log_prob(self, value: Tensor) -> Tensor:
        """Compute log probability with numerically stable Jacobian correction.

        Uses the identity: log(1 - tanh²(x)) = 2 * (log(2) - x - softplus(-2x))
        which is more stable than direct computation for extreme values.

        Args:
            value: Samples in (-1, 1) to evaluate.

        Returns:
            Log probability of the samples.
        """
        if value is self._cached_action and self._cached_pre_tanh is not None:
            return self.log_prob_from_pre_tanh(self._cached_pre_tanh)

        if self._validate_args:
            self._validate_sample(value)

        # recenter action
        value = (value - self._mean_offset) / self._scale

        # Inverse tanh (atanh) to get the pre-squashed value
        # Clamp to avoid numerical issues at boundaries
        eps = torch.finfo(value.dtype).eps
        value = value.clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh = torch.atanh(value)

        # Log prob of the base Normal distribution
        log_prob = self.base_dist.log_prob(pre_tanh)

        # Numerically stable log det Jacobian: -log(1 - tanh²(x))
        # Using: log(1 - tanh²(x)) = 2 * (log(2) - x - softplus(-2x))
        log_det_jacobian = 2.0 * (math.log(2.0) - pre_tanh - nn.functional.softplus(-2.0 * pre_tanh))

        return log_prob - log_det_jacobian - torch.log(self._scale)

    def action_from_pre_tanh(self, pre_tanh_value: Tensor) -> Tensor:
        """Transform a retained latent sample into the bounded action space."""
        return self._mean_offset + self._scale * torch.tanh(pre_tanh_value)

    def log_prob_from_pre_tanh(self, pre_tanh_value: Tensor) -> Tensor:
        """Compute log probability without numerically inverting a sampled action.

        Keeping the latent sample is important in float32: ``tanh`` can round an
        otherwise finite latent value to exactly ``-1`` or ``1``. Reconstructing
        that value with ``atanh`` then evaluates the density at a different
        latent point.
        """
        log_prob = self.base_dist.log_prob(pre_tanh_value)
        log_det_jacobian = 2.0 * (
            math.log(2.0)
            - pre_tanh_value
            - nn.functional.softplus(-2.0 * pre_tanh_value)
        )
        return log_prob - log_det_jacobian - torch.log(self._scale)

    def action_score_from_pre_tanh(self, pre_tanh_value: Tensor) -> Tensor:
        """Return ``d log pi(a) / da`` using the retained pre-tanh value.

        This is the exact elementwise score of the affine tanh-transformed
        diagonal Normal. The stable log-Jacobian expression avoids an
        ``atanh(tanh(z))`` round trip and remains differentiable with respect to
        the current policy parameters, as required by the WPO surrogate.
        """
        base_score = -(pre_tanh_value - self.loc) / self.scale.square()
        log_abs_det_jacobian = torch.log(self._scale) + 2.0 * (
            math.log(2.0)
            - pre_tanh_value
            - nn.functional.softplus(-2.0 * pre_tanh_value)
        )
        return (base_score + 2.0 * torch.tanh(pre_tanh_value)) * torch.exp(
            -log_abs_det_jacobian
        )

    def rsample_with_pre_tanh(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> tuple[Tensor, Tensor]:
        """Return a reparameterized action together with its exact latent sample."""
        pre_tanh_value = self.base_dist.rsample(sample_shape)
        value = self.action_from_pre_tanh(pre_tanh_value)
        self._cached_action = value
        self._cached_pre_tanh = pre_tanh_value
        return value, pre_tanh_value

    def sample_with_pre_tanh(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> tuple[Tensor, Tensor]:
        """Return a non-reparameterized action and its exact latent sample."""
        pre_tanh_value = self.base_dist.sample(sample_shape)
        value = self.action_from_pre_tanh(pre_tanh_value)
        self._cached_action = value
        self._cached_pre_tanh = pre_tanh_value
        return value, pre_tanh_value

    def rsample_and_log_prob(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> tuple[Tensor, Tensor]:
        """Sample an action and evaluate its density through one consistent path."""
        value, pre_tanh_value = self.rsample_with_pre_tanh(sample_shape)
        return value, self.log_prob_from_pre_tanh(pre_tanh_value)

    def sample_and_log_prob(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> tuple[Tensor, Tensor]:
        """Sample without gradients and evaluate density through the same latent."""
        value, pre_tanh_value = self.sample_with_pre_tanh(sample_shape)
        return value, self.log_prob_from_pre_tanh(pre_tanh_value)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Reparameterized sample from the distribution."""
        value, _ = self.rsample_with_pre_tanh(sample_shape)
        return value

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Sample from the distribution (not differentiable)."""
        value, _ = self.sample_with_pre_tanh(sample_shape)
        return value

    def entropy(self) -> Tensor:
        """Approximate entropy of the distribution.

        Note: The entropy of a tanh-squashed Normal does not have a closed form.
        This returns the entropy of the base Normal, which is an upper bound.
        For exact entropy, use Monte Carlo estimation.
        """
        return self.base_dist.entropy() - torch.log(self._scale)

    def expand(self, batch_shape: torch.Size, _instance=None) -> "TanhNormal":
        """Expand the distribution to a new batch shape."""
        new = self._get_checked_instance(TanhNormal, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new._mean_offset = self._mean_offset.expand(batch_shape)
        new._scale = self._scale.expand(batch_shape)
        new._cached_action = None
        new._cached_pre_tanh = None
        base_dist = Normal(new.loc, new.scale, validate_args=False)
        super(TanhNormal, new).__init__(base_dist, TanhTransform(cache_size=1), validate_args=False)
        new._validate_args = self._validate_args
        return new


def log_prob_from_tanh_normal(
    value: Tensor,
    loc: Tensor,
    scale: Tensor,
    pre_tanh_value: Tensor | None = None,
) -> Tensor:
    """Functional interface for computing log probability of tanh-squashed Normal.

    Args:
        value: Squashed samples in (-1, 1).
        loc: Mean of the underlying Normal.
        scale: Std of the underlying Normal.
        pre_tanh_value: Optional pre-tanh values to avoid atanh.

    Returns:
        Log probability tensor.
    """
    if pre_tanh_value is None:
        eps = torch.finfo(value.dtype).eps
        value = value.clamp(-1.0 + eps, 1.0 - eps)
        pre_tanh_value = torch.atanh(value)

    # Normal log prob
    var = scale**2
    log_scale = torch.log(scale)
    log_prob = -0.5 * (((pre_tanh_value - loc) ** 2) / var + 2 * log_scale + math.log(2 * math.pi))

    # Stable log det Jacobian
    log_det_jacobian = 2.0 * (math.log(2.0) - pre_tanh_value - nn.functional.softplus(-2.0 * pre_tanh_value))

    return log_prob - log_det_jacobian
