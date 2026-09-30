# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any
from abc import ABC, abstractmethod
from tensordict import TensorDict


@dataclass(frozen=True)
class TransitionQueryResult:
    """One-step transition query result returned by non-mutating env queries."""

    observations: TensorDict
    rewards: torch.Tensor
    dones: torch.Tensor
    extras: dict


class VecEnv(ABC):
    """Abstract class for a vectorized environment.

    The vectorized environment is a collection of environments that are synchronized. This means that the same type of
    action is applied to all environments and the same type of observation is returned from all environments.
    """

    num_envs: int
    """Number of environments."""

    num_actions: int
    """Number of actions."""

    max_episode_length: int | torch.Tensor
    """Maximum episode length.

    The maximum episode length can be a scalar or a tensor. If it is a scalar, it is the same for all environments.
    If it is a tensor, it is the maximum episode length for each environment. This is useful for dynamic episode
    lengths.
    """

    episode_length_buf: torch.Tensor
    """Buffer for current episode lengths."""

    device: torch.device | str
    """Device to use."""

    cfg: dict | object
    """Configuration object."""

    @abstractmethod
    def get_observations(self) -> TensorDict:
        """Return the current observations.

        Returns:
            The observations from the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Apply input action to the environment.

        Args:
            actions: Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            observations: Observations from the environment.
            rewards: Rewards from the environment. Shape: (num_envs,)
            dones: Done flags from the environment. Shape: (num_envs,)
            extras: Extra information from the environment.

        Observations:
            The observations TensorDict usually contains multiple observation groups. The `obs_groups`
            dictionary of the runner configuration specifies which observation groups are used for which
            purpose, i.e., it maps the available observation groups to observation sets. The observation sets
            (keys of the `obs_groups` dictionary) currently used by rsl_rl are:

            - "policy": Specified observation groups are used as input to the actor/student network.
            - "critic": Specified observation groups are used as input to the critic network.
            - "teacher": Specified observation groups are used as input to the teacher network.
            - "rnd_state": Specified observation groups are used as input to the RND network.

            Incomplete or incorrect configurations are handled in the `resolve_obs_groups()` function in
            `rsl_rl/utils/utils.py`.

        Extras:
            The extras dictionary includes metrics such as the episode reward, episode length, etc. The following
            dictionary keys are used by rsl_rl:

            - "time_outs" (torch.Tensor): Timeout flags for the environments. These correspond to done transitions
               caused by time limits and must be a subset of `dones`.

            - "terminal_observations" (TensorDict): Full-batch post-step observations before reset. This key is
               required whenever any value in "time_outs" is true. For timed-out environments these observations are
               used for exact value bootstrapping; for other environments they may match the regular next
               observations.

            - "log" (dict[str, float | torch.Tensor]): Additional information for logging and debugging purposes.
               The key should be a string and start with "/" for namespacing. The value can be a scalar or a
               tensor. If it is a tensor, the mean of the tensor is used for logging.
        """
        raise NotImplementedError

    def supports_transition_queries(self) -> bool:
        """Return whether this environment can query one-step transitions without mutating live state."""
        return False

    def capture_transition_query_state(self) -> Any:
        """Capture an opaque state snapshot for a later non-mutating transition query."""
        raise NotImplementedError(f"{type(self).__name__} does not support transition queries.")

    def offload_transition_query_state(self, query_state: Any) -> Any:
        """Move a retained snapshot out of accelerator memory when supported.

        Environments with host-resident or lightweight snapshots need no special
        handling. Accelerator simulators can override this and the matching
        preparation hook without changing their live environment state.
        """
        return query_state

    def prepare_transition_query_state(self, query_state: Any) -> Any:
        """Materialize one snapshot for a group of non-mutating queries.

        Callers retain the returned state only for that group, then release it.
        This must not restore or otherwise change the live environment.
        """
        return query_state

    def query_transitions(
        self,
        query_state: Any,
        env_ids: torch.Tensor,
        actions: torch.Tensor,
    ) -> TransitionQueryResult:
        """Query one-step transitions from a captured state without changing the live environment."""
        raise NotImplementedError(f"{type(self).__name__} does not support transition queries.")

    def query_transition_candidates(self, query_state: Any, actions: torch.Tensor) -> TransitionQueryResult:
        """Query [K,N,A] candidates; results have leading batch shape [K,N].

        The default preserves the per-query API. GPU simulators can override
        this to avoid repeated indexing and compile the independent branches.
        """
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (self.num_envs, self.num_actions) or actions.shape[0] < 1:
            raise ValueError("Candidate actions must have shape [K, num_envs, num_actions], K >= 1.")
        ids = torch.arange(self.num_envs, device=self.device)
        results = [self.query_transitions(query_state, ids, action) for action in actions]
        extras = {"time_outs": torch.stack([r.extras["time_outs"] for r in results])}
        terminal_obs = []
        for result in results:
            if "terminal_observations" in result.extras:
                terminal_obs.append(result.extras["terminal_observations"])
            elif torch.any(result.extras["time_outs"]):
                raise ValueError("extras['terminal_observations'] is required for timed-out candidate queries.")
            else:
                terminal_obs.append(result.observations)
        extras["terminal_observations"] = torch.stack(terminal_obs)
        return TransitionQueryResult(torch.stack([r.observations for r in results]),
                                     torch.stack([r.rewards for r in results]),
                                     torch.stack([r.dones for r in results]), extras)
