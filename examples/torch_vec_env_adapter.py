from __future__ import annotations

import torch
from collections.abc import Mapping
from tensordict import TensorDict, TensorDictBase
from typing import Any


def clone_tensor_tree(value: Any) -> Any:
    """Clone tensor leaves while preserving a nested observation/info structure."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (TensorDictBase, Mapping)):
        return {key: clone_tensor_tree(child) for key, child in value.items()}
    return value


def _select_observation(tree: Any, key: str | None) -> Any | None:
    if key is None or not isinstance(tree, (TensorDictBase, Mapping)):
        return None
    current = tree
    for component in key.split("."):
        if not isinstance(current, (TensorDictBase, Mapping)) or component not in current:
            return None
        current = current[component]
    return current


def _flatten_observation(value: Any, *, num_envs: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if tensor.ndim == 0:
            raise ValueError("Observation tensors must have a leading environment dimension.")
        if int(tensor.shape[0]) != num_envs:
            raise ValueError(
                "Observation tensor batch does not match the vector environment: "
                f"expected {num_envs}, got shape {tuple(tensor.shape)}."
            )
        return tensor.to(device=device, dtype=torch.float32).reshape(num_envs, -1)

    if isinstance(value, (TensorDictBase, Mapping)):
        flattened = [
            _flatten_observation(value[key], num_envs=num_envs, device=device)
            for key in sorted(value)
        ]
        if not flattened:
            raise ValueError("Observation mappings cannot be empty.")
        return torch.cat(flattened, dim=-1)

    try:
        tensor = torch.as_tensor(value, device=device)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Unsupported observation leaf type: {type(value).__name__}.") from error
    return _flatten_observation(tensor, num_envs=num_envs, device=device)


def _maybe_flatten_observation(
    value: Any,
    *,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor | None:
    try:
        return _flatten_observation(value, num_envs=num_envs, device=device)
    except (TypeError, ValueError):
        return None


def observations_to_tensordict(
    observations: Any,
    *,
    num_envs: int,
    device: torch.device | str,
    policy_key: str | None = None,
    critic_key: str | None = None,
) -> TensorDict:
    """Map native Torch/Gym observation trees to the RSL-RL observation groups."""
    torch_device = torch.device(device)
    mapped: dict[str, torch.Tensor] = {}

    if isinstance(observations, (TensorDictBase, Mapping)):
        for key, value in observations.items():
            flattened = _maybe_flatten_observation(value, num_envs=num_envs, device=torch_device)
            # A top-level metadata object is not an observation group. If
            # explicitly selected below, its conversion will still fail with
            # a useful error instead of being silently ignored.
            if flattened is not None:
                mapped[str(key)] = flattened

    policy_source = _select_observation(observations, policy_key)
    if policy_source is None and isinstance(observations, (TensorDictBase, Mapping)):
        for candidate in ("policy", "state", "observation", "obs"):
            if candidate in observations:
                policy_source = observations[candidate]
                break
    if policy_source is None:
        policy_source = observations
    policy = _flatten_observation(policy_source, num_envs=num_envs, device=torch_device)

    critic_source = _select_observation(observations, critic_key)
    if critic_source is None and isinstance(observations, (TensorDictBase, Mapping)):
        for candidate in ("critic", "privileged_state", "privileged_obs"):
            if candidate in observations:
                critic_source = observations[candidate]
                break
    critic = (
        policy
        if critic_source is None
        else _flatten_observation(critic_source, num_envs=num_envs, device=torch_device)
    )

    mapped.update(policy=policy, critic=critic)
    return TensorDict(mapped, batch_size=[num_envs], device=torch_device)


def action_dimension(env: Any, num_envs: int) -> int:
    """Resolve the flat continuous action dimension of a batched Gym environment."""
    base_env = getattr(env, "unwrapped", env)
    action_manager = getattr(base_env, "action_manager", None)
    if action_manager is not None and hasattr(action_manager, "total_action_dim"):
        return int(action_manager.total_action_dim)

    space = getattr(base_env, "single_action_space", None)
    if space is None:
        space = getattr(env, "single_action_space", None)
    if space is not None and getattr(space, "shape", None) is not None:
        shape = tuple(int(size) for size in space.shape)
    else:
        space = getattr(env, "action_space", None)
        shape = tuple(int(size) for size in getattr(space, "shape", ()))
        if shape and shape[0] == num_envs:
            shape = shape[1:]
    if not shape:
        raise ValueError("The adapter requires a flat, continuous action space with a shape.")

    dimension = 1
    for size in shape:
        dimension *= size
    return dimension


def episode_metrics(info: Mapping[str, Any], dones: torch.Tensor) -> dict[str, Any] | None:
    """Extract metrics for completed environments from Gym-style info dictionaries."""
    if not bool(dones.any().item()):
        return None
    source: Any = info.get("episode")
    final_info = info.get("final_info")
    if isinstance(final_info, Mapping) and isinstance(final_info.get("episode"), Mapping):
        source = final_info["episode"]
    if not isinstance(source, Mapping):
        return None

    metrics: dict[str, Any] = {}
    for key, value in source.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == dones.shape[0]:
            metrics[str(key)] = value[dones]
        else:
            metrics[str(key)] = value
    return metrics


class TorchVectorActionMixin:
    """Shared observation and action handling for native Torch simulators."""

    action_transform: str
    clip_actions: float | None
    cfg: dict

    def _map_observations(self, observations: Any) -> TensorDict:
        return observations_to_tensordict(
            observations,
            num_envs=self.num_envs,
            device=self.device,
            policy_key=self.policy_obs_key,
            critic_key=self.critic_obs_key,
        )

    def set_action_transform(self, action_transform: str) -> None:
        normalized = str(action_transform).lower()
        if normalized not in {"none", "tanh"}:
            raise ValueError("Action transform must be either 'none' or 'tanh'.")
        self.action_transform = normalized
        self.cfg["action_transform"] = normalized

    def transform_actions(self, actions: torch.Tensor) -> torch.Tensor:
        transformed = torch.tanh(actions) if self.action_transform == "tanh" else actions
        if self.clip_actions is not None:
            transformed = transformed.clamp(-self.clip_actions, self.clip_actions)
        return transformed
