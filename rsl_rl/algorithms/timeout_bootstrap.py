from __future__ import annotations

import torch
from tensordict import TensorDict


def validate_time_outs(dones: torch.Tensor, time_outs: torch.Tensor) -> None:
    if torch.any(time_outs & ~dones):
        raise ValueError("extras['time_outs'] must be true only where dones is true.")


def terminal_observations(
    time_outs: torch.Tensor,
    extras: dict[str, torch.Tensor | TensorDict],
    device: str,
) -> TensorDict:
    if torch.any(time_outs) and "terminal_observations" not in extras:
        raise ValueError("extras['terminal_observations'] is required whenever extras['time_outs'] contains True.")
    return extras["terminal_observations"].to(device)


def clone_hidden_state(hidden_state):
    if hidden_state is None:
        return None
    if isinstance(hidden_state, tuple):
        return tuple(tensor.clone() for tensor in hidden_state)
    return hidden_state.clone()


def clone_policy_hidden_states(policy):
    if not policy.is_recurrent:
        return None
    hidden_states = policy.get_hidden_states()
    return (
        clone_hidden_state(hidden_states[0]),
        clone_hidden_state(hidden_states[1]),
    )


def restore_policy_hidden_states(policy, hidden_states) -> None:
    if hidden_states is None:
        return
    policy.memory_a.reset(hidden_state=hidden_states[0])
    policy.memory_c.reset(hidden_state=hidden_states[1])
