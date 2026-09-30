from __future__ import annotations

import importlib
import torch
from collections.abc import Mapping
from tensordict import TensorDict
from typing import Any

from examples.torch_vec_env_adapter import (
    TorchVectorActionMixin,
    action_dimension,
    episode_metrics,
)
from rsl_rl.env import VecEnv


class ManiSkill3VecEnv(TorchVectorActionMixin, VecEnv):
    """RSL-RL adapter for ManiSkill 3's native GPU vector environments."""

    def __init__(self, env: Any, env_cfg: dict, seed: int, device: str | torch.device) -> None:
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        self.num_envs = int(env.num_envs)
        native_device = getattr(env, "device", getattr(self.unwrapped, "device", device))
        self.device = torch.device(native_device)
        self.num_actions = action_dimension(env, self.num_envs)
        self.max_episode_length = _max_episode_length(env, env_cfg)
        self.auto_reset = bool(env_cfg.get("auto_reset", getattr(env, "auto_reset", True)))
        self.action_transform = str(env_cfg.get("action_transform", "none")).lower()
        self.clip_actions = _optional_float(env_cfg.get("clip_actions", 1.0))
        self.policy_obs_key = _optional_string(env_cfg.get("policy_obs_key"))
        self.critic_obs_key = _optional_string(env_cfg.get("critic_obs_key"))
        self.bootstrap_terminations = bool(env_cfg.get("bootstrap_terminations", False))


        self.cfg = {
            **dict(env_cfg),
            "kind": "maniskill3",
            "task_id": str(env_cfg.get("task_id") or "PickCube-v1"),
            "num_envs": self.num_envs,
            "max_episode_length": self.max_episode_length,
            "seed": int(seed),
        }
        self.set_action_transform(self.action_transform)
        self._fallback_episode_length_buf = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._liftpeg_diagnostics_enabled = (
            self.cfg["task_id"] == "LiftPegUpright-v1"
            and bool(self.cfg.get("log_task_diagnostics", False))
        )
        self._init_liftpeg_diagnostic_buffers()
        self._observations, _ = self.env.reset(seed=int(seed))

    @property
    def episode_length_buf(self) -> torch.Tensor:
        for name in ("elapsed_steps", "_elapsed_steps"):
            value = getattr(self.unwrapped, name, None)
            if isinstance(value, torch.Tensor):
                return value
        return self._fallback_episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.episode_length_buf.copy_(value.to(device=self.device, dtype=torch.long))

    def get_observations(self) -> TensorDict:
        return self._map_observations(self._observations)

    def reset(self, seed: int | None = None) -> TensorDict:
        self._observations, _ = self.env.reset(seed=seed)
        self._reset_liftpeg_diagnostic_buffers()
        return self.get_observations()

    def _init_liftpeg_diagnostic_buffers(self) -> None:
        if not self._liftpeg_diagnostics_enabled:
            self._liftpeg_diag_sums = {}
            self._liftpeg_diag_mins = {}
            self._liftpeg_diag_true_counts = {}
            self._liftpeg_diag_once = {}
            self._liftpeg_diag_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._liftpeg_first_success_step = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
            return

        self._liftpeg_diag_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for key in _LIFTPEG_CONTINUOUS_KEYS
        }
        self._liftpeg_diag_mins = {
            key: torch.full(
                (self.num_envs,), float("inf"), dtype=torch.float32, device=self.device
            )
            for key in _LIFTPEG_MIN_KEYS
        }
        self._liftpeg_diag_true_counts = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for key in _LIFTPEG_BOOLEAN_KEYS
        }
        self._liftpeg_diag_once = {
            key: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for key in _LIFTPEG_BOOLEAN_KEYS
        }
        self._liftpeg_diag_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._liftpeg_first_success_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

    def _reset_liftpeg_diagnostic_buffers(self, mask: torch.Tensor | None = None) -> None:
        if not self._liftpeg_diagnostics_enabled:
            return
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for value in self._liftpeg_diag_sums.values():
            value[mask] = 0.0
        for value in self._liftpeg_diag_mins.values():
            value[mask] = float("inf")
        for value in self._liftpeg_diag_true_counts.values():
            value[mask] = 0.0
        for value in self._liftpeg_diag_once.values():
            value[mask] = False
        self._liftpeg_diag_steps[mask] = 0
        self._liftpeg_first_success_step[mask] = -1

    def _update_liftpeg_diagnostics(
        self,
        info: Mapping[str, Any],
        dones: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        if not self._liftpeg_diagnostics_enabled:
            return None

        diagnostics = _extract_liftpeg_step_diagnostics(
            info,
            num_envs=self.num_envs,
            device=self.device,
        )
        if not diagnostics:
            return None

        self._liftpeg_diag_steps += 1

        for key in _LIFTPEG_CONTINUOUS_KEYS:
            value = diagnostics[key].float()
            self._liftpeg_diag_sums[key] += value
            if key in self._liftpeg_diag_mins:
                self._liftpeg_diag_mins[key] = torch.minimum(
                    self._liftpeg_diag_mins[key], value
                )

        for key in _LIFTPEG_BOOLEAN_KEYS:
            value = diagnostics[key].bool()
            self._liftpeg_diag_true_counts[key] += value.float()
            self._liftpeg_diag_once[key] |= value

        success_now = diagnostics["liftpeg_success"].bool()
        first_success = success_now & (self._liftpeg_first_success_step < 0)
        self._liftpeg_first_success_step[first_success] = self._liftpeg_diag_steps[
            first_success
        ]

        if not bool(dones.any().item()):
            return None

        counts = self._liftpeg_diag_steps[dones].clamp_min(1).float()
        completed: dict[str, torch.Tensor] = {}

        for key in _LIFTPEG_CONTINUOUS_KEYS:
            completed[f"{key}_mean"] = self._liftpeg_diag_sums[key][dones] / counts

        for key in _LIFTPEG_MIN_KEYS:
            completed[f"{key}_min"] = self._liftpeg_diag_mins[key][dones]

        for key in _LIFTPEG_BOOLEAN_KEYS:
            completed[f"{key}_fraction"] = (
                self._liftpeg_diag_true_counts[key][dones] / counts
            )
            completed[f"{key}_once"] = self._liftpeg_diag_once[key][dones].float()

        completed["liftpeg_first_success_step"] = self._liftpeg_first_success_step[
            dones
        ].float()

        self._reset_liftpeg_diagnostic_buffers(dones)
        return completed


    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions = self.transform_actions(
            actions.to(device=self.device, dtype=torch.float32).reshape(self.num_envs, self.num_actions)
        )
        observations, rewards, terminated, truncated, info = self.env.step(actions)
        if not isinstance(info, Mapping):
            raise TypeError("ManiSkill vector environments must return an info mapping.")

        terminated = torch.as_tensor(
            terminated,
            device=self.device,
            dtype=torch.bool,
        ).reshape(self.num_envs)

        truncated = torch.as_tensor(
            truncated,
            device=self.device,
            dtype=torch.bool,
        ).reshape(self.num_envs)

        # The physical episode ends on either success termination
        # or time-limit truncation.
        dones = terminated | truncated

        # REPPO-style ManiSkill semantics:
        # success termination still resets the environment, but for
        # critic/return computation it is treated like a truncation
        # so that we bootstrap from the final observation.
        if self.bootstrap_terminations:
            time_outs = terminated | truncated
        else:
            # A genuine task termination takes precedence when success and the
            # time limit occur on the same environment step.
            time_outs = truncated & ~terminated

        terminal_raw = info.get("final_observation")
        if terminal_raw is None:
            if self.auto_reset and bool(dones.any().item()):
                raise RuntimeError(
                    "ManiSkill auto-reset did not provide info['final_observation']; exact timeout "
                    "bootstrapping requires ManiSkillVectorEnv(auto_reset=True)."
                )
            terminal_raw = observations

        self._observations = observations
        extras: dict[str, Any] = {
            "time_outs": time_outs,
            "terminal_observations": self._map_observations(terminal_raw),
        }
        metrics = episode_metrics(info, dones)
        liftpeg_metrics = self._update_liftpeg_diagnostics(info, dones)
        if liftpeg_metrics:
            if metrics is None:
                metrics = {}
            metrics.update(liftpeg_metrics)
        if metrics:
            extras["episode"] = metrics
        elif isinstance(info.get("log"), Mapping):
            extras["log"] = dict(info["log"])

        return (
            self._map_observations(observations),
            torch.as_tensor(rewards, device=self.device, dtype=torch.float32).reshape(self.num_envs),
            dones,
            extras,
        )

    def close(self) -> None:
        self.env.close()


def make_env(env_cfg: dict, seed: int, device: str) -> ManiSkill3VecEnv:
    """Create a state-based ManiSkill 3 benchmark environment."""
    cfg = dict(env_cfg)
    task_id = str(cfg.get("task_id") or "PickCube-v1")
    num_envs = int(cfg.get("num_envs", 256))
    torch_device = torch.device(device)
    if torch_device.type == "cpu" and num_envs != 1:
        raise ValueError("ManiSkill's CPU simulator supports num_envs=1; use a CUDA device for vectorized training.")
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ManiSkill3 adapter requested CUDA, but torch.cuda.is_available() is False.")

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 -- registers ManiSkill tasks
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ManiSkill3 dependencies are missing. Install them with "
            "`python -m pip install 'mani_skill>=3,<4'`."
        ) from error

    for module_name in cfg.get("task_modules", ()):
        importlib.import_module(str(module_name))

    env_kwargs = dict(cfg.get("env_kwargs") or {})
    for key in (
        "obs_mode",
        "control_mode",
        "reward_mode",
        "render_mode",
        "render_backend",
        "sim_backend",
        "reconfiguration_freq",
        "robot_uids",
        "sim_config",
    ):
        if key == "reconfiguration_freq":
            # REPPO uses None for training and 1 for the separate eval env.
            env_kwargs[key] = cfg.get(key, None)
        elif key in cfg and cfg[key] is not None:
            env_kwargs[key] = cfg[key]
    env_kwargs.setdefault("obs_mode", "state")
    env_kwargs.setdefault("render_backend", "none")
    env_kwargs.setdefault("sim_backend", _maniskill_sim_backend(torch_device))
    if cfg.get("max_episode_length") is not None:
        env_kwargs.setdefault("max_episode_steps", int(cfg["max_episode_length"]))

    raw_env = gym.make(task_id, num_envs=num_envs, **env_kwargs)

    # Resolve the native task horizon before custom wrappers are added.
    if cfg.get("max_episode_length") is None:
        spec = getattr(raw_env, "spec", None)
        native_horizon = getattr(spec, "max_episode_steps", None)
        if native_horizon is None:
            native_horizon = getattr(raw_env, "_max_episode_steps", None)
        if native_horizon is not None:
            cfg["max_episode_length"] = int(native_horizon)

    if task_id == "LiftPegUpright-v1" and bool(cfg.get("log_task_diagnostics", False)):
        class LiftPegDiagnosticsWrapper(gym.Wrapper):
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                info = dict(info)
                info.update(_liftpeg_step_diagnostics(self.unwrapped))
                return obs, reward, terminated, truncated, info

        raw_env = LiftPegDiagnosticsWrapper(raw_env)

    vector_env = ManiSkillVectorEnv(
        raw_env,
        auto_reset=bool(cfg.get("auto_reset", True)),
        ignore_terminations=bool(cfg.get("ignore_terminations")),
        record_metrics=bool(cfg.get("record_metrics", True)),
    )
    return ManiSkill3VecEnv(vector_env, cfg, seed, torch_device)



_LIFTPEG_CONTINUOUS_KEYS = (
    "liftpeg_rotation_reward_raw",
    "liftpeg_height_reward_raw",
    "liftpeg_reaching_reward_raw",
    "liftpeg_rotation_reward",
    "liftpeg_height_reward",
    "liftpeg_reaching_reward",
    "liftpeg_orientation_error_rad",
    "liftpeg_height_error_m",
    "liftpeg_reach_distance_m",
)

_LIFTPEG_MIN_KEYS = (
    "liftpeg_orientation_error_rad",
    "liftpeg_height_error_m",
)

_LIFTPEG_BOOLEAN_KEYS = (
    "liftpeg_orientation_ok",
    "liftpeg_height_ok",
    "liftpeg_grasping",
    "liftpeg_success",
)


def _liftpeg_step_diagnostics(env: Any) -> dict[str, torch.Tensor]:
    from mani_skill.utils.geometry import rotation_conversions

    qmat = rotation_conversions.quaternion_to_matrix(env.peg.pose.q)
    euler = rotation_conversions.matrix_to_euler_angles(qmat, "XYZ")

    orientation_error = torch.abs(torch.abs(euler[:, 2]) - torch.pi / 2)
    height_error = torch.abs(env.peg.pose.p[:, 2] - env.peg_half_length)
    orientation_ok = orientation_error < 0.08
    height_ok = height_error < 0.005
    success = orientation_ok & height_ok

    peg_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=qmat.dtype)
    rotation_reward_raw = torch.abs((qmat @ peg_axis)[:, 2])
    height_reward_raw = 1.0 - torch.tanh(5.0 * height_error)

    reach_distance = torch.linalg.norm(env.peg.pose.p - env.agent.tcp.pose.p, dim=1)
    grasping = env.agent.is_grasping(env.peg)
    reaching_reward_raw = 1.0 - torch.tanh(5.0 * reach_distance)
    reaching_reward_raw = torch.where(
        grasping,
        torch.ones_like(reaching_reward_raw),
        reaching_reward_raw,
    ) / 5.0

    return {
        "liftpeg_rotation_reward_raw": rotation_reward_raw,
        "liftpeg_height_reward_raw": height_reward_raw,
        "liftpeg_reaching_reward_raw": reaching_reward_raw,
        "liftpeg_rotation_reward": rotation_reward_raw / 3.0,
        "liftpeg_height_reward": height_reward_raw / 3.0,
        "liftpeg_reaching_reward": reaching_reward_raw / 3.0,
        "liftpeg_orientation_error_rad": orientation_error,
        "liftpeg_height_error_m": height_error,
        "liftpeg_reach_distance_m": reach_distance,
        "liftpeg_orientation_ok": orientation_ok,
        "liftpeg_height_ok": height_ok,
        "liftpeg_grasping": grasping,
        "liftpeg_success": success,
    }


def _extract_liftpeg_step_diagnostics(
    info: Mapping[str, Any],
    *,
    num_envs: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    # ManiSkillVectorEnv moves pre-reset info into final_info when auto-reset
    # occurs. final_info is a clone of the complete pre-reset info dictionary,
    # so it is the correct source for that step whenever present.
    source: Mapping[str, Any] = info
    final_info = info.get("final_info")
    if isinstance(final_info, Mapping) and any(
        key in final_info for key in _LIFTPEG_CONTINUOUS_KEYS
    ):
        source = final_info

    diagnostics: dict[str, torch.Tensor] = {}
    for key in (*_LIFTPEG_CONTINUOUS_KEYS, *_LIFTPEG_BOOLEAN_KEYS):
        value = source.get(key)
        if value is None:
            return {}
        diagnostics[key] = torch.as_tensor(value, device=device).reshape(num_envs)
    return diagnostics

def _maniskill_sim_backend(device: torch.device) -> str:
    if device.type == "cpu":
        return "physx_cpu"
    index = 0 if device.index is None else int(device.index)
    return f"physx_cuda:{index}"


def _max_episode_length(env: Any, env_cfg: dict) -> int:
    if env_cfg.get("max_episode_length") is not None:
        return int(env_cfg["max_episode_length"])
    base_env = getattr(env, "unwrapped", env)
    wrapped_env = getattr(env, "_env", None)
    for source, name in (
        (wrapped_env, "max_episode_steps"),
        (wrapped_env, "_max_episode_steps"),
        (base_env, "max_episode_steps"),
        (base_env, "_max_episode_steps"),
        (getattr(wrapped_env, "spec", None), "max_episode_steps"),
        (getattr(env, "spec", None), "max_episode_steps"),
    ):
        value = getattr(source, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not determine the ManiSkill episode length; set runner.env.max_episode_length.")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None or str(value).strip() == "" else str(value)


__all__ = ["ManiSkill3VecEnv", "make_env"]
