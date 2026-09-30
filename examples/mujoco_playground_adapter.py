from __future__ import annotations

import contextlib
import functools
import importlib.metadata
import importlib.util
import json
import torch
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from tensordict import TensorDict
from typing import Any

from rsl_rl.env import TransitionQueryResult, VecEnv
from examples.query_snapshot import PackedQuerySnapshot, QuerySnapshotCodec

TERMINAL_OBS_INFO_KEY = "rsl_rl_terminal_observations"


@dataclass(frozen=True)
class MuJoCoPlaygroundQueryState:
    env_state: Any
    episode_length_buf: torch.Tensor


class TerminalObservationAutoResetWrapper:
    """Playground auto-reset wrapper that preserves pre-reset terminal observations."""

    def __init__(self, env: Any, jax_module: Any, jp_module: Any, full_reset: bool = False) -> None:
        self.env = env
        self._jax = jax_module
        self._jp = jp_module
        self._full_reset = full_reset
        self._info_key = "RslRlAutoResetWrapper"

    def reset(self, rng: Any) -> Any:
        rng_key = self._jax.vmap(self._jax.random.split)(rng)
        rng, key = rng_key[..., 0], rng_key[..., 1]
        state = self.env.reset(key)
        state.info[f"{self._info_key}_first_data"] = state.data
        state.info[f"{self._info_key}_first_obs"] = state.obs
        state.info[f"{self._info_key}_rng"] = rng
        state.info[f"{self._info_key}_done_count"] = self._jp.zeros(key.shape[:-1], dtype=int)
        state.info[TERMINAL_OBS_INFO_KEY] = state.obs
        return state

    def step(self, state: Any, action: Any) -> Any:
        reset_state = None
        rng_key = self._jax.vmap(self._jax.random.split)(state.info[f"{self._info_key}_rng"])
        reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]
        if self._full_reset:
            reset_state = self.reset(reset_key)
            reset_data = reset_state.data
            reset_obs = reset_state.obs
        else:
            reset_data = state.info[f"{self._info_key}_first_data"]
            reset_obs = state.info[f"{self._info_key}_first_obs"]

        if "steps" in state.info:
            steps = state.info["steps"]
            steps = self._jp.where(state.done, self._jp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        state = state.replace(done=self._jp.zeros_like(state.done))
        state = self.env.step(state, action)
        terminal_obs = state.obs

        def where_done(x: Any, y: Any) -> Any:
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = self._jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return self._jp.where(done, x, y)

        data = self._jax.tree.map(where_done, reset_data, state.data)
        obs = self._jax.tree.map(where_done, reset_obs, state.obs)

        next_info = state.info
        done_count_key = f"{self._info_key}_done_count"
        if self._full_reset and reset_state is not None:
            next_info = self._jax.tree.map(where_done, reset_state.info, state.info)
            next_info[done_count_key] = state.info[done_count_key]

            if "steps" in next_info:
                next_info["steps"] = state.info["steps"]
            preserve_info_key = f"{self._info_key}_preserve_info"
            if preserve_info_key in next_info:
                next_info[preserve_info_key] = state.info[preserve_info_key]

        next_info[done_count_key] += state.done.astype(int)
        next_info[f"{self._info_key}_rng"] = reset_rng
        next_info[TERMINAL_OBS_INFO_KEY] = terminal_obs

        return state.replace(data=data, obs=obs, info=next_info)

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


class MuJoCoPlaygroundBackend:
    def __init__(
        self,
        *,
        env_cfg: dict,
        seed: int,
        device: torch.device,
    ) -> None:
        self.impl = str(env_cfg.get("impl") or "warp")
        self.query_snapshot_mode = str(env_cfg.get("query_snapshot_mode", "packed"))
        self.query_candidate_mode = str(env_cfg.get("query_candidate_mode", "compiled"))
        self.query_candidate_batch_size = int(env_cfg.get("query_candidate_batch_size", 1))
        if self.query_candidate_batch_size < 1:
            raise ValueError("query_candidate_batch_size must be positive.")
        if self.query_snapshot_mode not in {"full", "packed"}:
            raise ValueError("query_snapshot_mode must be 'full' or 'packed'.")
        if self.query_candidate_mode not in {"legacy", "full_batch", "compiled"}:
            raise ValueError("query_candidate_mode must be 'legacy', 'full_batch' or 'compiled'.")
        self._query_snapshot_codec = None
        self._query_snapshot_codecs = []
        self._candidate_step_fn = None
        self.warp_graph_mode = str(env_cfg.get("warp_graph_mode") or "warp").lower()
        if self.warp_graph_mode not in {"warp", "none"}:
            raise ValueError("runner.env.warp_graph_mode must be either 'warp' or 'none'.")
        graph_free_warmup = self.impl.lower() == "warp"
        graph_context = (
            _mujoco_warp_reset_without_graph_capture() if graph_free_warmup else contextlib.nullcontext()
        )
        with graph_context:
            # Enter before importing Playground: its import path can create
            # and cache the MuJoCo Warp FFI callable.
            modules = _load_playground_modules(env_cfg)
            self.registry = modules["registry"]
            self.playground_wrapper = modules["wrapper"]
            self.jax = modules["jax"]
            self.jp = modules["jp"]

            self.task_id = str(env_cfg.get("task_id") or "CheetahRun")
            self.num_envs = int(env_cfg.get("num_envs", 4096))
            self.seed = int(seed)
            self.device = device
            self.device_rank = _resolve_device_rank(device, self.jax)

            self.playground_config = self.registry.get_default_config(self.task_id)
            self.playground_config.impl = self.impl
            if env_cfg.get("episode_length") is not None:
                self.playground_config.episode_length = int(env_cfg["episode_length"])
            if env_cfg.get("action_repeat") is not None:
                self.playground_config.action_repeat = int(env_cfg["action_repeat"])

            self.max_episode_length = int(self.playground_config.episode_length)
            self.action_repeat = int(self.playground_config.action_repeat)
            self.clip_actions = env_cfg.get("clip_actions", 1.0)
            self.full_reset = bool(env_cfg.get("full_reset"))
            self.collect_step_metrics = bool(
                env_cfg.get("collect_step_metrics", True)
            )
            repeated_seed_group_size = env_cfg.get("repeated_seed_group_size")
            if repeated_seed_group_size is not None:
                repeated_seed_group_size = int(repeated_seed_group_size)
                if (
                    repeated_seed_group_size <= 0
                    or self.num_envs % repeated_seed_group_size != 0
                ):
                    raise ValueError(
                        "runner.env.repeated_seed_group_size must be positive "
                        "and divide runner.env.num_envs exactly."
                    )
            self.config_overrides = _parse_config_overrides(env_cfg.get("playground_config_overrides"))

            if env_cfg.get("warp_kernel_cache_dir") is not None and modules["warp"] is not None:
                modules["warp"].config.kernel_cache_dir = str(env_cfg["warp_kernel_cache_dir"])

            self.key = self.jax.random.PRNGKey(self.seed)
            if self.device_rank is not None:
                gpu_devices = self.jax.devices("gpu")
                self.key = self.jax.device_put(self.key, gpu_devices[self.device_rank])
            key_reset, key_randomization = self.jax.random.split(self.key)
            self.key_reset = _vectorized_keys(
                self.jax,
                self.jp,
                key_reset,
                self.num_envs,
                repeated_seed_group_size=repeated_seed_group_size,
            )

            raw_env = self.registry.load(
                self.task_id,
                config=self.playground_config,
                config_overrides=self.config_overrides,
            )
            randomizer = self.registry.get_domain_randomizer(self.task_id)
            self.env = _wrap_for_rsl_training(
                self.playground_wrapper,
                self.jax,
                self.jp,
                raw_env,
                episode_length=self.max_episode_length,
                action_repeat=self.action_repeat,
                randomization_fn=_vectorized_randomizer(
                    randomizer,
                    self.jax,
                    self.jp,
                    key_randomization,
                    self.num_envs,
                    repeated_seed_group_size=repeated_seed_group_size,
                )
                if randomizer is not None
                else None,
                full_reset=self.full_reset,
            )
            self.num_actions = int(raw_env.action_size)
            self.reset_fn = self.jax.jit(self.env.reset)
            self.env_state = self.reset_fn(self.key_reset)

            if graph_free_warmup:
                # MuJoCo Warp creates a few model-specific kernels lazily during
                # the first reset and step. Older CUDA drivers cannot load those
                # modules while Warp is already capturing its execution graph.
                # Warm both paths without graph capture, discard the warm step,
                # then retain Warp's normal graph mode for every training step.
                self.jax.block_until_ready(self.env_state)
                warm_step_fn = self.jax.jit(lambda state, action: self.env.step(state, action))
                warm_action = self.jp.zeros((self.num_envs, self.num_actions), dtype=self.jp.float32)
                warm_state = warm_step_fn(self.env_state, warm_action)
                self.jax.block_until_ready(warm_state)
        if graph_free_warmup and self.warp_graph_mode == "warp":
            # The underlying MuJoCo functions are jitted independently of the
            # adapter. Drop their graph-free traces so the real step retraces
            # with MuJoCo Warp's normal graph mode; loaded Warp modules remain.
            self.jax.clear_caches()
        self.step_fn = self.jax.jit(self.env.step)

    def current_observations(self) -> dict[str, torch.Tensor]:
        return _obs_to_torch_dict(self.env_state.obs)

    def centers_of_mass(self) -> torch.Tensor:
        """Return the whole-body center of mass for every vectorized environment."""
        subtree_com = getattr(self.env_state.data, "subtree_com", None)
        if subtree_com is None or len(subtree_com.shape) != 3:
            raise RuntimeError(
                "MuJoCo state does not expose batched subtree center-of-mass data."
            )
        return _jax_to_torch(subtree_com[:, 0, :]).to(
            device=self.device,
            dtype=torch.float32,
        )



    def _state_with_episode_lengths(self, env_state: Any, episode_length_buf: torch.Tensor) -> Any:
        if env_state is None or "steps" not in env_state.info:
            return env_state
        steps = _torch_to_jax(episode_length_buf.to(device=self.device, dtype=torch.float32).contiguous())
        info = dict(env_state.info)
        info["steps"] = steps
        return env_state.replace(info=info)

    def set_episode_lengths(self, episode_length_buf: torch.Tensor) -> None:
        self.env_state = self._state_with_episode_lengths(self.env_state, episode_length_buf)

    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -float(self.clip_actions), float(self.clip_actions))
        actions = actions.reshape(self.num_envs, self.num_actions).to(device=self.device, dtype=torch.float32)
        with self._step_graph_context():
            next_state = self.step_fn(self.env_state, _torch_to_jax(actions.contiguous()))
        if not self.full_reset and self._query_snapshot_codec is not None:
            # The auto-reset wrapper passes these fields through unchanged.
            # Retain one device copy across the live rollout and all snapshots.
            info = dict(next_state.info)
            for key in self._shared_reset_keys():
                info[key] = self.env_state.info[key]
            next_state = next_state.replace(info=info)
        self.env_state = next_state
        return self._state_to_step_result(self.env_state)

    @staticmethod
    def _shared_reset_keys():
        return ("RslRlAutoResetWrapper_first_data", "RslRlAutoResetWrapper_first_obs")

    def _step_graph_context(self) -> contextlib.AbstractContextManager[None]:
        if self.impl.lower() == "warp" and self.warp_graph_mode == "none":
            return _mujoco_warp_reset_without_graph_capture()
        return contextlib.nullcontext()

    def capture_transition_query_state(self, episode_length_buf: torch.Tensor) -> Any:
        state = self._state_with_episode_lengths(self.env_state, episode_length_buf)
        if hasattr(state, "replace") and isinstance(getattr(state, "info", None), dict):
            state = state.replace(info=dict(state.info))
        return state

    def offload_transition_query_state(self, query_state: Any) -> Any:
        if getattr(self, "query_snapshot_mode", "packed") == "packed":
            shared = {}
            if not getattr(self, "full_reset", False) and hasattr(query_state, "info"):
                shared = {key: query_state.info[key] for key in self._shared_reset_keys() if key in query_state.info}
                query_state = query_state.replace(info={key: value for key, value in query_state.info.items() if key not in shared})
            codec = getattr(self, "_query_snapshot_codec", None)
            signature = QuerySnapshotCodec.state_signature(self.jax, query_state)
            if codec is None or codec.signature != signature:
                codecs = getattr(self, "_query_snapshot_codecs", [])
                codec = next((item for item in codecs if item.signature == signature), None)
                if codec is None:
                    codec = QuerySnapshotCodec(self.jax, query_state)
                    codecs.append(codec)
                # Initial reset counters can have a different weak type from
                # subsequent states. Keep both compiled codecs alive for the
                # backend lifetime; never unload one during Warp graph capture.
                self._query_snapshot_codecs = codecs
                self._query_snapshot_codec = codec
            return codec.offload(query_state, shared)
        # Transfer every dynamic leaf, including contacts, warm-start state,
        # auto-reset data and RNG keys. Retaining any of these JAX device arrays
        # for the whole rollout can exhaust GPU memory on the humanoid tasks.
        # Keep JAX's weak scalar types as well as values/dtypes. Converting to
        # NumPy loses that metadata for e.g. G1's step counters and retraces the
        # query. Wait for the host copy so queued transfers cannot retain a
        # rollout's worth of GPU buffers.
        host_state = self.jax.device_put(query_state, self.jax.devices("cpu")[0])
        return self.jax.block_until_ready(host_state)

    def prepare_transition_query_state(self, query_state: Any) -> Any:
        # All original environment slots travel together: the randomized model
        # batch is tied to that order. This does not replace self.env_state.
        if isinstance(query_state, PackedQuerySnapshot):
            return query_state.codec.prepare(query_state, self.key.device)
        return self.jax.device_put(query_state, self.key.device)

    def query_candidate_states(self, query_state: Any, actions: torch.Tensor):
        """Query [K,N,A] actions, keeping each randomized model's original slot.

        lax.map compiles the candidate loop without allocating K copies of the
        full physics state. Only observations/rewards/terminal flags are kept.
        """
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (self.num_envs, self.num_actions) or actions.shape[0] < 1:
            raise ValueError("Candidate actions must have shape [K, num_envs, num_actions].")
        if self.clip_actions is not None:
            actions = actions.clamp(-float(self.clip_actions), float(self.clip_actions))
        actions = actions.to(device=self.device, dtype=torch.float32).contiguous()
        mode = getattr(self, "query_candidate_mode", "compiled")
        if mode == "legacy":
            ids = torch.arange(self.num_envs, device=self.device)
            results = [self.query_transition_state(query_state, ids, action) for action in actions]
            return self._stack_candidate_results(results)

        jax_module, step_fn = self.jax, self.step_fn

        def step_outputs(state, action):
            # Copy pytree containers for wrappers that update info dictionaries
            # during tracing. Array buffers stay shared and are never donated.
            state = jax_module.tree.map(lambda x: x, state)
            state = step_fn(state, action)
            return (state.obs, state.reward, state.done, state.info["truncation"], state.info[TERMINAL_OBS_INFO_KEY])

        with self._step_graph_context():
            if mode == "full_batch":
                results = [step_outputs(query_state, _torch_to_jax(action.contiguous())) for action in actions]
                result = self.jax.tree.map(lambda *xs: self.jp.stack(xs), *results)
            else:
                if self._candidate_step_fn is None:
                    candidate_batch_size = getattr(self, "query_candidate_batch_size", 1)
                    self._candidate_step_fn = jax_module.jit(
                        lambda state, candidates: jax_module.lax.map(
                            lambda action: step_outputs(state, action), candidates,
                            batch_size=candidate_batch_size if candidate_batch_size > 1 else None,
                        )
                    )
                result = self._candidate_step_fn(query_state, _torch_to_jax(actions))
        obs, rewards, dones, time_outs, terminal_obs = result
        return (_obs_to_torch_dict(obs), _jax_to_torch(rewards), _jax_to_torch(dones).bool(),
                {"time_outs": _jax_to_torch(time_outs).bool(), "terminal_observations": _obs_to_torch_dict(terminal_obs), "log": {}})

    @staticmethod
    def _stack_candidate_results(results):
        obs, rewards, dones, extras = zip(*results)
        return ({key: torch.stack([x[key] for x in obs]) for key in obs[0]},
                torch.stack(rewards), torch.stack(dones),
                {"time_outs": torch.stack([x["time_outs"] for x in extras]),
                 "terminal_observations": {key: torch.stack([x["terminal_observations"][key] for x in extras]) for key in extras[0]["terminal_observations"]}, "log": {}})

    def query_transition_state(
        self, query_state: Any, env_ids: torch.Tensor, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -float(self.clip_actions), float(self.clip_actions))
        env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        actions = actions.reshape(env_ids.shape[0], self.num_actions).to(device=self.device, dtype=torch.float32)
        sliced_state = self._slice_env_state(query_state, env_ids)
        with self._step_graph_context():
            next_state = self.step_fn(sliced_state, _torch_to_jax(actions.contiguous()))
        return self._state_to_step_result(next_state)

    def _slice_env_state(self, env_state: Any, env_ids: torch.Tensor) -> Any:
        env_ids_jax = _torch_to_jax(env_ids.to(device=self.device, dtype=torch.int32).contiguous())

        def take_batched(value: Any) -> Any:
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) > 0 and int(shape[0]) == self.num_envs:
                return self.jp.take(value, env_ids_jax, axis=0)
            return value

        return self.jax.tree.map(take_batched, env_state)

    def _state_to_step_result(
        self, env_state: Any
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        obs = _obs_to_torch_dict(env_state.obs)
        rewards = _jax_to_torch(env_state.reward).to(device=self.device, dtype=torch.float32)
        dones = _jax_to_torch(env_state.done).to(device=self.device).bool()
        time_outs = _jax_to_torch(env_state.info["truncation"]).to(device=self.device).bool()
        terminal_obs = _obs_to_torch_dict(env_state.info[TERMINAL_OBS_INFO_KEY])
        log = (
            {
                key: float(
                    _jax_to_torch(value)
                    .to(device=self.device, dtype=torch.float32)
                    .mean()
                    .item()
                )
                for key, value in env_state.metrics.items()
            }
            if self.collect_step_metrics
            else {}
        )
        return obs, rewards, dones, {"time_outs": time_outs, "terminal_observations": terminal_obs, "log": log}


class MuJoCoPlaygroundVecEnv(VecEnv):
    """RSL-RL VecEnv adapter for MuJoCo Playground tasks."""

    def __init__(
        self,
        backend: Any,
        env_cfg: dict,
        seed: int,
        device: str | torch.device,
    ) -> None:
        self.backend = backend
        self.device = torch.device(device)
        self.cfg = {
            **dict(env_cfg),
            "task_id": backend.task_id,
            "impl": backend.impl,
            "num_envs": backend.num_envs,
            "max_episode_length": backend.max_episode_length,
            "action_repeat": backend.action_repeat,
            "seed": int(seed),
        }
        self.num_envs = int(backend.num_envs)
        self.num_actions = int(backend.num_actions)
        self.max_episode_length = int(backend.max_episode_length)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.action_transform = str(env_cfg.get("action_transform", "none")).lower()
        if self.action_transform not in {"none", "tanh"}:
            raise ValueError("runner.env.action_transform must be either 'none' or 'tanh'.")
        self.unwrapped = self

    def set_action_transform(self, action_transform: str) -> None:
        action_transform = str(action_transform).lower()
        if action_transform not in {"none", "tanh"}:
            raise ValueError("Action transform must be either 'none' or 'tanh'.")
        self.action_transform = action_transform
        self.cfg["action_transform"] = action_transform

    def transform_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_transform == "tanh":
            return torch.tanh(actions)
        return actions

    def get_observations(self) -> TensorDict:
        self.backend.set_episode_lengths(self.episode_length_buf)
        return self._map_observations(self.backend.current_observations())

    def centers_of_mass(self) -> torch.Tensor:
        return self.backend.centers_of_mass()



    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self.backend.set_episode_lengths(self.episode_length_buf)
        raw_obs, rewards, dones, extras = self.backend.step(self.transform_actions(actions))
        dones = dones.to(device=self.device).bool()
        time_outs = extras["time_outs"].to(device=self.device).bool()
        terminal_observations = self._map_observations(extras["terminal_observations"])

        self.episode_length_buf += int(self.backend.action_repeat)
        self.episode_length_buf[dones] = 0

        mapped_extras: dict[str, torch.Tensor | TensorDict | dict[str, float]] = {
            "time_outs": time_outs,
            "terminal_observations": terminal_observations,
            "log": extras.get("log", {}),
        }
        return self._map_observations(raw_obs), rewards.to(device=self.device), dones, mapped_extras

    def supports_transition_queries(self) -> bool:
        return hasattr(self.backend, "capture_transition_query_state") and hasattr(
            self.backend, "query_transition_state"
        )

    def capture_transition_query_state(self) -> MuJoCoPlaygroundQueryState:
        return MuJoCoPlaygroundQueryState(
            env_state=self.backend.capture_transition_query_state(self.episode_length_buf),
            episode_length_buf=self.episode_length_buf.detach().clone(),
        )

    def offload_transition_query_state(
        self, query_state: MuJoCoPlaygroundQueryState
    ) -> MuJoCoPlaygroundQueryState:
        return MuJoCoPlaygroundQueryState(
            env_state=self.backend.offload_transition_query_state(query_state.env_state),
            episode_length_buf=query_state.episode_length_buf.detach().to("cpu", copy=True),
        )

    def prepare_transition_query_state(
        self, query_state: MuJoCoPlaygroundQueryState
    ) -> MuJoCoPlaygroundQueryState:
        return MuJoCoPlaygroundQueryState(
            env_state=self.backend.prepare_transition_query_state(query_state.env_state),
            episode_length_buf=query_state.episode_length_buf.to(self.device),
        )

    def restore_transition_query_state(self, query_state: MuJoCoPlaygroundQueryState) -> None:
        """Restore a state captured by :meth:`capture_transition_query_state`.

        This is primarily useful for deterministic evaluation, where several
        policies should see exactly the same vector of initial states without
        rebuilding and recompiling the Playground environment.
        """
        if not isinstance(query_state, MuJoCoPlaygroundQueryState):
            raise TypeError(
                "`query_state` must be a MuJoCoPlaygroundQueryState, got "
                f"{type(query_state).__name__}."
            )
        env_state = query_state.env_state
        if hasattr(env_state, "replace") and isinstance(getattr(env_state, "info", None), dict):
            env_state = env_state.replace(info=dict(env_state.info))
        self.backend.env_state = env_state
        self.episode_length_buf.copy_(
            query_state.episode_length_buf.to(
                device=self.episode_length_buf.device,
                dtype=self.episode_length_buf.dtype,
            )
        )

    def query_transitions(
        self,
        query_state: MuJoCoPlaygroundQueryState,
        env_ids: torch.Tensor,
        actions: torch.Tensor,
    ) -> TransitionQueryResult:
        raw_obs, rewards, dones, extras = self.backend.query_transition_state(
            query_state.env_state,
            env_ids,
            self.transform_actions(actions),
        )
        mapped_extras: dict[str, torch.Tensor | TensorDict | dict[str, float]] = {
            "time_outs": extras["time_outs"].to(device=self.device).bool(),
            "terminal_observations": self._map_observations(extras["terminal_observations"]),
            "log": extras.get("log", {}),
        }
        return TransitionQueryResult(
            observations=self._map_observations(raw_obs),
            rewards=rewards.to(device=self.device),
            dones=dones.to(device=self.device).bool(),
            extras=mapped_extras,
        )

    def query_transition_candidates(self, query_state, actions) -> TransitionQueryResult:
        if not hasattr(self.backend, "query_candidate_states"):
            return super().query_transition_candidates(query_state, actions)
        raw_obs, rewards, dones, extras = self.backend.query_candidate_states(
            query_state.env_state, self.transform_actions(actions)
        )
        count = actions.shape[0]
        def flatten(obs):
            return {key: value.reshape(count * self.num_envs, *value.shape[2:]) for key, value in obs.items()}
        return TransitionQueryResult(
            self._map_observations(flatten(raw_obs)).reshape(count, self.num_envs),
            rewards, dones,
            {"time_outs": extras["time_outs"],
             "terminal_observations": self._map_observations(flatten(extras["terminal_observations"])).reshape(count, self.num_envs)},
        )

    def _map_observations(self, obs: dict[str, torch.Tensor] | TensorDict) -> TensorDict:
        obs_dict = {
            key: value.to(device=self.device, dtype=torch.float32)
            for key, value in dict(obs.items()).items()
        }
        if "policy" in obs_dict:
            policy_obs = obs_dict["policy"]
        elif "state" in obs_dict:
            policy_obs = obs_dict["state"]
        else:
            policy_obs = next(iter(obs_dict.values()))

        if "critic" in obs_dict:
            critic_obs = obs_dict["critic"]
        elif "privileged_state" in obs_dict:
            critic_obs = obs_dict["privileged_state"]
        else:
            critic_obs = policy_obs

        mapped = {**obs_dict, "policy": policy_obs, "critic": critic_obs}
        return TensorDict(mapped, batch_size=[int(policy_obs.shape[0])], device=self.device)


def make_env(env_cfg: dict, seed: int, device: str) -> MuJoCoPlaygroundVecEnv:
    cfg = dict(env_cfg)
    cfg["task_id"] = cfg.get("task_id") or "CheetahRun"
    cfg["impl"] = cfg.get("impl") or "warp"
    cfg["warp_graph_mode"] = str(cfg.get("warp_graph_mode") or "warp").lower()
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MuJoCo Playground adapter requested CUDA, but torch.cuda.is_available() is False.")

    try:
        backend = MuJoCoPlaygroundBackend(env_cfg=cfg, seed=seed, device=torch_device)
    except ModuleNotFoundError as err:
        raise RuntimeError(_dependency_error_message(cfg["impl"])) from err
    except AttributeError as err:
        if str(cfg["impl"]).lower() == "warp":
            raise RuntimeError(_warp_construction_error_message(err)) from err
        raise
    except RuntimeError as err:
        if str(cfg["impl"]).lower() == "warp":
            raise RuntimeError(_warp_construction_error_message(err)) from err
        raise
    return MuJoCoPlaygroundVecEnv(backend=backend, env_cfg=cfg, seed=seed, device=torch_device)


def _wrap_for_rsl_training(
    playground_wrapper: Any,
    jax_module: Any,
    jp_module: Any,
    env: Any,
    *,
    episode_length: int,
    action_repeat: int,
    randomization_fn: Callable | None,
    full_reset: bool,
) -> Any:
    if randomization_fn is None:
        env = playground_wrapper.brax_training.VmapWrapper(env)
    else:
        env = playground_wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = playground_wrapper.brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    return TerminalObservationAutoResetWrapper(env, jax_module, jp_module, full_reset=full_reset)


def _vectorized_keys(
    jax_module: Any,
    jp_module: Any,
    key: Any,
    num_envs: int,
    *,
    repeated_seed_group_size: int | None = None,
) -> Any:
    """Split PRNG keys, optionally repeating one identical group of seeds."""
    group_size = (
        int(num_envs)
        if repeated_seed_group_size is None
        else int(repeated_seed_group_size)
    )
    if group_size <= 0 or int(num_envs) % group_size != 0:
        raise ValueError(
            "repeated_seed_group_size must be positive and divide num_envs."
        )
    grouped_keys = jax_module.random.split(key, group_size)
    return jp_module.tile(grouped_keys, (int(num_envs) // group_size, 1))


def _vectorized_randomizer(
    randomizer: Callable,
    jax_module: Any,
    jp_module: Any,
    key_randomization: Any,
    num_envs: int,
    *,
    repeated_seed_group_size: int | None = None,
) -> Callable:
    randomization_rng = _vectorized_keys(
        jax_module,
        jp_module,
        key_randomization,
        num_envs,
        repeated_seed_group_size=repeated_seed_group_size,
    )
    return functools.partial(randomizer, rng=randomization_rng)


def _load_playground_modules(env_cfg: dict) -> dict[str, Any]:
    impl = str(env_cfg.get("impl") or "warp").lower()
    _require_module("mujoco_playground", "pip install playground")
    if impl == "warp":
        _require_module("mujoco_warp", "pip install -U mujoco-warp")
        _require_module("warp", "pip install -U warp-lang")

    import jax
    from jax import numpy as jp
    from mujoco_playground import registry
    from mujoco_playground._src import wrapper as playground_wrapper

    modules = {"registry": registry, "wrapper": playground_wrapper, "jax": jax, "jp": jp, "warp": None}
    if impl == "warp":
        import warp

        _patch_warp_dtype_mapping(warp)
        _patch_mujoco_warp_graph_mode()
        modules["warp"] = warp
    return modules


def _patch_warp_dtype_mapping(warp_module: Any) -> None:
    """Provide the dtype map where MuJoCo MJX expects it for older Warp wheels."""
    if hasattr(warp_module.types, "warp_type_to_np_dtype"):
        return
    try:
        from warp._src import types as warp_src_types
    except ImportError:
        return
    if hasattr(warp_src_types, "warp_type_to_np_dtype"):
        warp_module.types.warp_type_to_np_dtype = warp_src_types.warp_type_to_np_dtype


def _patch_mujoco_warp_graph_mode() -> None:
    """Bridge MuJoCo 3.9 to Warp 1.14's relocated graph-mode enum."""
    from mujoco.mjx.warp import types as mjx_warp_types

    if hasattr(mjx_warp_types.GraphMode, "WARP"):
        return

    from mujoco.mjx.third_party.warp._src.jax_experimental import ffi as warp_ffi

    mjx_warp_types.GraphMode = warp_ffi.GraphMode


@contextlib.contextmanager
def _mujoco_warp_reset_without_graph_capture() -> Iterator[None]:
    """Disable Warp graph capture while model-specific reset kernels load."""
    from mujoco.mjx.third_party.warp._src.jax_experimental import ffi as warp_ffi
    from mujoco.mjx.warp import ffi as mjx_warp_ffi

    original = mjx_warp_ffi.jax_callable_variadic_tuple

    @functools.wraps(original)
    def graph_free_callable(*args: Any, **kwargs: Any) -> Any:
        if len(args) >= 3:
            positional = list(args)
            positional[2] = warp_ffi.GraphMode.NONE
            args = tuple(positional)
        else:
            kwargs["graph_mode"] = warp_ffi.GraphMode.NONE
        return original(*args, **kwargs)

    mjx_warp_ffi.jax_callable_variadic_tuple = graph_free_callable
    try:
        yield
    finally:
        mjx_warp_ffi.jax_callable_variadic_tuple = original


def _require_module(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(f"{module_name} is required by the MuJoCo Playground adapter. Try: {install_hint}")


def _resolve_device_rank(device: torch.device, jax_module: Any) -> int | None:
    if device.type != "cuda":
        return None
    device_rank = 0 if device.index is None else int(device.index)
    gpu_devices = jax_module.devices("gpu")
    if device_rank >= len(gpu_devices):
        raise RuntimeError(
            f"CUDA device '{device}' was requested, but JAX only sees {len(gpu_devices)} GPU device(s)."
        )
    return device_rank


def _parse_config_overrides(overrides: Any) -> dict[str, Any]:
    if overrides is None:
        return {}
    if isinstance(overrides, str):
        if overrides.strip() == "":
            return {}
        parsed = json.loads(overrides)
        if not isinstance(parsed, dict):
            raise ValueError("runner.env.playground_config_overrides must decode to a JSON object.")
        return parsed
    if isinstance(overrides, dict):
        return dict(overrides)
    raise ValueError("runner.env.playground_config_overrides must be a dict or JSON object string.")


def _obs_to_torch_dict(obs: Any) -> dict[str, torch.Tensor]:
    if isinstance(obs, dict):
        return {key: _jax_to_torch(value) for key, value in obs.items()}
    return {"state": _jax_to_torch(obs)}


def _jax_to_torch(tensor: Any) -> torch.Tensor:
    import torch.utils.dlpack as tpack

    return tpack.from_dlpack(tensor)


def _torch_to_jax(tensor: torch.Tensor) -> Any:
    from jax.dlpack import from_dlpack

    return from_dlpack(tensor)


def _dependency_error_message(impl: str) -> str:
    return (
        "MuJoCo Playground adapter dependencies are missing. Install MuJoCo Playground with "
        "`python -m pip install playground`; for impl='warp', also install compatible `mujoco-warp` "
        "and `warp-lang`. "
        f"Requested impl={impl!r}."
    )


def _warp_construction_error_message(err: BaseException) -> str:
    versions = {
        name: _package_version(name)
        for name in ("playground", "mujoco", "mujoco-mjx", "mujoco-warp", "warp-lang", "jax", "jaxlib")
    }
    return (
        "Failed to construct the MuJoCo Playground Warp environment. This is usually an install/version "
        f"compatibility issue. Installed versions: {versions}. Original error: {err}"
    )


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
