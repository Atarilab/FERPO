# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import time
import torch
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from tensordict import TensorDict, TensorDictBase
from torch import nn
from typing import Any

from rsl_rl.algorithms import (
    MaxEntMPO,
    MaxEntMPOValue,
    MaxEntMPOCached,
)
from rsl_rl.algorithms.config_validation import (
    validate_critic_reward_normalization,
)
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorQ,
    ActorV,
)
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_obs_groups
from rsl_rl.utils.logger import Logger
from rsl_rl.utils.numerics import non_finite_tensor_description

Algorithm = MaxEntMPO | MaxEntMPOValue | MaxEntMPOCached

_POLICY_CLASSES: dict[str, type[nn.Module]] = {
    policy_class.__name__: policy_class
    for policy_class in (
        ActorQ,
        ActorV,
    )
}
_ALGORITHM_CLASSES: dict[str, type[Algorithm]] = {
    algorithm_class.__name__: algorithm_class
    for algorithm_class in (
        MaxEntMPO,
        MaxEntMPOValue,
        MaxEntMPOCached,
    )
}


@dataclass(frozen=True)
class TrainingSummary:
    """Machine-readable summary of one call to :meth:`OnPolicyRunner.learn`."""

    start_iteration: int
    completed_iterations: int
    environment_steps: int
    collection_size: int
    stopped_early: bool
    stop_reason: str | None


def _to_python_scalar(value: Any) -> float | int | str | bool | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return str(value.detach().cpu().tolist())
    if isinstance(value, (float, int, str, bool)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _env_clip_metadata(env: VecEnv) -> tuple[str, float | int | str | bool | None]:
    sources = [env, getattr(env, "backend", None)]
    env_cfg = getattr(env, "cfg", None)
    if isinstance(env_cfg, dict):
        sources.append(env_cfg)

    action_transform = "none"
    action_clip = None
    for source in sources:
        if source is None:
            continue
        if isinstance(source, dict):
            if source.get("action_transform") is not None:
                action_transform = str(source["action_transform"]).lower()
            if "clip_actions" in source:
                action_clip = source["clip_actions"]
        else:
            if hasattr(source, "action_transform"):
                action_transform = str(source.action_transform).lower()
            if hasattr(source, "clip_actions"):
                action_clip = source.clip_actions

    if action_transform == "tanh":
        transform = "tanh" if action_clip is None else "tanh_then_clip"
        return transform, _to_python_scalar(action_clip)
    if action_clip is not None:
        return "clip", _to_python_scalar(action_clip)
    if action_transform == "none":
        return "none", None
    return action_transform, _to_python_scalar(action_clip)


def _policy_action_bounds(policy: nn.Module) -> dict[str, float | int | str | bool | None] | None:
    if not hasattr(policy, "action_lower_bound") or not hasattr(policy, "action_upper_bound"):
        return None
    return {
        "lower": _to_python_scalar(policy.action_lower_bound),
        "upper": _to_python_scalar(policy.action_upper_bound),
    }


def _effective_action_path(
    policy_action_transform: str,
    policy_distribution_type: str,
    env_action_transform: str,
) -> str:
    if policy_action_transform == "tanh_scaled" and env_action_transform == "clip":
        return "tanh_scaled_then_env_clip_guard"
    if policy_distribution_type == "normal" and env_action_transform == "tanh_then_clip":
        return "normal_sample_then_env_tanh_then_clip_guard"
    if policy_distribution_type == "normal" and env_action_transform == "tanh":
        return "normal_sample_then_env_tanh"
    if policy_action_transform == "tanh_scaled":
        return "tanh_scaled"
    if env_action_transform == "clip":
        if policy_distribution_type == "normal":
            return "normal_sample_then_env_clip"
        return "policy_sample_then_env_clip"
    if policy_distribution_type == "normal":
        return "normal_sample"
    return f"{policy_distribution_type}_sample"


def _entropy_domain(alg: Any, policy_distribution_type: str) -> str:
    if policy_distribution_type == "categorical":
        return "exact_categorical" if getattr(alg, "enumerate_actions", True) else "sampled_categorical"
    algorithm_class = alg.__class__.__name__
    if algorithm_class in {"PPO", "MaxEntPPO"}:
        if hasattr(alg, "use_tanh_entropy") and bool(alg.use_tanh_entropy):
            return "bounded_tanh_mc"
        return "unbounded_normal" if policy_distribution_type == "normal" else "policy_distribution_entropy"

    if hasattr(alg, "use_tanh_entropy"):
        if bool(alg.use_tanh_entropy):
            return "bounded_tanh_mc"
        return "configured_policy_distribution_fallback"

    return "policy_distribution_entropy"


def _tensor_tree_items(value: Any, prefix: str) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield named tensor leaves from nested mappings and TensorDict objects."""
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if isinstance(value, (TensorDictBase, Mapping)):
        for key, child in value.items():
            yield from _tensor_tree_items(child, f"{prefix}[{key}]")


def _timeout_source(env: VecEnv) -> str:
    env_cfg = getattr(env, "cfg", {})
    env_kind = str(env_cfg.get("kind", "")).lower() if isinstance(env_cfg, dict) else ""
    backend = getattr(env, "backend", None)
    class_names = {env.__class__.__name__, backend.__class__.__name__ if backend is not None else ""}
    if env_kind == "mujoco_playground" or "MuJoCoPlaygroundVecEnv" in class_names:
        return 'mujoco_playground_info["truncation"]'
    return "env_extras_time_outs"


def build_run_semantics_metadata(
    env: VecEnv,
    policy: nn.Module,
    alg: Any,
) -> dict[str, Any]:
    policy_distribution_type = str(getattr(policy, "distribution_type", "normal"))
    if policy_distribution_type == "tanh":
        policy_action_transform = "tanh_scaled"
    else:
        policy_action_transform = "none"
    env_action_transform, env_action_clip = _env_clip_metadata(env)

    metadata = {
        "algorithm_class": alg.__class__.__name__,
        "policy_class": policy.__class__.__name__,
        "policy_distribution_type": policy_distribution_type,
        "policy_action_transform": policy_action_transform,
        "policy_action_bounds": _policy_action_bounds(policy),
        "env_action_transform": env_action_transform,
        "env_action_clip": env_action_clip,
        "effective_action_path": _effective_action_path(
            policy_action_transform,
            policy_distribution_type,
            env_action_transform,
        ),
        "entropy_domain": _entropy_domain(alg, policy_distribution_type),
        "timeout_key": 'extras["time_outs"]',
        "timeout_source": _timeout_source(env),
        "timeout_bootstrap_enabled": True,
    }

    if hasattr(env, "max_episode_length"):
        metadata["max_episode_length"] = _to_python_scalar(env.max_episode_length)
    action_repeat = getattr(getattr(env, "backend", None), "action_repeat", None)
    if action_repeat is None:
        env_cfg = getattr(env, "cfg", None)
        if isinstance(env_cfg, dict):
            action_repeat = env_cfg.get("action_repeat")
    if action_repeat is not None:
        metadata["action_repeat"] = _to_python_scalar(action_repeat)

    action_chunk_size = int(getattr(policy, "action_chunk_size", 1))
    metadata["action_chunk_size"] = action_chunk_size
    metadata["environment_steps_per_policy_step"] = int(
        getattr(env, "environment_steps_per_policy_step", 1)
    )
    if action_chunk_size > 1:
        metadata["environment_action_dimension"] = int(
            policy.environment_action_dim
        )
        metadata["policy_action_dimension"] = int(policy.num_actions)
        metadata["action_chunk_reward_semantics"] = (
            "discounted_substep_sum_for_learning_undiscounted_sum_for_logging"
        )

    return metadata


class OnPolicyRunner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        algorithm_class_override: type[Algorithm] | None = None,
    ) -> None:
        self.cfg = train_cfg
        self.policy_cfg = train_cfg["policy"]
        self.alg_cfg = train_cfg["algorithm"]
        self.device = device
        if int(self.policy_cfg.get("action_chunk_size", 1)) > 1:
            raise ValueError(
                "`policy.action_chunk_size > 1` is not supported by MaxEntMPO policies."
            )
        self.env = env
        self.environment_steps_per_policy_step = int(
            getattr(self.env, "environment_steps_per_policy_step", 1)
        )
        self.algorithm_class_override = algorithm_class_override
        self.init_at_random_ep_len = bool(train_cfg.get("init_at_random_ep_len", True))
        self.log_interval_percent = float(train_cfg.get("log_interval_percent", 2.0))
        validate_critic_reward_normalization(
            self.alg_cfg.get("critic_loss_type"),
            bool(self.alg_cfg.get("normalize_rewards", False)),
        )

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], self._get_default_obs_sets())

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)
        environment_action_transform = getattr(self.alg, "environment_action_transform", None)
        if environment_action_transform is not None:
            set_action_transform = getattr(self.env, "set_action_transform", None)
            if not callable(set_action_transform):
                raise ValueError(
                    f"{type(self.alg).__name__} requested the environment action transform "
                    f"{environment_action_transform!r}, but {type(self.env).__name__} does not support it."
                )
            set_action_transform(environment_action_transform)
        run_semantics = build_run_semantics_metadata(self.env, self.alg.policy, self.alg)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
            run_semantics=run_semantics,
            environment_steps_per_policy_step=self.environment_steps_per_policy_step,
        )

        self.current_learning_iteration = 0

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool | None = None,
        eval_fn=None,
        eval_interval: int = 0,
        *,
        max_env_steps: int | None = None,
        checkpoint_env_steps: Iterable[int] | None = None,
        progress_callback: Callable[[dict[str, float | int]], None] | None = None,
        checkpoint_callback: Callable[[int], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        environment_step_offset: int = 0,
        total_environment_step_target: int | None = None,
    ) -> TrainingSummary:
        # Randomize initial episode lengths (for exploration)
        if self._should_init_at_random_ep_len(init_at_random_ep_len):
            self._randomize_initial_episode_lengths()

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_it = self.current_learning_iteration
        requested_iterations = int(num_learning_iterations)
        if requested_iterations < 1:
            raise ValueError(f"`num_learning_iterations` must be positive, got {requested_iterations}.")
        environment_step_offset = int(environment_step_offset)
        if environment_step_offset < 0:
            raise ValueError(
                "`environment_step_offset` must be non-negative, got "
                f"{environment_step_offset}."
            )
        if total_environment_step_target is not None:
            total_environment_step_target = int(total_environment_step_target)
            if total_environment_step_target < environment_step_offset:
                raise ValueError(
                    "`total_environment_step_target` cannot be smaller than "
                    "`environment_step_offset`."
                )

        collection_size = (
            self.cfg["num_steps_per_env"]
            * self.env.num_envs
            * self.gpu_world_size
            * self.environment_steps_per_policy_step
        )
        if max_env_steps is not None:
            max_env_steps = int(max_env_steps)
            max_complete_iterations = max_env_steps // collection_size
            if max_complete_iterations < 1:
                raise ValueError(
                    f"`max_env_steps` ({max_env_steps}) must fit at least one complete rollout "
                    f"of {collection_size} environment steps."
                )
            requested_iterations = min(requested_iterations, max_complete_iterations)

        total_it = start_it + requested_iterations
        log_interval = self._resolve_log_interval(requested_iterations)
        checkpoint_targets = sorted({int(step) for step in (checkpoint_env_steps or ()) if int(step) > 0})
        next_checkpoint = 0
        completed_iterations = 0
        completed_env_steps = 0
        stopped_early = False
        stop_reason = None
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    query_state = (
                        self.env.capture_transition_query_state()
                        if getattr(self.alg, "fresh_transition_updates", False)
                        else None
                    )
                    # Sample actions
                    actions = self.alg.act(obs)
                    if query_state is not None:
                        self.alg.record_transition_query_state(query_state)
                    # scale actions
                    if self.alg_cfg.get("scale_actions", False):
                        upper = self.alg_cfg.get("action_upper_bound", 1.0)
                        lower = self.alg_cfg.get("action_lower_bound", -1.0)
                        actions = actions * (upper - lower) / 2.0 + (upper + lower) / 2.0

                    # Step the environment
                    # print("mean abs actions: ", actions.abs().mean())  # DEBUG
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg_cfg["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                rollout_diagnostics = self._validate_and_summarize_rollout(obs)
                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            # Evaluation time is not training time. Stop the learning timer before
            # running deterministic evaluation.
            stop = time.time()
            learn_time = stop - start

            eval_metrics = None
            evaluation_due = (
                eval_fn is not None
                and eval_interval > 0
                and (
                    (it + 1) % eval_interval == 0
                    or it == total_it - 1
                )
            )

            if evaluation_due:
                self.eval_mode()
                try:
                    with torch.inference_mode():
                        eval_metrics = eval_fn(self.alg.policy)
                finally:
                    self.train_mode()

                # Critical for IsaacLab, where evaluation uses and resets the same
                # underlying environment. Harmless for separate MuJoCo/ManiSkill
                # evaluation environments.
                obs = self.env.get_observations().to(self.device)

            self.current_learning_iteration = it

            next_completed_iterations = completed_iterations + 1
            next_completed_env_steps = next_completed_iterations * collection_size
            total_environment_steps = (
                environment_step_offset + next_completed_env_steps
            )

            # Log information. Evaluation iterations must emit immediately so
            # eval/* is committed on the same Train/environment_steps row.
            emit_log = self._should_emit_log(it, start_it, total_it, log_interval)
            if not self.logger.uses_percent_log_interval():
                emit_log = True
            if eval_metrics is not None:
                emit_log = True

            # Queue deterministic evaluation scalars before logger.log(), because
            # the W&B writer commits the complete row inside logger.log().
            if (
                eval_metrics is not None
                and self.logger.writer is not None
                and not self.logger.disable_logs
            ):
                for key, value in eval_metrics.items():
                    self.logger.writer.add_scalar(key, value, it)

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=self.alg.rnd.weight if self.alg_cfg["rnd_cfg"] else None,
                reward_norm_metrics=(
                    self.alg.reward_normalizer_metrics() if hasattr(self.alg, "reward_normalizer_metrics") else {}
                ),
                environment_steps=total_environment_steps,
                max_environment_steps=total_environment_step_target,
                emit=emit_log,
            )


            self._validate_finite_training_state(loss_dict)
            completed_iterations = next_completed_iterations
            completed_env_steps = next_completed_env_steps
            progress = {
                "iteration": it,
                "completed_iterations": completed_iterations,
                "environment_steps": completed_env_steps,
                "total_environment_steps": total_environment_steps,
                "collection_size": collection_size,
                "collect_time_seconds": collect_time,
                "learn_time_seconds": learn_time,
                "iteration_time_seconds": collect_time + learn_time,
                "steps_per_second": collection_size / max(collect_time + learn_time, 1.0e-12),
                "learning_rate": float(self.alg.learning_rate),
                "mean_action_std": float(self.alg.policy.action_std.mean().item()),
                **rollout_diagnostics,
                **{key: float(value) for key, value in loss_dict.items()},
            }
            if self.logger.rewbuffer:
                progress["train_mean_reward"] = float(sum(self.logger.rewbuffer) / len(self.logger.rewbuffer))
                progress["train_mean_episode_length"] = float(sum(self.logger.lenbuffer) / len(self.logger.lenbuffer))
            if eval_metrics is not None:
                progress.update({key: float(value) for key, value in eval_metrics.items()})
            if progress_callback is not None:
                progress_callback(progress)

            while (
                next_checkpoint < len(checkpoint_targets) and completed_env_steps >= checkpoint_targets[next_checkpoint]
            ):
                if checkpoint_callback is not None:
                    checkpoint_callback(completed_env_steps)
                elif self.logger.log_dir is not None and not self.logger.disable_logs:
                    self.save(
                        os.path.join(
                            self.logger.log_dir,
                            f"model_steps_{completed_env_steps}.pt",
                        )
                    )
                next_checkpoint += 1

            # Optional periodic checkpoints. Set save_interval: 0 to disable.
            save_interval = int(self.cfg.get("save_interval", 0))
            if (
                save_interval > 0
                and (it + 1) % save_interval == 0
                and self.logger.log_dir is not None
                and not self.logger.disable_logs
            ):
                self.save(os.path.join(self.logger.log_dir, f"model_{it + 1}.pt"))  # type: ignore

            if stop_requested is not None and stop_requested():
                stopped_early = True
                stop_reason = "stop_requested"
                break


        return TrainingSummary(
            start_iteration=start_it,
            completed_iterations=completed_iterations,
            environment_steps=completed_env_steps,
            collection_size=collection_size,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )

    def _validate_and_summarize_rollout(self, final_obs: TensorDict) -> dict[str, float]:
        storage = self.alg.storage
        finite_tensors = [
            ("rollout.actions", storage.actions, 1),
            ("rollout.rewards", storage.rewards, 1),
            ("rollout.critic_values", storage.values, 1),
            *(
                (name, tensor, 1)
                for name, tensor in _tensor_tree_items(
                    storage.observations,
                    "rollout.observations",
                )
            ),
            *(
                (name, tensor, 0)
                for name, tensor in _tensor_tree_items(
                    final_obs,
                    "rollout.final_observations",
                )
            ),
        ]
        for name, tensor, environment_axis in finite_tensors:
            description = non_finite_tensor_description(
                name,
                tensor,
                environment_axis=environment_axis,
            )
            if description is not None:
                raise FloatingPointError(f"Non-finite rollout tensor detected: {description}.")

        if hasattr(storage, "final_env_steps"):
            final_mask = storage.final_env_steps.squeeze(-1).bool()
            final_latents = storage.actions[final_mask]
            transform_final_action = getattr(
                self.alg.policy,
                "transform_final_action",
                None,
            )
            executed_actions = (
                transform_final_action(final_latents)
                if callable(transform_final_action)
                else torch.tanh(final_latents)
            )
        else:
            executed_actions = storage.actions
        diagnostics = {
            "action_saturation_fraction": float(
                (executed_actions.abs() >= 0.98).float().mean().item()
            )
        }
        policy = self.alg.policy
        if getattr(policy, "is_discrete", False):
            diagnostics = {
                "invalid_action_fraction": float(
                    ((executed_actions < 0) | (executed_actions >= policy.num_actions)).float().mean().item()
                )
            }
        if hasattr(policy, "vmin") and hasattr(policy, "vmax"):
            vmin = float(policy.vmin)
            vmax = float(policy.vmax)
            margin = 0.05 * (vmax - vmin)
            near_boundary = (storage.values <= vmin + margin) | (storage.values >= vmax - margin)
            diagnostics["critic_support_saturation_fraction"] = float(near_boundary.float().mean().item())
        return diagnostics

    def _validate_finite_training_state(self, loss_dict: dict[str, float]) -> None:
        non_finite_metrics = [key for key, value in loss_dict.items() if not math.isfinite(float(value))]
        if non_finite_metrics:
            joined = ", ".join(sorted(non_finite_metrics))
            raise FloatingPointError(f"Non-finite training metrics detected: {joined}.")

        non_finite_parameters = [
            name for name, parameter in self.alg.policy.named_parameters() if not torch.isfinite(parameter).all()
        ]
        if non_finite_parameters:
            joined = ", ".join(non_finite_parameters[:10])
            raise FloatingPointError(f"Non-finite policy parameters detected: {joined}.")

    def _should_init_at_random_ep_len(self, init_at_random_ep_len: bool | None) -> bool:
        if init_at_random_ep_len is None:
            return self.init_at_random_ep_len
        return bool(init_at_random_ep_len)

    def _randomize_initial_episode_lengths(self) -> None:
        episode_lengths = self.env.episode_length_buf
        max_episode_length = torch.as_tensor(
            self.env.max_episode_length,
            device=episode_lengths.device,
        )
        if max_episode_length.ndim == 0:
            high = int(max_episode_length.item())
            if high <= 0:
                raise ValueError("env.max_episode_length must be positive.")
            randomized_lengths = torch.randint(
                high=high,
                size=episode_lengths.shape,
                device=episode_lengths.device,
                dtype=episode_lengths.dtype,
            )
        else:
            max_episode_length = torch.broadcast_to(max_episode_length, episode_lengths.shape)
            if torch.any(max_episode_length <= 0):
                raise ValueError("env.max_episode_length entries must be positive.")
            randomized_lengths = torch.floor(
                torch.rand(
                    episode_lengths.shape,
                    device=episode_lengths.device,
                    dtype=torch.float32,
                )
                * max_episode_length.to(dtype=torch.float32)
            ).to(dtype=episode_lengths.dtype)
        episode_lengths.copy_(randomized_lengths)

    def _resolve_log_interval(self, num_learning_iterations: int) -> int:
        if self.log_interval_percent <= 0.0:
            return 1
        return max(1, math.ceil(num_learning_iterations * self.log_interval_percent / 100.0))

    @staticmethod
    def _should_emit_log(it: int, start_it: int, total_it: int, log_interval: int) -> bool:
        if log_interval <= 1:
            return True
        return it == start_it or it == total_it - 1 or (it - start_it) % log_interval == 0

    def save(self, path: str, infos: dict | None = None) -> None:
        # Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if hasattr(self.alg, "reward_normalizer_state_dict"):
            reward_normalizer_state_dict = self.alg.reward_normalizer_state_dict()
            if reward_normalizer_state_dict is not None:
                saved_dict["reward_normalizer_state_dict"] = reward_normalizer_state_dict
        if hasattr(self.alg, "policy_snapshot_state_dict"):
            saved_dict["policy_snapshot_state_dict"] = self.alg.policy_snapshot_state_dict()
        # Save RND model if used
        if self.alg_cfg["rnd_cfg"]:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if resumed_training and hasattr(self.alg, "load_policy_snapshot_state_dict"):
            self.alg.load_policy_snapshot_state_dict(loaded_dict.get("policy_snapshot_state_dict"))
        # Load RND model if used
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if hasattr(self.alg, "load_reward_normalizer_state_dict"):
            self.alg.load_reward_normalizer_state_dict(loaded_dict.get("reward_normalizer_state_dict"))
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # RND optimizer if used
            if self.alg_cfg["rnd_cfg"]:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # Load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        # PPO
        self.alg.policy.train()
        if hasattr(self.alg, "train_mode"):
            self.alg.train_mode()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        # PPO
        self.alg.policy.eval()
        if hasattr(self.alg, "eval_mode"):
            self.alg.eval_mode()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.git_status_repos.append(repo_file_path)

    def _get_default_obs_sets(self) -> list[str]:
        """Get the the default observation sets required for the algorithm.

        .. note::
            See :func:`resolve_obs_groups` for more details on the handling of observation sets.
        """
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        return default_sets

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs: TensorDict) -> Algorithm:
        """Construct the actor-critic algorithm."""
        for feature in ("rnd_cfg", "symmetry_cfg"):
            if self.alg_cfg.get(feature):
                raise ValueError(f"{feature} is outside the FERPO paper configurations.")
            self.alg_cfg[feature] = None

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        policy_class_name = self.policy_cfg.pop("class_name")
        try:
            actor_critic_class = _POLICY_CLASSES[policy_class_name]
        except KeyError as error:
            raise ValueError(
                f"Unsupported policy class {policy_class_name!r}. Available classes: {sorted(_POLICY_CLASSES)}."
            ) from error
        actor_critic: nn.Module = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        storage = RolloutStorage(
            "rl",
            self.env.num_envs,
            self.cfg["num_steps_per_env"],
            obs,
            getattr(actor_critic, "action_shape", [self.env.num_actions]),
            self.device,
            actions_dtype=getattr(actor_critic, "action_dtype", torch.float32),
        )

        # Initialize the algorithm
        algorithm_class_name = self.alg_cfg.pop("class_name")
        if self.algorithm_class_override is not None:
            alg_class = self.algorithm_class_override
        else:
            try:
                alg_class = _ALGORITHM_CLASSES[algorithm_class_name]
            except KeyError as error:
                raise ValueError(
                    f"Unsupported algorithm class {algorithm_class_name!r}. "
                    f"Available classes: {sorted(_ALGORITHM_CLASSES)}."
                ) from error
        alg: Algorithm = alg_class(
            actor_critic, storage, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )
        if getattr(alg, "fresh_transition_updates", False):
            alg.set_transition_query_env(self.env)

        return alg
