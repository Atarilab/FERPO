#!/usr/bin/env python3
# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# ruff: noqa: E402
import argparse
import math
import pathlib
import sys
import torch
import copy
from collections.abc import Mapping

# Allow running this script from source without installing the package.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.run_utils import (
    RunArtifacts,
    StopSignal,
    classify_failure,
    complete_iterations_for_step_limit,
    parse_step_list,
    runtime_metadata,
    seed_everything,
)
from examples.training_utils import (
    apply_no_log_checkpoint_dir,
    build_env,
    load_train_cfg,
    resolve_log_dir,
    update_run_metadata,
)
from rsl_rl.runners import OnPolicyRunner


def evenly_spaced_checkpoint_steps(
    *,
    num_learning_iterations: int,
    collection_size: int,
    num_intervals: int,
) -> tuple[int, ...]:
    """Return rollout-aligned checkpoint steps, including the final rollout."""
    if num_learning_iterations < 1:
        raise ValueError("`num_learning_iterations` must be positive.")
    if collection_size < 1:
        raise ValueError("`collection_size` must be positive.")
    if num_intervals < 1:
        raise ValueError("`checkpointing.num_intervals` must be positive.")

    checkpoint_iterations = {
        math.ceil(index * num_learning_iterations / num_intervals)
        for index in range(1, num_intervals + 1)
    }
    return tuple(
        iteration * collection_size
        for iteration in sorted(checkpoint_iterations)
    )


def parse_args(argv=None, default_config=None, default_env=None, default_device="cuda:0", default_log_dir="auto"):
    parser = argparse.ArgumentParser(description="Train FERPO on the paper's MuJoCo Playground or ManiSkill tasks.")
    parser.add_argument("--config", type=pathlib.Path, required=default_config is None, default=default_config,
                        help="Path to a paper YAML configuration under config/.")
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--task-id")
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--checkpoint-env-steps")
    parser.add_argument("--log-dir", default=default_log_dir)
    parser.add_argument("--warp-graph-mode", choices=["warp", "none", "recorded"], default="warp",
                        help="Use Warp capture for new runs; 'recorded' preserves the YAML execution mode.")
    parser.set_defaults(env=None, env_id=None, env_factory=None, env_backend="auto")
    return parser.parse_args(argv)


def resolve_config_path(args: argparse.Namespace) -> pathlib.Path:
    return args.config


def _env_tag(args: argparse.Namespace, train_cfg: dict, resolved_backend: str) -> str:
    if args.env is not None:
        return str(args.env).lower()
    if args.env_id is not None:
        return args.env_id.lower().replace("-", "_")

    env_cfg = train_cfg.get("env", {})
    if "task_id" in env_cfg:
        return str(env_cfg["task_id"]).lower()
    if "id" in env_cfg:
        return str(env_cfg["id"]).lower().replace("-", "_")
    if "kind" in env_cfg:
        return str(env_cfg["kind"]).lower()
    return resolved_backend

@torch.inference_mode()
def deterministic_evaluate(
    policy,
    eval_env,
    device: torch.device,
    *,
    initial_eval_state=None,
    reset_kwargs=None,
):
    reset_kwargs = reset_kwargs or {}

    # MuJoCo Playground:
    # restore exactly the same initial evaluation state every time.
    if initial_eval_state is not None:
        eval_env.restore_transition_query_state(
            initial_eval_state
        )
        obs = eval_env.get_observations().to(device)

    # ManiSkill:
    # use their backend-specific reset behavior.
    else:
        obs = eval_env.reset(
            **reset_kwargs
        ).to(device)

    episode_returns = torch.zeros(
        eval_env.num_envs,
        device=device,
        dtype=torch.float32,
    )

    episode_lengths = torch.zeros(
        eval_env.num_envs,
        device=device,
        dtype=torch.float32,
    )

    done_mask = torch.zeros(
        eval_env.num_envs,
        device=device,
        dtype=torch.bool,
    )
    success_once = torch.zeros(
        eval_env.num_envs,
        device=device,
        dtype=torch.bool,
    )
    success_available = False


    max_steps = int(
        getattr(
            eval_env,
            "max_episode_length",
            getattr(
                eval_env,
                "max_episode_steps",
                1000,
            ),
        )
    )

    for _ in range(max_steps):

        # Deterministic policy action
        actions = policy.act_inference(obs)

        obs, rewards, dones, extras = eval_env.step(
            actions.to(eval_env.device)
        )

        obs = obs.to(device)
        rewards = rewards.to(device).reshape(-1)
        dones = dones.to(device).reshape(-1).bool()

        episode = extras.get("episode") if isinstance(extras, Mapping) else None
        if isinstance(episode, Mapping):
            for key in ("success_once", "success_at_end", "success"):
                if key not in episode:
                    continue
                values = torch.as_tensor(episode[key], device=device).reshape(-1).bool()
                done_indices = torch.nonzero(dones, as_tuple=False).view(-1)
                if values.numel() == eval_env.num_envs:
                    success_once |= values
                elif values.numel() == done_indices.numel():
                    success_once[done_indices] |= values
                else:
                    raise ValueError(
                        f"Evaluation metric {key!r} has {values.numel()} values for "
                        f"{done_indices.numel()} completed environments."
                    )
                success_available = True
                break

        active = ~done_mask

        episode_returns[active] += rewards[active]
        episode_lengths[active] += 1.0


        done_mask |= dones

        if done_mask.all():
            break

    metrics = {
        "eval/avg_return":
            episode_returns.mean().item(),

        "eval/avg_length":
            episode_lengths.mean().item(),
    }
    if success_available:
        metrics["eval/success"] = success_once.float().mean().item()


    success_text = (
        f", success={metrics['eval/success']:.4f}"
        if "eval/success" in metrics
        else ""
    )
    print(
        "Deterministic evaluation: "
        f"return={metrics['eval/avg_return']:.2f}, "
        f"length={metrics['eval/avg_length']:.2f}{success_text}"
    )

    return metrics


def apply_maniskill_reppo_discount(train_cfg: dict, env: object, resolved_backend: str) -> None:
    """Apply REPPO's ManiSkill gamma = 1 - offset / native horizon rule."""
    if resolved_backend != "maniskill3":
        return
    env_cfg = train_cfg.get("env", {})
    if not bool(env_cfg.get("reppo_gamma_from_native_horizon", False)):
        return

    horizon = int(getattr(env, "max_episode_length"))
    offset = float(env_cfg.get("reppo_gamma_horizon_offset", 10.0))
    gamma = 1.0 - offset / horizon
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            "REPPO's ManiSkill discount rule produced an invalid gamma: "
            f"horizon={horizon}, offset={offset}, gamma={gamma}."
        )

    env_cfg["max_episode_length"] = horizon
    train_cfg["algorithm"]["gamma"] = gamma
    print(
        "REPPO ManiSkill discount: "
        f"native_horizon={horizon}, offset={offset:g}, gamma={gamma:.8f}"
    )


def run_training(args: argparse.Namespace) -> None:
    config_path = resolve_config_path(args)
    train_cfg = load_train_cfg(config_path)
    if "env" not in train_cfg or train_cfg["env"] is None:
        train_cfg["env"] = {}
    env_cfg = train_cfg["env"]

    if args.num_envs is not None:
        env_cfg["num_envs"] = int(args.num_envs)
    if args.task_id is not None:
        env_cfg["task_id"] = args.task_id
    if args.env_id is not None:
        env_cfg["id"] = args.env_id
    if args.seed is not None:
        train_cfg["seed"] = int(args.seed)
    if args.run_id is not None:
        train_cfg["run_name"] = args.run_id
    if env_cfg.get("kind") == "mujoco_playground" and args.warp_graph_mode != "recorded":
        env_cfg["warp_graph_mode"] = args.warp_graph_mode

    seed_everything(int(train_cfg.get("seed", 1)))

    run_id = args.run_id
    if run_id is None and args.output_dir is not None:
        run_id = args.output_dir.name
    artifacts = RunArtifacts(args.output_dir, run_id or "run") if args.output_dir is not None else None
    stop_signal = StopSignal()
    stop_signal.install()
    algo_tag = str(train_cfg["algorithm"]["class_name"]).lower()
    update_run_metadata(train_cfg, _env_tag(args, train_cfg, args.env_backend), algo_tag)

    def env_close_fn() -> None:
        return None

    def eval_close_fn() -> None:
        return None

    try:
        env, env_close_fn, resolved_backend = build_env(
            env_backend=args.env_backend,
            train_cfg=train_cfg,
            device=args.device,
            env_id=args.env_id,
            env_factory=args.env_factory,
        )
        apply_maniskill_reppo_discount(train_cfg, env, resolved_backend)

        # Write artifacts only after environment-dependent parameters such as
        # ManiSkill's native-horizon gamma have been resolved.
        if artifacts is not None:
            artifacts.start(train_cfg, runtime_metadata(REPO_ROOT, args.device))

        evaluation_cfg = train_cfg.get("evaluation", {}) or {}
        evaluation_enabled = bool(evaluation_cfg.get("enabled", False))
        eval_fn = None

        if evaluation_enabled and resolved_backend == "mujoco_playground":
            eval_train_cfg = copy.deepcopy(train_cfg)

            # Honor the explicitly configured evaluation size.  Previously
            # this always duplicated the full training environment (often
            # another 1,024 Warp environments), which is especially costly
            # for G1/T1 and Walker tasks.
            eval_train_cfg["env"]["num_envs"] = int(
                evaluation_cfg.get("num_envs", min(int(env.num_envs), 128))
            )

            eval_train_cfg["env"]["collect_step_metrics"] = False

            eval_env, eval_close_fn, _ = build_env(
                env_backend=resolved_backend,
                train_cfg=eval_train_cfg,
                device=args.device,
                env_id=args.env_id,
                env_factory=args.env_factory,
            )

            initial_eval_state = (
                eval_env.capture_transition_query_state()
            )

            eval_device = torch.device(args.device)

            def eval_fn(policy):
                return deterministic_evaluate(
                    policy=policy,
                    eval_env=eval_env,
                    initial_eval_state=initial_eval_state,
                    device=eval_device,
                )

        if evaluation_enabled and resolved_backend == "maniskill3":
            eval_train_cfg = copy.deepcopy(train_cfg)

            # Do not silently duplicate the full training simulator.  A
            # 1,024-environment ManiSkill training run previously created a
            # second 1,024-environment SAPIEN instance here and kept it alive
            # for the entire run, which can exhaust both GPU and host memory.
            eval_train_cfg["env"]["num_envs"] = int(
                evaluation_cfg.get("num_envs", min(int(env.num_envs), 128))
            )

            # Keep the scene fixed and reset its state for repeatable evaluation.
            eval_train_cfg["env"]["reconfiguration_freq"] = None
            eval_train_cfg["env"]["ignore_terminations"] = True
            eval_train_cfg["env"]["record_metrics"] = True

            eval_env, eval_close_fn, _ = build_env(
                env_backend="maniskill3",
                train_cfg=eval_train_cfg,
                device=args.device,
                env_id=args.env_id,
                env_factory=args.env_factory,
            )


        configured_max_iters = int(args.max_iters if args.max_iters is not None else train_cfg["max_iterations"])
        max_iters = configured_max_iters
        if args.max_env_steps is not None:
            # An explicit interaction budget is authoritative.  The YAML
            # iteration count remains a cap only when the caller also supplied
            # --max-iters explicitly.
            iteration_cap = (
                configured_max_iters
                if args.max_iters is not None
                else int(args.max_env_steps)
                // (
                    env.num_envs
                    * int(train_cfg["num_steps_per_env"])
                )
            )
            max_iters = complete_iterations_for_step_limit(
                max_env_steps=args.max_env_steps,
                num_envs=env.num_envs,
                num_steps_per_env=int(train_cfg["num_steps_per_env"]),
                configured_max_iterations=iteration_cap,
                environment_steps_per_policy_step=1,
            )
        if evaluation_enabled:
            num_eval = int(evaluation_cfg.get("num_eval", 20))
            if num_eval < 1:
                raise ValueError("`evaluation.num_eval` must be positive when evaluation is enabled.")
            eval_interval = max(1, max_iters // num_eval)
        else:
            eval_interval = 0
            print("Inline deterministic evaluation disabled by configuration.")
        log_dir = (
            str(args.output_dir.resolve()) if args.output_dir is not None else resolve_log_dir(args.log_dir, train_cfg)
        )
        runner = OnPolicyRunner(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=args.device,
        )

        if evaluation_enabled and resolved_backend == "maniskill3":

            # Use a deterministic evaluation stream that is independent of
            # the corresponding training stream and identical at every
            # evaluation checkpoint within the run.
            eval_seed = int(train_cfg.get("seed", 0)) + 1_000_000

            def eval_fn(policy):
                return deterministic_evaluate(
                    policy=policy,
                    eval_env=eval_env,
                    device=torch.device(args.device),
                    reset_kwargs={"seed": eval_seed},
                )

        if log_dir is None:
            apply_no_log_checkpoint_dir(runner)

        checkpointing_cfg = train_cfg.get("checkpointing", {}) or {}
        checkpointing_enabled = bool(checkpointing_cfg.get("enabled", False))
        checkpoint_env_steps = parse_step_list(args.checkpoint_env_steps)
        if checkpointing_enabled and not checkpoint_env_steps:
            collection_size = (
                int(train_cfg["num_steps_per_env"])
                * int(env.num_envs)
                * int(runner.gpu_world_size)
                * int(runner.environment_steps_per_policy_step)
            )
            checkpoint_env_steps = evenly_spaced_checkpoint_steps(
                num_learning_iterations=max_iters,
                collection_size=collection_size,
                num_intervals=int(checkpointing_cfg.get("num_intervals", 20)),
            )

        if (
            checkpointing_enabled
            and bool(checkpointing_cfg.get("save_initial", False))
            and log_dir is not None
        ):
            initial_checkpoint = pathlib.Path(log_dir) / "model_steps_0.pt"
            runner.save(str(initial_checkpoint), infos={"environment_steps": 0})

        summary = runner.learn(
            num_learning_iterations=max_iters,
            max_env_steps=args.max_env_steps,
            checkpoint_env_steps=checkpoint_env_steps,
            progress_callback=artifacts.append_progress if artifacts is not None else None,
            stop_requested=stop_signal,
            eval_fn=eval_fn,
            eval_interval=eval_interval,
        )

        final_checkpoint = None
        if (
            checkpointing_enabled
            and bool(checkpointing_cfg.get("save_final", True))
            and log_dir is not None
        ):
            final_checkpoint = pathlib.Path(log_dir) / f"model_steps_{summary.environment_steps}.pt"
            if not final_checkpoint.exists():
                runner.save(
                    str(final_checkpoint),
                    infos={"environment_steps": int(summary.environment_steps)},
                )
        if artifacts is not None:
            artifacts.finish(summary, final_checkpoint)
    except BaseException as error:
        if artifacts is not None:
            artifacts.fail(error, classify_failure(error))
        raise
    finally:
        eval_close_fn()
        env_close_fn()


def main(
    argv: list[str] | None = None,
    default_config: pathlib.Path | None = None,
    default_env: str | None = None,
    default_device: str = "cuda:0",
    default_log_dir: str = "auto",
) -> None:
    args = parse_args(
        argv=argv,
        default_config=default_config,
        default_env=default_env,
        default_device=default_device,
        default_log_dir=default_log_dir,
    )
    run_training(args)


if __name__ == "__main__":
    main()
