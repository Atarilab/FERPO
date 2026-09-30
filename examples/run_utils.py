from __future__ import annotations

import importlib.metadata
import json
import math
import numpy as np
import os
import pathlib
import platform
import random
import re
import signal
import sys
import time
import torch
from dataclasses import asdict, is_dataclass
from typing import Any

import yaml

def complete_iterations_for_step_limit(
    max_env_steps: int,
    num_envs: int,
    num_steps_per_env: int,
    configured_max_iterations: int,
    environment_steps_per_policy_step: int = 1,
) -> int:
    """Return complete rollout iterations that do not exceed an interaction limit."""
    values = {
        "max_env_steps": max_env_steps,
        "num_envs": num_envs,
        "num_steps_per_env": num_steps_per_env,
        "configured_max_iterations": configured_max_iterations,
        "environment_steps_per_policy_step": environment_steps_per_policy_step,
    }
    for name, value in values.items():
        if int(value) < 1:
            raise ValueError(f"`{name}` must be positive, got {value}.")
    collection_size = (
        int(num_envs)
        * int(num_steps_per_env)
        * int(environment_steps_per_policy_step)
    )
    iterations = int(max_env_steps) // collection_size
    if iterations < 1:
        raise ValueError(
            f"`max_env_steps` ({max_env_steps}) is smaller than one complete rollout ({collection_size} steps)."
        )
    return min(iterations, int(configured_max_iterations))


def parse_step_list(value: str | None) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return ()
    steps = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if any(step <= 0 for step in steps):
        raise ValueError("Checkpoint environment-step targets must be positive.")
    return steps


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch without enabling slower deterministic kernels."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def close_runner_logger(runner: Any) -> None:
    """Flush and close an RSL-RL runner's optional logging backend."""
    writer = runner.logger.writer
    if writer is None:
        return
    writer.flush()
    stop = getattr(writer, "stop", None)
    if callable(stop):
        stop()
    writer.close()


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return value


def atomic_write_json(path: pathlib.Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_value(document), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata(repo_root: pathlib.Path, device: str) -> dict[str, Any]:
    packages = {
        package: _package_version(package)
        for package in (
            "torch",
            "tensordict",
            "playground",
            "jax",
            "jaxlib",
            "mujoco",
            "mujoco-mjx",
            "mujoco-warp",
            "warp-lang",
            "mani-skill",
        )
    }

    commit = None
    dirty = None
    release_name = repo_root.resolve().name
    if commit is None and re.fullmatch(r"[0-9a-f]{40}", release_name):
        commit = release_name
        dirty = False
    try:
        import git
    except ImportError:
        git = None
    if git is not None:
        try:
            repository = git.Repo(repo_root, search_parent_directories=True)
            commit = repository.head.commit.hexsha
            dirty = repository.is_dirty(untracked_files=True)
        except (
            git.InvalidGitRepositoryError,
            git.NoSuchPathError,
            OSError,
            ValueError,
        ):
            pass

    gpu = None
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if torch_device.index is None else torch_device.index
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": [properties.major, properties.minor],
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    return {
        "created_at_unix": time.time(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": commit,
        "git_dirty": dirty,
        "benchmark_gpu_share": float(os.environ.get("BENCHMARK_GPU_SHARE", "1")),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "gpu": gpu,
    }


class StopSignal:
    """Turn SIGINT/SIGTERM into an iteration-boundary stop request."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signal_number: int, _frame: Any) -> None:
        self.requested = True
        self.signal_number = signal_number

    def __call__(self) -> bool:
        return self.requested


class RunArtifacts:
    """Canonical, append-only benchmark evidence for one training run."""

    def __init__(self, output_dir: pathlib.Path, run_id: str) -> None:
        self.output_dir = output_dir.resolve()
        self.run_id = run_id
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.status_path = self.output_dir / "status.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        self.last_environment_steps = 0

    def start(self, train_cfg: dict, metadata: dict[str, Any]) -> None:
        if (self.output_dir / "run.json").exists() or self.metrics_path.exists():
            raise FileExistsError(
                f"Canonical benchmark artifacts already exist for run {self.run_id!r} in {self.output_dir}."
            )
        (self.output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump({"runner": train_cfg}, sort_keys=False),
            encoding="utf-8",
        )
        atomic_write_json(
            self.output_dir / "run.json",
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "train_cfg": train_cfg,
                "runtime": metadata,
            },
        )
        self._write_status("running")

    def append_progress(self, metrics: dict[str, float | int]) -> None:
        serialized = _json_value(metrics)
        serialized.setdefault("elapsed_wall_time_seconds", time.time() - self.started_at)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serialized, sort_keys=True) + "\n")
            handle.flush()
        self.last_environment_steps = int(metrics["environment_steps"])
        self._write_status("running")

    def finish(self, summary: Any, final_checkpoint: pathlib.Path | None = None) -> None:
        atomic_write_json(
            self.output_dir / "training_summary.json",
            {
                **_json_value(summary),
                "final_checkpoint": None if final_checkpoint is None else str(final_checkpoint),
                "wall_time_seconds": time.time() - self.started_at,
            },
        )
        self._write_status(
            "stopped" if bool(getattr(summary, "stopped_early", False)) else "completed",
            stop_reason=getattr(summary, "stop_reason", None),
        )

    def fail(self, error: BaseException, classification: str = "training_failure") -> None:
        atomic_write_json(
            self.output_dir / "failure.json",
            {
                "classification": classification,
                "error_type": type(error).__name__,
                "message": str(error),
                "environment_steps": self.last_environment_steps,
                "wall_time_seconds": time.time() - self.started_at,
            },
        )
        self._write_status("failed", failure_classification=classification)

    def _write_status(self, state: str, **extra: Any) -> None:
        atomic_write_json(
            self.status_path,
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "state": state,
                "updated_at_unix": time.time(),
                "environment_steps": self.last_environment_steps,
                **extra,
            },
        )


def classify_failure(error: BaseException) -> str:
    message = str(error).lower()
    if isinstance(error, FloatingPointError) or "nan" in message or "non-finite" in message:
        return "non_finite"
    if "out of memory" in message or "resource exhausted" in message:
        return "out_of_memory"
    if "mujoco" in message or "warp" in message or "jax" in message:
        return "environment_failure"
    return "training_failure"
