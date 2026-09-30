from __future__ import annotations

import pathlib
from rsl_rl.runners import OnPolicyRunner

def load_train_cfg(config_path: pathlib.Path) -> dict:
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "runner" not in cfg:
        raise ValueError(f"Configuration at '{config_path}' must contain a top-level 'runner' key.")
    return cfg["runner"]


def resolve_log_dir(log_dir_arg: str, train_cfg: dict) -> str | None:
    normalized = log_dir_arg.lower()
    if normalized == "none":
        return None
    if normalized == "auto":
        experiment_name = str(train_cfg.get("experiment_name", "rsl_rl_run"))
        return str(pathlib.Path("logs") / experiment_name)
    return str(pathlib.Path(log_dir_arg))


def apply_no_log_checkpoint_dir(runner: OnPolicyRunner) -> None:
    checkpoint_dir = pathlib.Path(".ferpo_no_log_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    runner.logger.log_dir = str(checkpoint_dir)
    runner.logger.log = lambda *args, **kwargs: None


def update_run_metadata(train_cfg: dict, env_tag: str, algo_tag: str) -> None:
    name = str(train_cfg.get("run_name", "")).strip()
    train_cfg["run_name"] = name or f"{env_tag}_{algo_tag}_s{train_cfg.get('seed', 1)}"



def build_env(env_backend: str, train_cfg: dict, device: str, env_id=None, env_factory=None):
    """Construct one of the two simulator backends used by the paper."""
    env_cfg = train_cfg["env"]
    seed = int(train_cfg.get("seed", 1))
    backend = env_cfg["kind"] if env_backend == "auto" else env_backend
    if backend == "mujoco_playground":
        from examples.mujoco_playground_adapter import make_env
        result = make_env(env_cfg=env_cfg, seed=seed, device=device)
        if isinstance(result, tuple):
            return result[0], result[1], backend
        return result, getattr(result, "close", lambda: None), backend
    if backend == "maniskill3":
        from examples.maniskill3_adapter import make_env
        env = make_env(env_cfg=env_cfg, seed=seed, device=device)
        return env, env.close, backend
    raise ValueError(f"Unsupported paper environment backend: {backend}")
