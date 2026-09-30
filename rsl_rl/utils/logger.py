# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib
import statistics
import time
import torch
from collections import deque

import rsl_rl


class Logger:
    """Logger to save the learning metrics to different logging services."""

    def __init__(
        self,
        log_dir: str | None,
        cfg: dict,
        env_cfg: dict | object,
        num_envs: int,
        is_distributed: bool,
        gpu_world_size: int,
        gpu_global_rank: int,
        device: str,
        run_semantics: dict | None = None,
        environment_steps_per_policy_step: int = 1,
    ) -> None:
        self.log_dir = log_dir
        self.cfg = cfg
        self.run_semantics = run_semantics
        self.num_envs = num_envs
        self.gpu_world_size = gpu_world_size
        self.environment_steps_per_policy_step = int(environment_steps_per_policy_step)
        if self.environment_steps_per_policy_step < 1:
            raise ValueError("environment_steps_per_policy_step must be positive.")
        self.device = device
        self.git_status_repos = [rsl_rl.__file__]
        self.tot_timesteps = 0
        self.tot_time = 0
        self.logger_type = self.cfg.get("logger", "tensorboard")
        self.logger_type = self.logger_type.lower()

        # Create buffers
        self.ep_extras = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Create RND buffers
        if self.cfg["algorithm"]["rnd_cfg"]:
            self.erewbuffer = deque(maxlen=100)
            self.irewbuffer = deque(maxlen=100)
            self.cur_ereward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.cur_ireward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = is_distributed and gpu_global_rank != 0

        # Initialize the writer
        self._prepare_logging_writer()

        # Log code state
        self._store_code_state()


    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        intrinsic_rewards: torch.Tensor | None = None,
    ) -> None:
        """Add metrics from the environment step to the buffers."""
        if self.log_dir is not None:
            if "episode" in extras:
                self.ep_extras.append(extras["episode"])
            elif "log" in extras:
                self.ep_extras.append(extras["log"])

            # Update rewards and episode length
            if intrinsic_rewards is not None:
                self.cur_ereward_sum += rewards
                self.cur_ireward_sum += intrinsic_rewards
                self.cur_reward_sum += rewards + intrinsic_rewards
            else:
                self.cur_reward_sum += rewards
            action_chunk_steps = extras.get("action_chunk_steps")
            if action_chunk_steps is None:
                self.cur_episode_length += 1
            else:
                step_counts = torch.as_tensor(
                    action_chunk_steps,
                    device=self.device,
                    dtype=self.cur_episode_length.dtype,
                ).reshape_as(self.cur_episode_length)
                self.cur_episode_length += step_counts

            # Clear data for completed episodes
            new_ids = (dones > 0).nonzero(as_tuple=False)
            self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0
            if intrinsic_rewards is not None:
                self.erewbuffer.extend(self.cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.irewbuffer.extend(self.cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.cur_ereward_sum[new_ids] = 0
                self.cur_ireward_sum[new_ids] = 0

    @staticmethod
    def _loss_metric_tag(key: str) -> str:
        if "/" in key:
            return key
        return f"Loss/{key}"

    @staticmethod
    def _loss_metric_console_label(key: str) -> str:
        if "/" in key:
            return key
        return f"Mean {key} loss"

    def uses_percent_log_interval(self) -> bool:
        return False

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        reward_norm_metrics: dict[str, float] | None = None,
        environment_steps: int | None = None,
        max_environment_steps: int | None = None,
        emit: bool = True,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        """Log the training metrics to the logging service and print them to the console."""
        if self.log_dir is not None and not self.disable_logs:
            collection_size = (
                self.cfg["num_steps_per_env"]
                * self.num_envs
                * self.gpu_world_size
                * getattr(self, "environment_steps_per_policy_step", 1)
            )
            iteration_time = collect_time + learn_time
            self.tot_timesteps += collection_size
            self.tot_time += iteration_time

            if not emit:
                return

            # Log episode extras
            extras_string = ""
            if self.ep_extras:
                # Iterate over all keys in the episode info dictionary
                for key in self.ep_extras[0]:
                    infotensor = torch.tensor([], device=self.device)
                    # Iterate over all steps
                    for ep_info in self.ep_extras:
                        # Handle missing, scalar, and zero dimensional tensors
                        if key not in ep_info:
                            continue
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    if "/" in key:
                        self.writer.add_scalar(key, value, it)
                        extras_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar("Episode/" + key, value, it)
                        extras_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

            # Log losses
            for key, value in loss_dict.items():
                self.writer.add_scalar(self._loss_metric_tag(key), value, it)
            self.writer.add_scalar("Loss/learning_rate", learning_rate, it)

            reported_environment_steps = (
                self.tot_timesteps
                if environment_steps is None
                else int(environment_steps)
            )
            self.writer.add_scalar(
                "Train/environment_steps",
                reported_environment_steps,
                it,
            )
            if max_environment_steps is not None:
                remaining_environment_steps = max(
                    int(max_environment_steps) - reported_environment_steps,
                    0,
                )
                self.writer.add_scalar(
                    "Train/environment_steps_remaining",
                    remaining_environment_steps,
                    it,
                )
                self.writer.add_scalar(
                    "Train/progress_fraction",
                    min(
                        reported_environment_steps
                        / max(int(max_environment_steps), 1),
                        1.0,
                    ),
                    it,
                )

            # Log reward-normalization diagnostics. These are separate from raw reward curves.
            if reward_norm_metrics:
                for key, value in reward_norm_metrics.items():
                    self.writer.add_scalar(f"RewardNorm/{key}", value, it)

            # Log noise std
            std_metric = (
                "Policy/action_index_std"
                if (self.run_semantics or {}).get("policy_distribution_type") == "categorical"
                else "Policy/mean_noise_std"
            )
            self.writer.add_scalar(std_metric, action_std.mean().item(), it)

            # Log performance
            fps = int(collection_size / (collect_time + learn_time))
            self.writer.add_scalar("Perf/total_fps", fps, it)
            self.writer.add_scalar("Perf/collection_time", collect_time, it)
            self.writer.add_scalar("Perf/learning_time", learn_time, it)

            # Log rewards and episode length
            if len(self.rewbuffer) > 0:
                if self.cfg["algorithm"]["rnd_cfg"]:
                    self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
                    self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
                    self.writer.add_scalar("Rnd/weight", rnd_weight, it)
                self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)
                if True:
                    self.writer.add_scalar(
                        "Train/mean_reward/time", statistics.mean(self.rewbuffer), int(self.tot_time)
                    )
                    self.writer.add_scalar(
                        "Train/mean_episode_length/time", statistics.mean(self.lenbuffer), int(self.tot_time)
                    )

            # WandbSummaryWriter batches all scalars above so the W&B internal
            # step advances exactly once per emitted training row.
            commit = getattr(self.writer, "commit", None)
            if callable(commit):
                commit()

            # Print to console
            log_string = f"""{"#" * width}\n"""
            log_string += f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""

            # Print run name if provided
            run_name = self.cfg.get("run_name")
            log_string += f"""{"Run name:":>{pad}} {run_name}\n""" if run_name else ""

            # Print performance
            log_string += (
                f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
                f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
                f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
                f"""{"Learning time:":>{pad}} {learn_time:.3f}s \n"""
            )

            # Print losses
            for key, value in loss_dict.items():
                log_string += f"""{f"{self._loss_metric_console_label(key)}:":>{pad}} {value:.4f}\n"""

            # Print compact reward normalization diagnostics
            if reward_norm_metrics:
                log_string += f"""{"Reward norm std:":>{pad}} {reward_norm_metrics["std"]:.4f}\n"""
                log_string += (
                    f"""{"Norm reward batch mean:":>{pad}} """
                    f"""{reward_norm_metrics["normalized_reward_batch_mean"]:.4f}\n"""
                )

            # Print rewards and episode length
            if len(self.rewbuffer) > 0:
                if self.cfg["algorithm"]["rnd_cfg"]:
                    log_string += f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(self.erewbuffer):.2f}\n"""
                    log_string += f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(self.irewbuffer):.2f}\n"""
                log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
                log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

            # Print noise std
            std_label = (
                "Action index std:"
                if (self.run_semantics or {}).get("policy_distribution_type") == "categorical"
                else "Mean action noise std:"
            )
            log_string += f"""{std_label:>{pad}} {action_std.mean().item():.2f}\n"""

            # Print episode extras
            if not print_minimal:
                log_string += extras_string

            # Print footer
            done_it = it + 1 - start_it
            remaining_it = total_it - start_it - done_it
            eta = self.tot_time / done_it * remaining_it
            log_string += (
                f"""{"-" * width}\n"""
                f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
                f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
                f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
            )
            print(log_string)

            # Clear extras buffer
            self.ep_extras.clear()

    def save_model(self, path: str, it: int) -> None:
        """The training runner saves checkpoints directly to the local output directory."""
        return None

    def _prepare_logging_writer(self) -> None:
        """Record training metrics locally with TensorBoard."""
        if self.log_dir is not None and not self.disable_logs:
            if self.logger_type != "tensorboard":
                raise ValueError("The paper release uses local TensorBoard logging.")
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            self.writer = None

    def _store_code_state(self) -> None:
        """Store the current git diff of the code repositories involved in the experiment."""
        if self.log_dir is not None and not self.disable_logs:
            try:
                import git
            except ImportError:
                print("Git executable is unavailable. Skipping code-state diff.")
                return
            git_log_dir = os.path.join(self.log_dir, "git")
            os.makedirs(git_log_dir, exist_ok=True)
            file_paths = []
            # Iterate over all repositories to log
            for repository_file_path in self.git_status_repos:
                try:
                    repo = git.Repo(repository_file_path, search_parent_directories=True)
                    t = repo.head.commit.tree
                except Exception:
                    print(f"Could not find git repository in {repository_file_path}. Skipping.")
                    continue
                # Get the name of the repository
                repo_name = pathlib.Path(repo.working_dir).name
                diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
                # Check if the diff file already exists
                if os.path.isfile(diff_file_name):
                    continue
                # Write the diff file
                print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
                with open(diff_file_name, "x", encoding="utf-8") as f:
                    content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
                    f.write(content)
                # Retain the path of the saved local code-state record
                file_paths.append(diff_file_name)
