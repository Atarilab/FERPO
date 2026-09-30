# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import ActorQ
from rsl_rl.networks import DiscountedRewardStdNormalizer, TanhNormal
from rsl_rl.storage import RolloutStorage

from .config_validation import validate_critic_reward_normalization
from .gaussian_kl import diagonal_gaussian_kl
from .timeout_bootstrap import (
    clone_policy_hidden_states,
    restore_policy_hidden_states,
    terminal_observations,
    validate_time_outs,
)


class REPPO:
    """Relative Entropy Pathwise Policy Optimization algorithm ()."""

    policy: ActorQ
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorQ,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        learning_rate: float = 0.001,
        temperature_learning_rate: float | None = None,
        optimizer_class: str = "adam",
        optimizer_weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        desired_kl: float = 0.01,
        target_entropy: float = -1.0,
        use_tanh_entropy: bool = True,
        tanh_entropy_num_samples: int = 1,
        interleave_actor_critic_updates: bool = True,
        critic_loss_type: str = "hl_gauss",
        aux_loss_mult: float = 0.0,
        mask_critic_loss_on_truncation: bool = True,
        auto_value_range_tuning: bool = False,
        value_range_update_interval: int = 1,
        value_range_ema_alpha: float = 0.05,
        value_range_std_multiplier: float = 3.0,
        value_range_min_half_span: float = 200.0,
        normalize_rewards: bool = False,
        reward_norm_use_ema: bool = True,
        reward_norm_ema_alpha: float = 0.01,
        reward_norm_gamma: float | None = None,
        reward_norm_eps: float = 1.0e-8,
        reward_norm_clip: float | None = None,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        if rnd_cfg or symmetry_cfg:
            raise ValueError("The FERPO paper configurations do not use RND or symmetry augmentation.")
        self.rnd = None
        self.rnd_optimizer = None
        self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # old policy (for KL computation)
        self.old_policy = copy.deepcopy(self.policy)
        self.old_policy.to(self.device)
        self.old_policy.eval()

        # The public REPPO Torch implementation uses AdamW. Keep Adam as the
        # default for backward compatibility with existing RSL-RL configs.
        optimizer_class = str(optimizer_class).lower()
        optimizer_types = {"adam": optim.Adam, "adamw": optim.AdamW}
        if optimizer_class not in optimizer_types:
            raise ValueError(
                f"Unknown optimizer_class {optimizer_class!r}; choose one of {sorted(optimizer_types)}."
            )
        self.optimizer = optimizer_types[optimizer_class](
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=float(optimizer_weight_decay),
        )

        self.temperature_learning_rate = (
            float(learning_rate)
            if temperature_learning_rate is None
            else float(temperature_learning_rate)
        )

        if self.temperature_learning_rate <= 0.0:
            raise ValueError("temperature_learning_rate must be positive.")

        temperature_parameter = self.policy.log_alpha_temp

        # Find the optimizer group currently containing log_alpha_temp.
        temperature_group = None

        for group in self.optimizer.param_groups:
            if any(parameter is temperature_parameter for parameter in group["params"]):
                temperature_group = group
                break

        if temperature_group is None:
            raise RuntimeError(
                "Could not find policy.log_alpha_temp in the optimizer."
            )

        # Remove alpha from the main actor/critic/dual parameter group.
        temperature_group["params"] = [
            parameter
            for parameter in temperature_group["params"]
            if parameter is not temperature_parameter
        ]

        # Copy all of the optimizer's existing settings so that only LR changes.
        new_group = {
            key: value
            for key, value in temperature_group.items()
            if key != "params"
        }

        new_group["params"] = [temperature_parameter]
        new_group["lr"] = self.temperature_learning_rate
        new_group["name"] = "temperature"

        if "initial_lr" in new_group:
            new_group["initial_lr"] = self.temperature_learning_rate

        self.optimizer.add_param_group(new_group)

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # REPPO parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.desired_kl = desired_kl
        self.target_entropy = target_entropy * self.policy.num_actions
        self.learning_rate = learning_rate
        if tanh_entropy_num_samples < 1:
            raise ValueError(f"`tanh_entropy_num_samples` has to be >= 1, got {tanh_entropy_num_samples}.")
        self.use_tanh_entropy = use_tanh_entropy
        self.tanh_entropy_num_samples = tanh_entropy_num_samples
        self.interleave_actor_critic_updates = interleave_actor_critic_updates
        valid_critic_loss_types = {"hl_gauss", "mse"}
        if critic_loss_type not in valid_critic_loss_types:
            raise ValueError(
                f"Unknown critic loss type '{critic_loss_type}'. "
                f"Choose one of {sorted(valid_critic_loss_types)}."
            )
        self.critic_loss_type = critic_loss_type
        self.aux_loss_mult = float(aux_loss_mult)
        self.mask_critic_loss_on_truncation = bool(mask_critic_loss_on_truncation)
        if self.aux_loss_mult < 0.0:
            raise ValueError(f"`aux_loss_mult` must be non-negative, got {aux_loss_mult}.")
        validate_critic_reward_normalization(
            self.critic_loss_type,
            bool(normalize_rewards),
        )

        # Optional critic support auto-tuning for distributional value targets.
        if value_range_update_interval < 1:
            raise ValueError(
                f"`value_range_update_interval` has to be >= 1, got {value_range_update_interval}."
            )
        if not (0.0 < value_range_ema_alpha <= 1.0):
            raise ValueError(f"`value_range_ema_alpha` has to be in (0, 1], got {value_range_ema_alpha}.")
        if value_range_std_multiplier <= 0.0:
            raise ValueError(
                f"`value_range_std_multiplier` has to be > 0, got {value_range_std_multiplier}."
            )
        if value_range_min_half_span <= 0.0:
            raise ValueError(
                f"`value_range_min_half_span` has to be > 0, got {value_range_min_half_span}."
            )

        self.auto_value_range_tuning = auto_value_range_tuning
        self.value_range_update_interval = value_range_update_interval
        self.value_range_ema_alpha = value_range_ema_alpha
        self.value_range_std_multiplier = value_range_std_multiplier
        self.value_range_min_half_span = value_range_min_half_span
        self._value_range_update_count = 0
        self._value_range_stats_initialized = False
        self._value_range_ema_mean = 0.0
        self._value_range_ema_std = 0.0
        self.current_critic_vmin = float(self.policy.vmin)
        self.current_critic_vmax = float(self.policy.vmax)
        self.current_critic_value_mean = 0.5 * (self.current_critic_vmin + self.current_critic_vmax)
        self.current_critic_value_std = 0.0
        self.current_critic_half_span = 0.5 * (self.current_critic_vmax - self.current_critic_vmin)
        self.reward_normalizer = None
        if normalize_rewards:
            normalizer_gamma = self.gamma if reward_norm_gamma is None else reward_norm_gamma
            self.reward_normalizer = DiscountedRewardStdNormalizer(
                num_envs=self.storage.num_envs,
                gamma=normalizer_gamma,
                eps=reward_norm_eps,
                use_ema=reward_norm_use_ema,
                ema_alpha=reward_norm_ema_alpha,
                clip=reward_norm_clip,
                device=self.device,
            )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        environment_actions, critic_actions, actions_log_prob = self._sample_policy_action(
            self.policy,
            obs,
            diagnostic_context="rollout action sampling",
        )
        self.transition.actions = critic_actions.detach()
        self.transition.values = self.policy.evaluate(obs, self.transition.actions).detach()
        self.transition.actions_log_prob = actions_log_prob.detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs
        return environment_actions.detach()

    def _sample_policy_action(
        self,
        policy,
        obs: TensorDict,
        *,
        diagnostic_context: str = "policy action sampling",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample the environment input, critic action, and corresponding log probability."""
        actions = self._act_policy_with_diagnostics(
            policy,
            obs,
            context=diagnostic_context,
        )
        return actions, actions, policy.get_actions_log_prob(actions)

    @staticmethod
    def _act_policy_with_diagnostics(
        policy,
        obs: TensorDict,
        *,
        context: str,
    ) -> torch.Tensor:
        try:
            return policy.act(obs)
        except (FloatingPointError, RuntimeError, ValueError) as error:
            message = str(error).lower()
            if "out of memory" in message or "resource exhausted" in message:
                raise
            diagnose = getattr(policy, "diagnose_action_sampling_failure", None)
            if callable(diagnose):
                diagnose(obs, context=context, error=error)
            raise

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        dones = dones.to(self.device).bool()
        time_outs = extras["time_outs"].to(self.device).bool().view_as(dones)
        validate_time_outs(dones, time_outs)

        # Update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        self.transition.rewards = self._normalize_rewards(rewards, dones).clone()
        self.transition.dones = dones
        self.transition.truncations = time_outs

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        self.transition.soft_rewards = self.transition.rewards.clone()
        self.transition.timeout_bootstrap_values = self._compute_timeout_bootstrap_values(time_outs, extras)

        # REPPO auxiliary target: encode (s_{t+1}, a_{t+1}) with the current
        # feature encoder and stop gradients through the stored target.
        if self.aux_loss_mult > 0.0:
            if self.policy.is_recurrent or not hasattr(self.policy, "critic_features"):
                raise NotImplementedError(
                    "REPPO auxiliary critic loss currently requires the feed-forward ActorQ policy."
                )
            with torch.no_grad():
                # Vector environments return the auto-reset observation for
                # completed episodes. REPPO instead constructs the auxiliary
                # target from the true final observation on those transitions.
                # Keep the ordinary next observation for continuing episodes.
                next_obs_for_aux = obs
                if torch.any(time_outs):
                    final_obs = terminal_observations(time_outs, extras, self.device)
                    timeout_mask = time_outs.reshape(-1)
                    next_obs_for_aux = obs.clone()
                    next_obs_for_aux[timeout_mask] = final_obs[timeout_mask]

                _, next_actions, _ = self._sample_policy_action(
                    self.policy,
                    next_obs_for_aux,
                    diagnostic_context="auxiliary next-feature target",
                )
                self.transition.next_embeddings = self.policy.critic_features(
                    next_obs_for_aux, next_actions
                ).detach()

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def _normalize_rewards(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        rewards = rewards.to(self.device)
        if self.reward_normalizer is None:
            return rewards
        return self.reward_normalizer(rewards, dones)

    def train_mode(self) -> None:
        if self.reward_normalizer is not None:
            self.reward_normalizer.train()

    def eval_mode(self) -> None:
        if self.reward_normalizer is not None:
            self.reward_normalizer.eval()

    def reward_normalizer_state_dict(self) -> dict[str, torch.Tensor] | None:
        if self.reward_normalizer is None:
            return None
        return self.reward_normalizer.state_dict()

    def reward_normalizer_metrics(self) -> dict[str, float]:
        if self.reward_normalizer is None:
            return {}
        return self.reward_normalizer.metrics()

    def load_reward_normalizer_state_dict(self, state_dict: dict[str, torch.Tensor] | None) -> None:
        if self.reward_normalizer is not None and state_dict is not None:
            self.reward_normalizer.load_checkpoint_state_dict(state_dict)


    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        alpha = self.policy.alpha_temp.detach()

        # Value and log-probability at the final post-rollout state.
        _, last_action, last_action_log_prob = self._sample_policy_action(
            self.policy,
            obs,
            diagnostic_context="final-observation bootstrap",
        )

        last_action = last_action.detach()
        last_action_log_prob = last_action_log_prob.detach().view(-1, 1)

        last_values = (
            self.policy.evaluate(obs, last_action)
            .detach()
            .view(-1, 1)
        )

        # IMPORTANT:
        # The recursive value starts from the ordinary Q value, not the
        # soft value. Entropy is incorporated through the soft reward below.
        recurr_value = last_values

        for step in reversed(range(st.num_transitions_per_env)):
            time_outs = st.truncations[step].bool()
            true_terminals = st.dones[step].bool() & ~time_outs

            # Q(s_{t+1}, a_{t+1}) and log pi(a_{t+1}|s_{t+1})
            if step == st.num_transitions_per_env - 1:
                next_values = last_values
                next_log_prob = last_action_log_prob
            else:
                next_values = st.values[step + 1]
                next_log_prob = st.actions_log_prob[step + 1]

            # REPPO-style soft reward:
            #
            # r_t^soft =
            #     r_t - gamma * alpha * log pi(a_{t+1}|s_{t+1})
            #
            # Notice that the entropy term is OUTSIDE the
            # (1 - lambda) / lambda mixture.
            soft_reward = (
                st.rewards[step]
                - self.gamma * alpha * next_log_prob
            )

            # Ordinary non-terminal TD(lambda) recursion.
            normal_return = soft_reward + self.gamma * (
                (1.0 - self.lam) * next_values
                + self.lam * recurr_value
            )

            # timeout_bootstrap_values already stores
            #
            # Q(s_terminal, a) - alpha * log pi(a|s_terminal)
            #
            # so do NOT subtract the entropy term a second time here.
            timeout_return = (
                st.rewards[step]
                + self.gamma * st.timeout_bootstrap_values[step]
            )

            # A true terminal has no bootstrap from the next state.
            terminal_return = st.rewards[step]

            recurr_value = torch.where(
                time_outs,
                timeout_return,
                torch.where(
                    true_terminals,
                    terminal_return,
                    normal_return,
                ),
            )

            st.returns[step] = recurr_value
    '''
    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        # Compute soft value for the final post-rollout observation.
        _, last_action, last_action_log_prob = self._sample_policy_action(
            self.policy,
            obs,
            diagnostic_context="final-observation bootstrap",
        )
        last_action = last_action.detach()
        last_action_log_prob = last_action_log_prob.detach().view(-1, 1)
        last_values = self.policy.evaluate(obs, last_action).detach().view(-1, 1)
        last_soft_values = last_values - self.policy.alpha_temp.detach() * last_action_log_prob

        recurr_value = last_soft_values
        for step in reversed(range(st.num_transitions_per_env)):
            time_outs = st.truncations[step].bool()
            true_terminals = st.dones[step].bool() & ~time_outs
            if step == st.num_transitions_per_env - 1:
                next_soft_values = last_soft_values
            else:
                next_soft_values = (
                    st.values[step + 1] - self.policy.alpha_temp.detach() * st.actions_log_prob[step + 1]
                )
            next_soft_values = torch.where(time_outs, st.timeout_bootstrap_values[step], next_soft_values)
            recurrent_soft_values = torch.where(time_outs, st.timeout_bootstrap_values[step], recurr_value)
            has_bootstrap = 1.0 - true_terminals.float()
            recurr_value = st.rewards[step] + has_bootstrap * self.gamma * (
                (1.0 - self.lam) * next_soft_values + self.lam * recurrent_soft_values
            )
            st.returns[step] = recurr_value
    '''
    def _compute_timeout_bootstrap_values(
        self, time_outs: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        timeout_bootstrap_values = torch.zeros_like(self.transition.values)
        if not torch.any(time_outs):
            return timeout_bootstrap_values

        terminal_obs = terminal_observations(time_outs, extras, self.device)
        hidden_states = clone_policy_hidden_states(self.policy)
        _, terminal_actions, terminal_log_prob = self._sample_policy_action(
            self.policy,
            terminal_obs,
            diagnostic_context="timeout terminal-observation bootstrap",
        )
        terminal_actions = terminal_actions.detach()
        terminal_log_prob = terminal_log_prob.detach().view_as(timeout_bootstrap_values)
        terminal_values = (
            self.policy.evaluate(terminal_obs, terminal_actions).detach().view_as(timeout_bootstrap_values)
        )
        restore_policy_hidden_states(self.policy, hidden_states)
        terminal_soft_values = terminal_values - self.policy.alpha_temp.detach() * terminal_log_prob
        timeout_bootstrap_values[time_outs.view(-1)] = terminal_soft_values[time_outs.view(-1)]
        return timeout_bootstrap_values

    def _estimate_policy_entropy(
        self,
        distribution,
        use_distribution_entropy_when_legacy: bool = False,
    ) -> torch.Tensor:
        if not self.use_tanh_entropy:
            if use_distribution_entropy_when_legacy:
                return distribution.entropy().sum(dim=-1)
            if isinstance(distribution, TanhNormal):
                _, log_prob = distribution.rsample_and_log_prob()
                return -log_prob.sum(dim=-1)
            samples = distribution.rsample()
            return -distribution.log_prob(samples).sum(dim=-1)

        if isinstance(distribution, TanhNormal):
            entropy_distribution = distribution
        elif isinstance(distribution, Normal):
            entropy_distribution = TanhNormal(distribution.loc, distribution.scale)
        else:
            raise ValueError(f"Unsupported policy distribution for tanh entropy: {type(distribution).__name__}.")

        sample_shape = (
            torch.Size()
            if self.tanh_entropy_num_samples == 1
            else torch.Size([self.tanh_entropy_num_samples])
        )
        _, log_prob = entropy_distribution.rsample_and_log_prob(sample_shape)
        entropy = -log_prob.sum(dim=-1)
        if self.tanh_entropy_num_samples == 1:
            return entropy
        return entropy.mean(dim=0)

    def _policy_act(self, policy, obs_batch, hidden_states_batch, masks_batch) -> None:
        if policy.is_recurrent:
            policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
        else:
            policy.act(obs_batch)

    def _policy_evaluate(
        self,
        obs_batch,
        actions_batch,
        hidden_states_batch,
        masks_batch,
        return_logits: bool = False,
    ) -> torch.Tensor:
        if self.policy.is_recurrent:
            return self.policy.evaluate(
                obs_batch,
                actions_batch,
                masks=masks_batch,
                hidden_state=hidden_states_batch[1],
                return_logits=return_logits,
            )
        return self.policy.evaluate(obs_batch, actions_batch, return_logits=return_logits)

    def _evaluate_sampled_action_values(
        self,
        obs_batch: TensorDict,
        sampled_actions: torch.Tensor,
        hidden_states_batch,
        masks_batch,
    ) -> torch.Tensor:
        if self.policy.is_recurrent:
            q_values = []
            for sampled_action in sampled_actions.unbind(0):
                q_pred = self._policy_evaluate(obs_batch, sampled_action, hidden_states_batch, masks_batch)
                q_values.append(q_pred.reshape(sampled_action.shape[:-1]))
            return torch.stack(q_values, dim=0)

        sample_count = sampled_actions.shape[0]
        obs_batch_size = tuple(obs_batch.batch_size)
        batch_ndim = len(obs_batch_size)
        flat_batch_size = sample_count * math.prod(obs_batch_size)
        action_shape = sampled_actions.shape[1 + batch_ndim :]
        flat_actions = sampled_actions.reshape(flat_batch_size, *action_shape)
        flat_obs = TensorDict(
            {
                key: value.unsqueeze(0)
                .expand(sample_count, *value.shape)
                .reshape(flat_batch_size, *value.shape[batch_ndim:])
                for key, value in obs_batch.items()
            },
            batch_size=[flat_batch_size],
            device=obs_batch.device,
        )
        q_pred = self._policy_evaluate(flat_obs, flat_actions, hidden_states_batch, masks_batch)
        # Scalar critics may return vectors or columns; preserve every state axis.
        return q_pred.reshape(sample_count, *obs_batch_size)

    def _lagrange_metrics(self) -> dict[str, float]:
        return {
            "Lagrange/temperature": self.policy.alpha_temp.item(),
            "Lagrange/alpha_kl": self.policy.alpha_kl.item(),
        }

    def _mini_batch_generator(self):
        if self.policy.is_recurrent:
            return self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        return self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs, with_indices=True
        )

    def _build_minibatch(self, batch: tuple) -> dict:
        base_batch = batch[:11]
        (
            obs_batch,
            actions_batch,
            _,
            _,
            returns_batch,
            truncations_batch,
            _,
            old_mean_batch,
            old_std_batch,
            hidden_states_batch,
            masks_batch,
        ) = base_batch
        minibatch = {
            "obs_batch": obs_batch,
            "actions_batch": actions_batch,
            "returns_batch": returns_batch,
            "truncations_batch": truncations_batch,
            "hidden_states_batch": hidden_states_batch,
            "masks_batch": masks_batch,
            "old_mean": old_mean_batch,
            "old_std": old_std_batch,
        }
        if len(batch) == 13:
            time_indices_batch, env_indices_batch = batch[11:]
            minibatch["time_indices_batch"] = time_indices_batch
            minibatch["env_indices_batch"] = env_indices_batch
            minibatch["rewards_batch"] = self.storage.rewards[
                time_indices_batch, env_indices_batch
            ]
            minibatch["dones_batch"] = self.storage.dones[
                time_indices_batch, env_indices_batch
            ]
            if self.storage.next_embeddings is not None:
                minibatch["next_embeddings_batch"] = self.storage.next_embeddings[
                    time_indices_batch, env_indices_batch
                ]
        return minibatch

    def _staged_critic_updates(self):
        """Yield critic metrics before the staged actor phase begins."""
        for batch in self._mini_batch_generator():
            yield self.update_critic(self._build_minibatch(batch))

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_actor_metrics: dict[str, float] = {}
        mean_critic_metrics: dict[str, float] = {}
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        with torch.no_grad():
            self.old_policy.load_state_dict(self.policy.state_dict())
            self._maybe_update_value_range()

        if self.interleave_actor_critic_updates:
            for batch in self._mini_batch_generator():
                minibatch = self._build_minibatch(batch)
                critic_metrics = self.update_critic(minibatch)
                actor_metrics = self.update_actor(minibatch)

                mean_value_loss += critic_metrics["value_loss"]
                for key, value in critic_metrics.items():
                    if key != "value_loss":
                        mean_critic_metrics[key] = mean_critic_metrics.get(key, 0.0) + float(value)
                mean_entropy += actor_metrics["entropy"]
                mean_surrogate_loss += actor_metrics["actor_loss"]
                for key, value in actor_metrics.items():
                    if key in {"actor_loss", "entropy"}:
                        continue
                    if key not in mean_actor_metrics:
                        mean_actor_metrics[key] = 0.0
                    mean_actor_metrics[key] += float(value)
        else:
            critic_updates = 0
            for critic_metrics in self._staged_critic_updates():
                critic_updates += 1
                mean_value_loss += critic_metrics["value_loss"]
                for key, value in critic_metrics.items():
                    if key != "value_loss":
                        mean_critic_metrics[key] = mean_critic_metrics.get(key, 0.0) + float(value)

            for batch in self._mini_batch_generator():
                minibatch = self._build_minibatch(batch)
                actor_metrics = self.update_actor(minibatch)
                mean_entropy += actor_metrics["entropy"]
                mean_surrogate_loss += actor_metrics["actor_loss"]
                for key, value in actor_metrics.items():
                    if key in {"actor_loss", "entropy"}:
                        continue
                    if key not in mean_actor_metrics:
                        mean_actor_metrics[key] = 0.0
                    mean_actor_metrics[key] += float(value)

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        num_critic_updates = num_updates if self.interleave_actor_critic_updates else critic_updates
        mean_value_loss /= num_critic_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        for key in mean_actor_metrics:
            mean_actor_metrics[key] /= num_updates
        for key in mean_critic_metrics:
            mean_critic_metrics[key] /= num_critic_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "Entropy/policy": mean_entropy,
            "Entropy/target": self.target_entropy,
        }
        for key, value in mean_actor_metrics.items():
            if "/" in key:
                loss_dict[key] = value
            else:
                loss_dict[f"actor_{key}"] = value
        for key, value in mean_critic_metrics.items():
            loss_dict[f"Critic/{key}"] = value
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        loss_dict["Critic/vmin"] = self.current_critic_vmin
        loss_dict["Critic/vmax"] = self.current_critic_vmax
        if self.auto_value_range_tuning:
            loss_dict["Critic/value_mean_ema"] = self.current_critic_value_mean
            loss_dict["Critic/value_std_ema"] = self.current_critic_value_std
            loss_dict["Critic/half_span"] = self.current_critic_half_span
        return loss_dict

    def update_actor(self, minibatch: dict) -> dict:
        """Update the actor network.

        Args:
            minibatch: Dictionary containing the minibatch data with keys:
                - obs_batch: Observations
                - actions_batch: Actions
                - returns_batch: Returns
                - truncations_batch: Truncation flags
                - hidden_states_batch: Hidden states (for recurrent policies)
                - masks_batch: Masks (for recurrent policies)

        Returns:
            Dictionary containing actor metrics.
        """
        obs_batch = minibatch["obs_batch"]
        hidden_states_batch = minibatch["hidden_states_batch"]
        masks_batch = minibatch["masks_batch"]

        # Get current actor outputs
        self._policy_act(self.policy, obs_batch, hidden_states_batch, masks_batch)
        predicted_policy = self.policy.distribution
        predicted_actions = predicted_policy.rsample()
        on_policy_values = self._policy_evaluate(obs_batch, predicted_actions, hidden_states_batch, masks_batch)

        # Temperature component
        entropy = self._estimate_policy_entropy(predicted_policy)
        entropy_loss = self.policy.alpha_temp.detach() * entropy.view_as(on_policy_values)
        primary_policy_loss = -(on_policy_values + entropy_loss)

        # Exact consecutive-policy KL. A shared tanh/affine transform leaves
        # the KL unchanged, so transformed Gaussian policies are evaluated
        # through their diagonal Gaussian base distributions.
        with torch.no_grad():
            self._policy_act(self.old_policy, obs_batch, hidden_states_batch, masks_batch)
            old_policy_distribution = self.old_policy.distribution
        # Keep the old distribution frozen while retaining derivatives through
        # the current policy.  The trust-region branch below must be able to
        # move the current policy back toward the rollout policy.
        kl_divergence = diagonal_gaussian_kl(
            old_policy_distribution,
            predicted_policy,
        ).view_as(primary_policy_loss)

        # Clipped Policy Loss
        policy_loss = torch.where(
            (kl_divergence < self.desired_kl).detach(),
            primary_policy_loss,
            self.policy.alpha_kl.detach() * kl_divergence,
        ).mean()

        # Temperature updates
        temp_target_loss = self.policy.alpha_temp * (entropy.mean() - self.target_entropy).detach()
        kl_target_loss = self.policy.alpha_kl * (self.desired_kl - kl_divergence.mean()).detach()

        # Freeze critic parameters to prevent policy loss from updating them
        self._set_critic_grad(False)
        self.optimizer.zero_grad()
        actor_loss = policy_loss + temp_target_loss + kl_target_loss
        actor_loss.backward()
        # Collect gradients from all GPUs
        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Unfreeze critic parameters
        self._set_critic_grad(True)

        return {
            "actor_loss": actor_loss.item(),
            "entropy": entropy.mean().item(),
            "Policy/kl_divergence": kl_divergence.mean().item(),
            "Policy/on_policy_values_mean": on_policy_values.mean().item(),
            **self._lagrange_metrics(),
        }

    def update_critic(self, minibatch: dict) -> dict:
        """Fit the critic to the rollout's fixed regression targets."""
        value_loss, metrics = self._critic_loss_and_metrics(minibatch)
        self.optimizer.zero_grad()
        value_loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return metrics

    def _critic_loss_and_metrics(self, minibatch: dict) -> tuple[torch.Tensor, dict]:
        """Shared training/validation objective, including auxiliary loss and masks."""
        obs_batch = minibatch["obs_batch"]
        actions_batch = minibatch["actions_batch"]
        returns_batch = minibatch["returns_batch"]
        hidden_states_batch = minibatch["hidden_states_batch"]
        masks_batch = minibatch["masks_batch"]
        truncations_batch = minibatch["truncations_batch"].view(-1).bool()

        # VALUE LOSS
        if self.critic_loss_type == "hl_gauss":
            if self.policy.is_recurrent or not hasattr(self.policy, "evaluate_full"):
                raise NotImplementedError(
                    "The exact REPPO critic and auxiliary loss require feed-forward ActorQ."
                )
            value_prediction, value_logits, predicted_features, _ = (
                self.policy.evaluate_full(obs_batch, actions_batch)
            )
            embedded_returns = self.policy.hlgauss_embed(returns_batch.view(-1)).view(value_logits.shape).detach()
            categorical_loss = -(embedded_returns * torch.log_softmax(value_logits, dim=-1)).sum(-1)

            if self.aux_loss_mult > 0.0:
                required = {"next_embeddings_batch"}
                missing = required.difference(minibatch)
                if missing:
                    raise RuntimeError(
                        "Auxiliary critic targets are missing from the rollout minibatch: "
                        + ", ".join(sorted(missing))
                    )
                feature_error = (
                    predicted_features - minibatch["next_embeddings_batch"].detach()
                ).square()
                auxiliary_loss_per_sample = feature_error.mean(dim=-1)
            else:
                auxiliary_loss_per_sample = torch.zeros_like(categorical_loss)

            valid = (
                (~truncations_batch).to(categorical_loss.dtype)
                if self.mask_critic_loss_on_truncation
                else torch.ones_like(categorical_loss)
            )
            value_loss = (
                valid * (categorical_loss + self.aux_loss_mult * auxiliary_loss_per_sample)
            ).mean()
            q_loss = (valid * categorical_loss).mean()
        else:
            value_prediction = self._policy_evaluate(
                obs_batch,
                actions_batch,
                hidden_states_batch,
                masks_batch,
            )
            squared_error = (value_prediction.view_as(returns_batch) - returns_batch).square().view(-1)
            valid = (
                (~truncations_batch).to(squared_error.dtype)
                if self.mask_critic_loss_on_truncation
                else torch.ones_like(squared_error)
            )
            value_loss = (valid * squared_error).mean()
            q_loss = value_loss
            categorical_loss = torch.zeros_like(squared_error)
            auxiliary_loss_per_sample = torch.zeros_like(squared_error)

        # Compute metrics for logging
        value_prediction_error = (value_prediction.view(-1) - returns_batch.view(-1)).abs().mean().item()

        metrics = {
            "value_loss": value_loss.item(),
            "q_loss": q_loss.item(),
            "value_prediction_error": value_prediction_error,
            "categorical_loss": categorical_loss.mean().item(),
            "auxiliary_loss": auxiliary_loss_per_sample.mean().item(),
            "auxiliary_loss_masked": (valid * auxiliary_loss_per_sample).mean().item(),
        }
        if self.critic_loss_type == "hl_gauss":
            decoded_returns = self.policy.hlgauss_decode(torch.log(embedded_returns)).detach()
            metrics["enc_dec_error"] = (decoded_returns.view(-1) - returns_batch.view(-1)).abs().mean().item()
        return value_loss, metrics

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def _maybe_update_value_range(self) -> None:
        """Adapt critic HL-Gauss support from current training-return statistics."""
        if not self.auto_value_range_tuning:
            return
        if not hasattr(self.policy, "set_critic_value_range"):
            return

        self._value_range_update_count += 1
        if self._value_range_update_count % self.value_range_update_interval != 0:
            return

        flat_returns = self.storage.returns.detach().view(-1)
        if flat_returns.numel() == 0:
            return

        batch_mean = float(flat_returns.mean().item())
        batch_std = float(flat_returns.std(unbiased=False).item())

        if not self._value_range_stats_initialized:
            self._value_range_ema_mean = batch_mean
            self._value_range_ema_std = max(batch_std, 1.0e-8)
            self._value_range_stats_initialized = True
        else:
            alpha = self.value_range_ema_alpha
            self._value_range_ema_mean = (1.0 - alpha) * self._value_range_ema_mean + alpha * batch_mean
            self._value_range_ema_std = (1.0 - alpha) * self._value_range_ema_std + alpha * batch_std

        half_span = max(
            self.value_range_std_multiplier * self._value_range_ema_std,
            self.value_range_min_half_span,
        )
        new_vmin = self._value_range_ema_mean - half_span
        new_vmax = self._value_range_ema_mean + half_span
        if not torch.isfinite(torch.tensor([new_vmin, new_vmax])).all() or new_vmax <= new_vmin:
            return

        self.policy.set_critic_value_range(new_vmin, new_vmax)
        self.old_policy.set_critic_value_range(new_vmin, new_vmax)
        self.current_critic_vmin = new_vmin
        self.current_critic_vmax = new_vmax
        self.current_critic_value_mean = self._value_range_ema_mean
        self.current_critic_value_std = self._value_range_ema_std
        self.current_critic_half_span = half_span

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel

    def _set_critic_grad(self, requires_grad: bool) -> None:
        """Enable or disable gradient computation for critic parameters.

        This is used to prevent the policy loss from updating critic parameters
        while still allowing gradients to flow through the critic to the actor.

        Args:
            requires_grad: Whether to enable gradient computation for critic parameters.
        """
        for param in self.policy.critic.parameters():
            param.requires_grad = requires_grad
        for param in self.policy.critic_embedding_layer.parameters():
            param.requires_grad = requires_grad
        if self.policy.is_recurrent:
            for param in self.policy.memory_c.parameters():
                param.requires_grad = requires_grad
