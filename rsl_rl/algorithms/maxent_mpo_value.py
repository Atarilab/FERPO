"""FERPO with saved-state one-step simulation and a soft state-value critic."""

from __future__ import annotations

import torch
from torch.distributions import Normal
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorV
from rsl_rl.networks import TanhNormal

from .maxent_mpo import MaxEntMPO
from .cached_projection import CachedProjection
from .timeout_bootstrap import terminal_observations, validate_time_outs


class MaxEntMPOValue(CachedProjection, MaxEntMPO):
    """Fit V first, then fit the actor to cached r + gamma V(s') candidates.

    V estimates rollout-policy return including entropy at its current state.
    Candidate Q estimates include entropy only through the next-state V.
    The runner's transition-query protocol captures each pre-action state.
    """

    fresh_transition_updates = True

    def __init__(self, policy, storage, *args, candidate_query_batch_size: int | None = None,
                 candidate_collection: str = "update", **kwargs):
        if not isinstance(policy, ActorV) or policy.is_recurrent:
            raise ValueError("MaxEntMPOValue requires the feed-forward ActorV policy.")
        kwargs.setdefault("interleave_actor_critic_updates", False)
        if kwargs["interleave_actor_critic_updates"]:
            raise ValueError("MaxEntMPOValue requires staged critic/actor updates to keep cached V scores fixed.")
        if kwargs.get("normalize_rewards", False) or kwargs.get("rnd_cfg"):
            raise ValueError("MaxEntMPOValue requires raw environment rewards (no reward normalization or RND).")
        if kwargs.get("symmetry_cfg") or kwargs.get("scale_actions", False):
            raise ValueError("MaxEntMPOValue requires unaugmented actions with scaling handled by the environment.")
        if not kwargs.pop("fresh_transition_updates", True):
            raise ValueError("MaxEntMPOValue always requires saved-state transition queries.")
        kwargs.setdefault("mask_critic_loss_on_truncation", False)
        super().__init__(policy, storage, *args, **kwargs)
        self.candidate_query_batch_size = self.num_action_samples if candidate_query_batch_size is None else int(candidate_query_batch_size)
        if self.candidate_query_batch_size < 1:
            raise ValueError("candidate_query_batch_size must be positive.")
        if self.num_learning_epochs < 1 or self.num_mini_batches < 1:
            raise ValueError("MaxEntMPOValue requires positive epoch and minibatch counts.")
        rollout_size = storage.num_envs * storage.num_transitions_per_env
        if rollout_size % self.num_mini_batches != 0:
            raise ValueError("MaxEntMPOValue requires minibatches to cover the complete rollout evenly.")
        self.transition_query_env: VecEnv | None = None
        self.rollout_temperature = self.policy.alpha_temp.detach().clone()
        self._candidate_cache: dict[str, torch.Tensor] | None = None
        self.candidate_transitions_total = 0
        self._candidate_transitions_this_update = 0
        if candidate_collection not in {"update", "rollout"}:
            raise ValueError("candidate_collection must be 'update' or 'rollout'.")
        if candidate_collection == "rollout" and self.q_weight_ess_mode == "statewise":
            raise ValueError("Rollout candidates require a controller supporting proposal-density correction.")
        self.candidate_collection = candidate_collection
        self._rollout_candidates = [None] * storage.num_transitions_per_env
        self._candidate_generator = torch.Generator(device=self.device)
        self._candidate_generator.manual_seed((torch.initial_seed() + 104729) % (2**63))

    def set_transition_query_env(self, env: VecEnv) -> None:
        if not env.supports_transition_queries():
            raise ValueError("MaxEntMPOValue requires an environment supporting non-mutating transition queries.")
        if env.num_envs != self.storage.num_envs:
            raise ValueError("The query environment must match the rollout environment count.")
        self.transition_query_env = env

    def record_transition_query_state(self, query_state: object) -> None:
        if self.transition_query_env is None:
            raise RuntimeError("No saved-state query environment has been attached to MaxEntMPOValue.")
        if self.candidate_collection == "rollout":
            self._collect_candidate_transitions(query_state)
        else:
            self.transition.transition_query_state = self.transition_query_env.offload_transition_query_state(query_state)

    @torch.no_grad()
    def _collect_candidate_transitions(self, query_state: object) -> None:
        """Branch while the simulator state is resident; defer V until after fitting.

        The proposal is the policy distribution used for this rollout action.
        Actor normalization may subsequently change, so retain the actual
        proposal density and correct to the end-of-rollout reference later.
        A separate RNG prevents candidate sampling from advancing rollout RNG.
        """
        if self.storage.step == 0:
            self._candidate_transitions_this_update = 0
        distribution = self.policy.distribution
        if not isinstance(distribution, (Normal, TanhNormal)):
            raise TypeError("Rollout candidates require a Normal or TanhNormal policy.")
        scale = distribution.scale * self.action_proposal_std_scale
        noise = torch.randn((self.num_action_samples, *distribution.loc.shape),
                            device=self.device, dtype=distribution.loc.dtype,
                            generator=self._candidate_generator)
        latent = distribution.loc + scale * noise
        if isinstance(distribution, TanhNormal):
            actions = distribution.action_from_pre_tanh(latent)
            reference_log_prob = distribution.log_prob_from_pre_tanh(latent).sum(-1)
            correction = (distribution.base_dist.log_prob(latent)
                          - Normal(distribution.loc, scale).log_prob(latent)).sum(-1)
            sampled = {"actions": actions, "pre_tanh": latent,
                       "proposal_log_prob": reference_log_prob - correction}
        else:
            sampled = {"actions": latent,
                       "proposal_log_prob": Normal(distribution.loc, scale).log_prob(latent).sum(-1)}
        sampled["results"] = []
        for chunk in sampled["actions"].split(self.candidate_query_batch_size, dim=0):
            result = self.transition_query_env.query_transition_candidates(
                query_state, chunk.to(self.transition_query_env.device))
            count = len(chunk) * self.storage.num_envs
            if result.rewards.numel() != count:
                raise ValueError("Expected one candidate transition per action and environment.")
            sampled["results"].append(result)
            self._candidate_transitions_this_update += count
            self.candidate_transitions_total += count
        self._rollout_candidates[self.storage.step] = sampled

    @torch.no_grad()
    def _build_rollout_candidate_cache(self) -> None:
        if self.storage.step != self.storage.num_transitions_per_env or any(
                row is None for row in self._rollout_candidates):
            raise RuntimeError("Candidate evaluation requires a complete rollout.")
        cache = {}
        for step, row in enumerate(self._rollout_candidates):
            obs = self.storage.observations[step]
            self._policy_act(self.old_policy, obs, None, None)
            reference = self.old_policy.distribution
            sampled = {k: v for k, v in row.items() if k != "results"}
            pre_tanh = sampled.get("pre_tanh")
            sampled["reference_log_prob"] = (
                reference.log_prob_from_pre_tanh(pre_tanh) if pre_tanh is not None
                else reference.log_prob(sampled["actions"])).sum(-1)
            sampled["q_values"] = torch.cat([
                self._candidate_value(result).reshape(-1, self.storage.num_envs)
                for result in row["results"]], dim=0)
            if self._uses_momentum():
                self._policy_act(self.previous_policy, obs, None, None)
                previous = self.previous_policy.distribution
                sampled["older_log_prob"] = (
                    previous.log_prob_from_pre_tanh(pre_tanh) if pre_tanh is not None
                    else previous.log_prob(sampled["actions"])).sum(-1)
            for name, value in sampled.items():
                if name not in cache:
                    cache[name] = value.new_empty((self.storage.num_transitions_per_env, *value.shape))
                cache[name][step].copy_(value)
            self._rollout_candidates[step] = None
        self._candidate_cache = cache

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.storage.step == 0:
            self.rollout_temperature = self.policy.alpha_temp.detach().clone()
        # ActorV ignores the action argument when the parent records V(s).
        return super().act(obs)

    def _compute_timeout_bootstrap_values(self, time_outs, extras):
        values = torch.zeros_like(self.transition.values)
        if torch.any(time_outs):
            final_obs = terminal_observations(time_outs, extras, self.device)
            mask = time_outs.reshape(-1)
            # V already includes future entropy; do not subtract log pi again.
            values[mask] = self.policy.evaluate(final_obs[mask]).detach().view_as(values[mask])
        return values

    def process_env_step(self, obs, rewards, dones, extras):
        dones = dones.to(self.device).bool()
        time_outs = extras["time_outs"].to(self.device).bool().view_as(dones)
        validate_time_outs(dones, time_outs)
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards.to(self.device).detach().clone()
        self.transition.soft_rewards = self.transition.rewards.clone()
        self.transition.dones = dones
        self.transition.truncations = time_outs
        self.transition.timeout_bootstrap_values = self._compute_timeout_bootstrap_values(time_outs, extras)
        if self.aux_loss_mult > 0:
            next_obs = obs
            if torch.any(dones):
                if "terminal_observations" not in extras:
                    raise ValueError("The state-feature auxiliary loss requires final observations for completed episodes.")
                next_obs = obs.clone()
                mask = dones.reshape(-1)
                next_obs[mask] = extras["terminal_observations"].to(self.device)[mask]
            with torch.no_grad():
                self.transition.next_embeddings = self.policy.critic_features(next_obs).detach()
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    @torch.no_grad()
    def compute_returns(self, obs: TensorDict) -> None:
        """Soft V TD(lambda), with current entropy and timeout-aware traces."""
        st = self.storage
        last_values = self.policy.evaluate(obs).detach().view(-1, 1)
        recurrent_value = last_values
        for step in reversed(range(st.num_transitions_per_env)):
            time_outs = st.truncations[step].bool()
            true_terminals = st.dones[step].bool() & ~time_outs
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            soft_reward = st.rewards[step] - self.rollout_temperature * st.actions_log_prob[step]
            normal_return = soft_reward + self.gamma * (
                (1.0 - self.lam) * next_values + self.lam * recurrent_value
            )
            timeout_return = soft_reward + self.gamma * st.timeout_bootstrap_values[step]
            recurrent_value = torch.where(
                time_outs, timeout_return, torch.where(true_terminals, soft_reward, normal_return)
            )
            st.returns[step] = recurrent_value

    @torch.no_grad()
    def _candidate_value(self, result) -> torch.Tensor:
        rewards = result.rewards.to(self.device).view(-1)
        dones = result.dones.to(self.device).bool().view(-1)
        time_outs = result.extras["time_outs"].to(self.device).bool().view(-1)
        validate_time_outs(dones, time_outs)
        next_obs = result.observations.to(self.device).reshape(-1)
        if torch.any(time_outs):
            final_obs = terminal_observations(time_outs, result.extras, self.device).reshape(-1)
            next_obs = next_obs.clone()
            next_obs[time_outs] = final_obs[time_outs]
        bootstrap = torch.zeros_like(rewards)
        valid = ~(dones & ~time_outs)
        if torch.any(valid):
            bootstrap[valid] = self.policy.evaluate(next_obs[valid]).reshape(-1)
        return rewards + self.gamma * bootstrap

    @torch.no_grad()
    def _build_candidate_cache(self) -> None:
        if self.transition_query_env is None:
            raise RuntimeError("No saved-state query environment has been attached to MaxEntMPOValue.")
        if self.candidate_collection == "rollout":
            self._build_rollout_candidate_cache()
            return
        st = self.storage
        if st.step != st.num_transitions_per_env or any(state is None for state in st.transition_query_states):
            raise RuntimeError("Candidate evaluation requires a complete rollout with every pre-action state saved.")
        cache = {}
        for step, query_state in enumerate(st.transition_query_states):
            obs = st.observations[step]
            self._policy_act(self.old_policy, obs, None, None)
            reference = self.old_policy.distribution
            proposal_log_prob = None
            pre_tanh = None
            if self.action_proposal_std_scale != 1.0:
                actions, reference_log_prob, proposal_log_prob, pre_tanh = self._sample_wider_proposal(reference)
            elif isinstance(reference, TanhNormal):
                actions, pre_tanh = reference.sample_with_pre_tanh(torch.Size([self.num_action_samples]))
                reference_log_prob = reference.log_prob_from_pre_tanh(pre_tanh).sum(-1)
            else:
                actions = reference.sample((self.num_action_samples,))
                reference_log_prob = reference.log_prob(actions).sum(-1)
            values = []
            prepared_state = self.transition_query_env.prepare_transition_query_state(query_state)
            try:
                for candidate_actions in actions.split(self.candidate_query_batch_size, dim=0):
                    # Restore this timestep once for all K candidates, preserving
                    # the model's original environment-slot order throughout.
                    result = self.transition_query_env.query_transition_candidates(
                        prepared_state, candidate_actions.to(self.transition_query_env.device)
                    )
                    transitions = len(candidate_actions) * st.num_envs
                    if result.rewards.numel() != transitions:
                        raise ValueError("A candidate query must return one transition per candidate and rollout environment.")
                    values.append(self._candidate_value(result).reshape(len(candidate_actions), st.num_envs))
                    self._candidate_transitions_this_update += transitions
                    self.candidate_transitions_total += transitions
            finally:
                del prepared_state
            # The frozen candidate scores are all actor fitting needs from here.
            st.transition_query_states[step] = None
            sampled = {
                "actions": actions,
                "reference_log_prob": reference_log_prob,
                "q_values": torch.cat(values, dim=0),
            }
            if pre_tanh is not None:
                sampled["pre_tanh"] = pre_tanh
            if proposal_log_prob is not None:
                sampled["proposal_log_prob"] = proposal_log_prob
            if self._uses_momentum():
                self._policy_act(self.previous_policy, obs, None, None)
                previous = self.previous_policy.distribution
                sampled["older_log_prob"] = (
                    previous.log_prob_from_pre_tanh(pre_tanh)
                    if pre_tanh is not None else previous.log_prob(actions)
                ).sum(-1)
            for name, value in sampled.items():
                if name not in cache:
                    cache[name] = value.new_empty((st.num_transitions_per_env, *value.shape))
                cache[name][step].copy_(value)
        # V stays fixed during actor fitting. Cache its one-step scores rather
        # than K copies of every next observation, which can dominate G1 memory.
        self._candidate_cache = cache

    def update(self) -> dict[str, float]:
        self._candidate_cache = None
        if self.candidate_collection == "update":
            self._candidate_transitions_this_update = 0
        try:
            metrics = super().update()
            metrics.update({
                "ValueMPO/candidate_transitions": float(self._candidate_transitions_this_update),
                "ValueMPO/candidate_transitions_total": float(self.candidate_transitions_total),
                "ValueMPO/candidate_cache_bytes": float(sum(
                    value.numel() * value.element_size() for value in self._candidate_cache.values()
                )),
                "ValueMPO/rollout_temperature": self.rollout_temperature.item(),
                "ValueMPO/critic_frozen_during_actor": 1.0,
            })
            return metrics
        finally:
            self._candidate_cache = None
            self._rollout_candidates[:] = [None] * len(self._rollout_candidates)

    def policy_snapshot_state_dict(self) -> dict:
        state = super().policy_snapshot_state_dict()
        state["value_candidate_transitions_total"] = self.candidate_transitions_total
        state["value_candidate_generator_state"] = self._candidate_generator.get_state()
        return state

    def load_policy_snapshot_state_dict(self, state_dict: dict | None) -> None:
        super().load_policy_snapshot_state_dict(state_dict)
        self.candidate_transitions_total = int((state_dict or {}).get("value_candidate_transitions_total", 0))
        if state_dict and "value_candidate_generator_state" in state_dict:
            self._candidate_generator.set_state(state_dict["value_candidate_generator_state"].cpu())
        self._candidate_cache = None
