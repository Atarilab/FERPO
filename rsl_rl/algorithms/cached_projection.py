"""Shared actor projection over fixed, rollout-indexed candidate scores."""

import torch


class CachedProjection:
    @torch.no_grad()
    def _actor_samples(self, minibatch: dict) -> tuple:
        if self._candidate_cache is None:
            self._build_candidate_cache()
        times = minibatch["time_indices_batch"]
        envs = minibatch["env_indices_batch"]
        samples = {name: value[times, :, envs].transpose(0, 1) for name, value in self._candidate_cache.items()}
        reference_log_prob = samples["reference_log_prob"]
        older_log_prob = samples.get("older_log_prob", reference_log_prob)
        proposal_log_prob = samples.get("proposal_log_prob")
        weights, log_ratio = self._maxent_weights(
            samples["q_values"], reference_log_prob, older_log_prob, proposal_log_prob
        )
        obs = minibatch["obs_batch"]
        self._policy_act(self.old_policy, obs, None, None)
        previous_distribution = None
        if self._uses_momentum():
            self._policy_act(self.previous_policy, obs, None, None)
            previous_distribution = self.previous_policy.distribution
        return (
            samples["actions"], samples["q_values"], weights, log_ratio,
            reference_log_prob - older_log_prob, self.old_policy.distribution,
            previous_distribution, samples.get("pre_tanh"), proposal_log_prob,
        )

    def update_critic(self, minibatch: dict) -> dict:
        if self._candidate_cache is not None:
            raise RuntimeError("The critic must remain frozen after candidate caching begins.")
        return super().update_critic(minibatch)
