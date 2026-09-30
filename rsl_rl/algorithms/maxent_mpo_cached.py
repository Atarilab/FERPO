"""Staged, cached-Q control for the state-value ablation."""

import torch

from rsl_rl.modules import ActorQ, ActorV
from rsl_rl.networks import TanhNormal
from .cached_projection import CachedProjection
from .critic_stopping import CriticStopping
from .maxent_mpo import MaxEntMPO


class MaxEntMPOCached(CriticStopping, CachedProjection, MaxEntMPO):
    """Fit Q, freeze it, then reuse K scored candidates throughout actor fitting."""

    def __init__(self, policy, storage, *args, **kwargs):
        if not isinstance(policy, ActorQ) or isinstance(policy, ActorV) or policy.is_recurrent:
            raise ValueError("MaxEntMPOCached requires the feed-forward ActorQ policy.")
        kwargs.setdefault("interleave_actor_critic_updates", False)
        if kwargs["interleave_actor_critic_updates"]:
            raise ValueError("MaxEntMPOCached requires staged critic/actor updates.")
        if kwargs.get("symmetry_cfg"):
            raise ValueError("Cached candidates require unaugmented observations.")
        super().__init__(policy, storage, *args, **kwargs)
        if self.num_learning_epochs < 1 or self.num_mini_batches < 1:
            raise ValueError("Cached projection requires positive epoch and minibatch counts.")
        if storage.num_envs * storage.num_transitions_per_env % self.num_mini_batches:
            raise ValueError("Cached projection requires minibatches to cover the complete rollout evenly.")
        self._candidate_cache = None

    @torch.no_grad()
    def _build_candidate_cache(self):
        st = self.storage
        if st.step != st.num_transitions_per_env:
            raise RuntimeError("Candidate evaluation requires a complete rollout.")
        cache = {}
        for step in range(st.num_transitions_per_env):
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
            sampled = {
                "actions": actions,
                "reference_log_prob": reference_log_prob,
                "q_values": self._evaluate_sampled_action_values(obs, actions, None, None),
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
        self._candidate_cache = cache

    def update(self):
        self._candidate_cache = None
        try:
            metrics = super().update()
            metrics.update(self._critic_fit_metrics)
            metrics["CachedQ/candidate_cache_bytes"] = float(sum(
                value.numel() * value.element_size() for value in self._candidate_cache.values()
            ))
            metrics["CachedQ/critic_frozen_during_actor"] = 1.0
            return metrics
        finally:
            self._candidate_cache = None
