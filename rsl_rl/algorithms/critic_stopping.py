"""Trajectory-held-out critic fitting for staged cached-Q experiments."""

import copy
import math
import time

import torch


class CriticStopping:
    """Keep validation trajectories out of critic gradients, but not actor fitting."""

    def __init__(
        self, *args, critic_validation_fraction=0.0, critic_early_stopping=False,
        critic_max_epochs=64, critic_patience=3, critic_min_relative_improvement=1e-4,
        critic_validation_objective="total",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0 <= critic_validation_fraction < 1:
            raise ValueError("critic_validation_fraction must be in [0, 1).")
        if critic_early_stopping and critic_validation_fraction == 0:
            raise ValueError("Critic early stopping requires a validation split.")
        if int(critic_max_epochs) != critic_max_epochs or critic_max_epochs < 1:
            raise ValueError("critic_max_epochs must be a positive integer.")
        if int(critic_patience) != critic_patience or critic_patience < 1:
            raise ValueError("critic_patience must be a positive integer.")
        if not math.isfinite(critic_min_relative_improvement) or critic_min_relative_improvement < 0:
            raise ValueError("critic_min_relative_improvement must be finite and nonnegative.")
        if critic_validation_fraction and self.is_multi_gpu:
            raise ValueError("Trajectory-held-out critic fitting currently requires one GPU.")
        if critic_validation_objective not in ("total", "q"):
            raise ValueError("critic_validation_objective must be 'total' or 'q'.")
        self.critic_validation_fraction = float(critic_validation_fraction)
        self.critic_early_stopping = bool(critic_early_stopping)
        self.critic_max_epochs = int(critic_max_epochs)
        self.critic_patience = int(critic_patience)
        self.critic_min_relative_improvement = float(critic_min_relative_improvement)
        # Keep old configs reproducible; Q-only stopping is an explicit variant.
        self.critic_validation_objective = critic_validation_objective
        self._critic_split_rng = torch.Generator().manual_seed(torch.initial_seed())
        self._critic_fit_metrics = {}
        self.critic_fit_history = []
        if critic_validation_fraction:
            n_valid = round(self.storage.num_envs * critic_validation_fraction)
            if not 0 < n_valid < self.storage.num_envs:
                raise ValueError("The validation split must contain at least one environment per set.")
            if (self.storage.num_envs - n_valid) * self.storage.num_transitions_per_env < self.num_mini_batches:
                raise ValueError("The critic training split is smaller than the minibatch count.")

    def _critic_split_indices(self):
        """Hold out complete environment streams for this rollout (no adjacent-step leakage)."""
        st = self.storage
        envs = torch.randperm(st.num_envs, generator=self._critic_split_rng).to(self.device)
        count = round(st.num_envs * self.critic_validation_fraction)
        times = torch.arange(st.num_transitions_per_env, device=self.device)[:, None]
        validation = (times * st.num_envs + envs[:count]).flatten()
        training = (times * st.num_envs + envs[count:]).flatten()
        # Match the existing rollout generator's single permutation reused each epoch.
        order = torch.randperm(training.numel(), generator=self._critic_split_rng).to(self.device)
        return training[order], validation

    def _critic_index_batch(self, indices):
        st = self.storage
        times = torch.div(indices, st.num_envs, rounding_mode="floor")
        envs = indices % st.num_envs
        result = {
            "obs_batch": st.observations[times, envs],
            "actions_batch": st.actions[times, envs],
            "returns_batch": st.returns[times, envs],
            "truncations_batch": st.truncations[times, envs],
            "hidden_states_batch": (None, None), "masks_batch": None,
            "time_indices_batch": times, "env_indices_batch": envs,
        }
        if st.next_embeddings is not None:
            result["next_embeddings_batch"] = st.next_embeddings[times, envs]
        return result

    @torch.no_grad()
    def _evaluate_critic_indices(self, indices):
        # Bound evaluation memory by the original training minibatch size.
        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        totals = {}
        training = self.policy.critic.training
        self.policy.critic.eval()
        try:
            for chunk in indices.split(batch_size):
                _, metrics = self._critic_loss_and_metrics(self._critic_index_batch(chunk))
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value) * chunk.numel()
        finally:
            self.policy.critic.train(training)
        result = {key: value / indices.numel() for key, value in totals.items()}
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("Nonfinite held-out critic evaluation.")
        return result

    def _critic_checkpoint(self):
        # The optimizer is shared with the actor; rewind only critic state and Adam moments.
        return (
            copy.deepcopy(self.policy.critic.state_dict()),
            {param: copy.deepcopy(self.optimizer.state[param])
             for param in self.policy.critic.parameters() if param in self.optimizer.state},
        )

    def _restore_critic_checkpoint(self, checkpoint):
        weights, states = checkpoint
        self.policy.critic.load_state_dict(weights)
        for param in self.policy.critic.parameters():
            self.optimizer.state.pop(param, None)
            if param in states:
                self.optimizer.state[param] = states[param]
            param.grad = None

    def _staged_critic_updates(self):
        self._critic_fit_metrics = {}
        self.critic_fit_history = []
        if not self.critic_validation_fraction:
            yield from super()._staged_critic_updates()
            return
        start = time.perf_counter()
        training, validation = self._critic_split_indices()
        if self.mask_critic_loss_on_truncation:
            truncations = self.storage.truncations.flatten()
            if bool(truncations[training].all()) or bool(truncations[validation].all()):
                raise ValueError("Critic training and validation must each contain unmasked targets.")
        batches = training.tensor_split(self.num_mini_batches)
        monitor_key = "q_loss" if self.critic_validation_objective == "q" else "value_loss"
        components = {
            "q_loss": "q_loss", "auxiliary_loss": "auxiliary_loss_masked", "total_loss": "value_loss",
        }

        def record_epoch(epoch, validation_metrics, training_metrics=None):
            row = {"epoch": epoch}
            for split, metrics in (("validation", validation_metrics), ("training", training_metrics)):
                if metrics is not None:
                    row[f"{split}_loss"] = metrics[monitor_key]
                    row.update({f"{split}_{label}": metrics[key] for label, key in components.items()})
            self.critic_fit_history.append(row)

        validation_metrics = self._evaluate_critic_indices(validation)
        initial_loss = best_loss = validation_metrics[monitor_key]
        best_epoch = bad_epochs = 0
        checkpoint = self._critic_checkpoint() if self.critic_early_stopping else None
        limit = self.critic_max_epochs if self.critic_early_stopping else self.num_learning_epochs
        stopped = False
        record_epoch(0, validation_metrics)
        for epoch in range(1, limit + 1):
            train_totals = {key: 0.0 for key in components.values()}
            for indices in batches:
                metrics = self.update_critic(self._critic_index_batch(indices))
                if not math.isfinite(metrics["value_loss"]):
                    raise FloatingPointError("Nonfinite critic training loss.")
                for key in train_totals:
                    train_totals[key] += metrics[key] * indices.numel()
                yield metrics
            validation_metrics = self._evaluate_critic_indices(validation)
            validation_loss = validation_metrics[monitor_key]
            record_epoch(epoch, validation_metrics, {
                key: value / training.numel() for key, value in train_totals.items()
            })
            significant = validation_loss < best_loss - self.critic_min_relative_improvement * max(abs(best_loss), 1e-12)
            bad_epochs = 0 if significant else bad_epochs + 1
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                if self.critic_early_stopping:
                    checkpoint = self._critic_checkpoint()
            if self.critic_early_stopping and bad_epochs >= self.critic_patience:
                stopped = True
                break
        last_loss = validation_loss
        if self.critic_early_stopping:
            self._restore_critic_checkpoint(checkpoint)
            del checkpoint
            validation_metrics = self._evaluate_critic_indices(validation)
        final_training = self._evaluate_critic_indices(training)
        self._critic_fit_metrics = {
            "CriticFit/epochs": float(epoch),
            "CriticFit/selected_epoch": float(best_epoch if self.critic_early_stopping else epoch),
            "CriticFit/best_epoch": float(best_epoch),
            "CriticFit/stopped_on_validation": float(stopped),
            "CriticFit/reached_epoch_cap": float(self.critic_early_stopping and not stopped),
            "CriticFit/validation_objective_is_q": float(self.critic_validation_objective == "q"),
            "CriticFit/validation_initial": initial_loss,
            "CriticFit/validation_best": best_loss,
            "CriticFit/validation_last": last_loss,
            "CriticFit/validation_selected": validation_metrics[monitor_key],
            "CriticFit/training_selected": final_training[monitor_key],
            "CriticFit/generalization_gap": validation_metrics[monitor_key] - final_training[monitor_key],
            "CriticFit/train_samples": float(training.numel()),
            "CriticFit/validation_samples": float(validation.numel()),
            "CriticFit/time_seconds": time.perf_counter() - start,
        }
        for key, value in validation_metrics.items():
            self._critic_fit_metrics[f"CriticFit/validation_{key}"] = value
        for row in self.critic_fit_history:
            for key, value in row.items():
                if key != "epoch":
                    self._critic_fit_metrics[f"CriticFit/{key}_epoch_{row['epoch']:02d}"] = value

    def policy_snapshot_state_dict(self):
        state = super().policy_snapshot_state_dict()
        state["critic_split_rng"] = self._critic_split_rng.get_state()
        return state

    def load_policy_snapshot_state_dict(self, state_dict):
        super().load_policy_snapshot_state_dict(state_dict)
        if state_dict and "critic_split_rng" in state_dict:
            self._critic_split_rng.set_state(state_dict["critic_split_rng"].cpu())
