"""The FERPO actor paired with a state-only soft value critic."""

from tensordict import TensorDict
from torch import Tensor

from .actor_q import ActorQ


class ActorV(ActorQ):
    """Reuse ActorQ's architecture and distributional head without action inputs."""

    critic_uses_actions = False

    def critic_features(self, obs: TensorDict, act: Tensor | None = None) -> Tensor:
        return self.critic(self.critic_obs_normalizer(self.get_critic_obs(obs)))

    def evaluate(self, obs: TensorDict, act: Tensor | None = None, *args, return_logits=False):
        return super().evaluate(obs, act, *args, return_logits=return_logits)
