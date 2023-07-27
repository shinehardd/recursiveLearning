import torch
from types import SimpleNamespace
from copy import deepcopy
from rl.utils.util import hard_update, soft_update

from rl.utils.logging import Logger
from rl.envs.environment import EnvironmentSpec
from rl.networks.network import QNetwork_DQN
from rl.agents.base import NetworkBase
from rl.utils.action_selectors import EpsilonGreedyActionSelector

class DQNNetwork(NetworkBase):
    def __init__(self,
                 config: SimpleNamespace,
                 logger: Logger,
                 environment_spec: EnvironmentSpec):

        super(DQNNetwork, self).__init__(config, logger, environment_spec)

        if self.environment_spec.b_continuous_action:
            raise Exception("DQN doesn't support continuous action space")

        self.critic = self.make_critic()
        self.target_critic = deepcopy(self.critic)
        hard_update(self.critic, self.target_critic)

        # Epsilon Greedy Action Selector
        self.action_selector = EpsilonGreedyActionSelector(self.config)

    def make_critic(self):
        # QNetwork_DQN outputs q_values of all actions
        return QNetwork_DQN(config=self.config,
                            state_dim=self.state_dim,
                            action_dim=self.action_dim,
                            hidden_dims=self.config.critic_hidden_dims)

    def hard_update_target(self):
        hard_update(self.critic, self.target_critic)

    def soft_update_target(self):
        soft_update(self.critic, self.target_critic, self.config.tau)

    @torch.no_grad()
    def select_action(self,
                      state: torch.Tensor,
                      total_n_timesteps: int) -> torch.Tensor:

        # get q_values of all actions
        q_values = self.critic(state)

        # Epsilon Greedy
        chosen_actions = self.action_selector.select_action(agent_input=q_values,
                                                            total_n_timesteps=total_n_timesteps)
        return chosen_actions


    def cuda(self):
        self.critic.cuda(self.config.device_num)
        self.target_critic.cuda(self.config.device_num)