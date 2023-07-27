import torch
from types import SimpleNamespace
from rl.utils.logging import Logger
from rl.envs.environment import EnvironmentSpec
from rl.networks.network import GaussianMLPPolicy, CategoricalMLPPolicy
from rl.agents.base import NetworkBase
from rl.networks.network import MLP


class REINFORCENetwork(NetworkBase):
    def __init__(self,
                 config: SimpleNamespace,
                 logger: Logger,
                 environment_spec: EnvironmentSpec):

        super(REINFORCENetwork, self).__init__(config, logger, environment_spec)

        self.config = config
        self.logger = logger
        self.environment_spec = environment_spec

        self.b_continuous_action = self.environment_spec.b_continuous_action
        self.action_dim = self.environment_spec.action_dim
        self.state_dim = self.environment_spec.state_spec.shape[0]

        self.policy = self.make_policy()

        self.baseline = self.make_baseline()

    def make_policy(self):

        hidden_dim = self.config.actor_hidden_dims
        policy_argument = [self.config, self.state_dim, hidden_dim, self.action_dim]

        if self.b_continuous_action:
            return GaussianMLPPolicy(*policy_argument)

        return CategoricalMLPPolicy(*policy_argument)

    def select_action(self,
                      state: torch.Tensor,
                      total_n_timesteps: int) -> torch.Tensor:

        return self.policy.select_action(state, self.config.training_mode)

    def cuda(self):
        self.policy.cuda(self.config.device_num)
        self.baseline.cuda(self.config.device_num)

    def _log_prob(self, distribution, action):
        # change to 1 dimension (safe log probability calculation)
        b_squeeze = self.b_continuous_action is False and action.shape[-1] == 1
        if b_squeeze: action = action.squeeze()

        # log probability and entropy
        log_prob = distribution.log_prob(action)

        if b_squeeze: log_prob = log_prob.unsqueeze(-1)
        if (log_prob < -1e05).any(): print(log_prob)

        return log_prob

    def forward(self, state, action):

        b_action_squeeze = self.b_continuous_action is False and action.shape[-1]==1
        if b_action_squeeze:
            action = action.squeeze()
        # action distribution
        distribution = self.policy.distribution(state)

        # log probability
        if self.b_continuous_action:
            clipped_action = torch.atanh(torch.clamp(action, -1 + 1e-7, 1 - 1e-7))
            log_prob = distribution.log_prob(clipped_action)
        else:
            log_prob = distribution.log_prob(action)

        if b_action_squeeze:
            log_prob = log_prob.unsqueeze(-1)

        baseline = self.baseline(state)
        return log_prob,baseline

    def make_baseline(self):
        layer_dims = self.config.actor_hidden_dims + [1]
        return MLP(self.config, self.state_dim, layer_dims)