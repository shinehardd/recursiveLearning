import abc
import torch
import torch.nn as nn
from typing import Callable, List
from types import SimpleNamespace
from torch.distributions import Normal, Categorical


# Initialize Policy weights
def orthogonal_init(m, nonlinearity="tanh"):
    gain = 0.01
    if nonlinearity != "policy":
        gain = torch.nn.init.calculate_gain(nonlinearity)

    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight.data, gain)
        torch.nn.init.zeros_(m.bias.data)


class MLP(nn.Module):
    def __init__(self,
                 config: SimpleNamespace,
                 input_dim: int,
                 layer_dims: List[int],
                 activation: Callable[[torch.Tensor],torch.Tensor] = nn.ReLU,
                 output_activation: Callable[[torch.Tensor],torch.Tensor] = nn.Identity):
        super(MLP, self).__init__()
        self.config = config

        # activation
        activations = [activation for _ in layer_dims[:-1]]
        activations.append(output_activation)

        # create layers
        layers = []
        for output_dim, activation in zip(layer_dims, activations):
            layer = nn.Sequential(nn.Linear(input_dim, output_dim), activation())
            layers.append(layer)
            input_dim = output_dim

        self.layers = nn.Sequential(*layers)
        self.apply(orthogonal_init)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class Policy(abc.ABC):
    """method is empty"""
    state_dim = 0
    action_dim = 0


class StochasticPolicy(Policy):

    @abc.abstractmethod
    def distribution(self, state):
        """return action distribution"""

    @abc.abstractmethod
    def select_action(self, state: torch.Tensor, training_mode: bool = True):
        """return action sample"""


class DeterministicPolicy(Policy):

    @torch.no_grad()
    def select_action(self, state):
        return self(state).detach()


class GaussianPolicy(StochasticPolicy):

    def distribution(self, state):
        mean, log_std = self(state)
        return Normal(mean, log_std.exp())

    @torch.no_grad()
    def select_action(self, state: torch.Tensor, training_mode: bool = True):
        """
            none-differentiable action sampling
        """
        distribution = self.distribution(state)
        if training_mode:
            action = distribution.sample()
            action = torch.tanh(action)
            action = torch.atanh(torch.clamp(action, -1 + 1e-7, 1 - 1e-7))
        else:
            action = distribution.mean
        return action


class CategoricalPolicy(StochasticPolicy):

    def distribution(self, state):
        return Categorical(self(state))

    @torch.no_grad()
    def select_action(self, state: torch.Tensor, training_mode: bool = True):
        distribution = self.distribution(state)
        if training_mode:
            action = distribution.sample()
        else:
            action = distribution.probs.argmax(dim=-1, keepdim=True)
        return action


class GaussianMLPPolicy(MLP, GaussianPolicy):
    def __init__(self,
                 config: SimpleNamespace,
                 state_dim: int,
                 hidden_dims: List[int],
                 action_dim: int,
                 ):

        super(GaussianMLPPolicy, self).__init__(config, state_dim, hidden_dims)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        self.apply(orthogonal_init)

    def forward(self, state):
        x = super(GaussianMLPPolicy, self).forward(state)
        mean = self.mean_head(x)
        log_std = torch.tanh(self.log_std_head(x))
        return mean, log_std


class CategoricalMLPPolicy(MLP, CategoricalPolicy):
    def __init__(self,
                 config: SimpleNamespace,
                 state_dim: int,
                 hidden_dims: List[int],
                 action_dim: int):

        layer_dims = hidden_dims + [action_dim]
        super(CategoricalMLPPolicy, self).__init__(config,
                                                state_dim,
                                                layer_dims,
                                                activation=nn.Tanh,
                                                output_activation=nn.Identity)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.Softmax = nn.Softmax(dim=-1)
        self.apply(lambda m: orthogonal_init(m, "policy"))

    def forward(self, state):
        logits = super(CategoricalMLPPolicy, self).forward(state)
        return self.Softmax(logits)


class ValueNetwork(MLP):
    def __init__(self,
                 config: SimpleNamespace,
                 state_dim: int,
                 hidden_dims: List[int]):
        output_dim = 1
        layer_dims = hidden_dims + [output_dim]
        super(ValueNetwork, self).__init__(config, state_dim, layer_dims)

    def forward(self, state):
        return super(ValueNetwork, self).forward(state)


class QNetwork(MLP):
    def __init__(self,
                 config: SimpleNamespace,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int]):

        input_dim = state_dim + action_dim
        output_dim = 1
        layer_dims = hidden_dims + [output_dim]
        super(QNetwork, self).__init__(config, input_dim, layer_dims)

        self.apply(orthogonal_init)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        return super(QNetwork, self).forward(state_action)


class QNetwork_DQN(MLP):
    def __init__(self,
                 config: SimpleNamespace,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int]):

        input_dim = state_dim
        output_dim = action_dim
        layer_dims = hidden_dims + [output_dim]
        super(QNetwork_DQN, self).__init__(config, input_dim, layer_dims)

        self.apply(orthogonal_init)

    def forward(self, state):
        return super(QNetwork_DQN, self).forward(state)