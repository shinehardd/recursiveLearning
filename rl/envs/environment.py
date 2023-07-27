import abc
import numpy as np
from rl.utils.array_types import Array, BoundedArray


class Environment(abc.ABC):

    @abc.abstractmethod
    def render(self):
        """ render environment"""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """ reset environment
            Returns initial observations and states"""

        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action):
        """ An agent acts on the environment and moves to the next state
            Returns reward, terminated, info
        """

        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """ close environment """
        raise NotImplementedError

    @abc.abstractmethod
    def environment_spec(self):
        """ return environment spec"""
        raise NotImplementedError

    @abc.abstractmethod
    def max_episode_limit(self):
        """ return max episode limit"""
        raise NotImplementedError

    def get_num_of_agents(self):
        return self.n_agents

class EnvironmentSpec:
    """Environment specification."""

    def __init__(self,
                 action_shape: list,
                 action_dtype,
                 action_high,
                 action_low,
                 action_dim,
                 b_continuous_action,
                 state_shape: list,
                 state_dtype=np.float32,
                 full_state_shape: list = None,
                 full_state_dtype=np.float32,
                 reward_dtype=np.float32,
                 discount_dtype=np.float32,
                 n_agents=1):
        """Initialize the environment specification
        Args:
          action_shape: action dimensions.
          action_dtype : dtype of the action spaces.
          state_shape: state dimensions.
          state_dtype : dtype of the state spaces.
          reward_dtype: dtype of the reward
          discount_dtype: dtype of the discounts.
        """

        self.b_continuous_action = b_continuous_action
        self.action_spec = BoundedArray(action_shape, action_dtype, action_low, action_high)
        self.action_dim = action_dim
        self.state_spec = Array(state_shape, state_dtype)
        self.n_agents = n_agents
        self.full_state_spec = Array(full_state_shape, full_state_dtype) if full_state_shape else self.state_spec
        self.reward_spec = Array((), reward_dtype)
        self.discount_spec = BoundedArray((), discount_dtype, 0.0, 1.0)
