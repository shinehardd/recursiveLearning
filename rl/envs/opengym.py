from types import SimpleNamespace
from rl.envs.environment import Environment, EnvironmentSpec
import numpy as np
import gym
from rl.utils.util import scale_bias

class OpenGym(Environment):

    def __init__(self, config: SimpleNamespace):

        self.config = config
        env_name = self.config.env_name
        random_seed = self.config.random_seed

        render_mode = None if self.config.training_mode and not self.config.render else "human"
        self.env = gym.make(env_name, render_mode=render_mode)  # render_mode: human, rgb_array
        self.env.action_space.seed(random_seed)
        self.env.observation_space.seed(random_seed)
        self.b_continuous_action = False if isinstance(self.env.action_space, gym.spaces.Discrete) else True
        self.n_agents = 1

        # calculate scale and bias of action scope in [-1,1] to make the action scope in [low, high]
        if self.b_continuous_action:
            self.action_scale, self.action_bias = scale_bias(self.env.action_space.high,self.env.action_space.low)

    def render(self):
        return self.env.render()    # render_mode: human, rgb_array render_mode="human"

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        # recover to original action scale
        action = self.original_scale(action)

        # converse the shape of action to the shape of action in environment
        if self.b_continuous_action:
            action = np.reshape(action, self.env.action_space.shape)
        else:
            if isinstance(action, np.ndarray):
                action = action.item()

        return self.env.step(action)

    def close(self):
        self.env.close()

    def original_scale(self, action):
        """action scope [-1,1] to [low, high]"""
        if self.b_continuous_action:
            return self.action_scale*action + self.action_bias
        return action

    def normed_scale(self, value, bias, scale):
        """action scope [low, high] to [-1,1]"""
        if self.b_continuous_action:
            return (value - bias)/scale
        return (value - bias)/scale

    def environment_spec(self):
        # action space type
        self.b_continuous_action = False if isinstance(self.env.action_space, gym.spaces.Discrete) else True

        # action space dimension
        if self.b_continuous_action:
            action_shape = self.env.action_space.shape
            action_dim = self.env.action_space.shape[0]
            action_high = self.env.action_space.high
            action_low = self.env.action_space.low
        else:
            action_shape = self.env.action_space.shape or [1]
            action_dim = self.env.action_space.n
            action_high = [action_dim - 1]
            action_low = [0]

        environment_spec = EnvironmentSpec(
            action_shape=action_shape,
            action_dtype=self.env.action_space.dtype,
            action_high=action_high,
            action_low=action_low,
            action_dim=action_dim,
            b_continuous_action=self.b_continuous_action,
            state_shape=self.env.observation_space.shape,
            state_dtype=self.env.observation_space.dtype)

        return environment_spec

    def select_action(self):
        action = self.env.action_space.sample()
        # normalize an action
        if self.b_continuous_action:
            action = self.normed_scale(action, self.action_bias, self.action_scale)
        return action

    def max_episode_limit(self):
        return self.env._max_episode_steps