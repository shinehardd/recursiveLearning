from typing import List, Union, Tuple
import numpy as np
import torch
from types import SimpleNamespace
from rl.utils.logging import Logger
from rl.envs.environment import Environment
from rl.datasets.buffer_info import BufferInfo
from rl.agents.base import Learner
from rl.datasets.buffer import Buffer
from rl.datasets.rollout_buffer import RolloutBuffer
from rl.datasets.replay_buffer import ReplayBuffer
from rl.agents.base import NetworkBase
from rl.agents.actor import Actor
from rl.agents.base import VariableSource


class Agent(Learner):
    """Agent class which combines acting and learning.

    This has actors, networks (policy and value), laerner, buffer as its components
    and typically provides an implementation of the `Learner` interface which learns.
    """

    def __init__(self,
                 config: SimpleNamespace,
                 logger: Logger,
                 env: Environment,
                 buffer_info: BufferInfo,
                 network_class: NetworkBase,
                 learner_class: Learner,
                 actor_class=Actor,
                 buffer_class: Buffer = None,
                 policy_type: str = "on_policy"):

        self.config = config
        self.logger = logger
        self.env = env
        self.buffer_info = buffer_info
        self.actor_class = actor_class
        self.policy_type = policy_type
        self.n_agents = 1

        # inference & training network
        self.network = network_class(
            config=config,
            logger=logger,
            environment_spec=env.environment_spec(),
        )

        self.learner = None
        self.buffer = None
        self.n_envs = 1

        # training mode
        if config.training_mode:
            self.n_envs = config.n_envs

            buffer_class = buffer_class or self.default_buffer_class()
            buffer_shape = self.default_buffer_shape()

            # make replay/rollout buffer
            self.buffer = buffer_class(
                config=config,
                buffer_info=buffer_info,
                buffer_shape=buffer_shape)

            # make learner
            self.learner = learner_class(
                config=config,
                logger=logger,
                environment_spec=env.environment_spec(),
                network=self.network,
                buffer=self.buffer)

        # make actors
        actor_buffer_shape = self.default_actor_buffer_shape()
        self.actors = self.make_actors(
            config=config,
            logger=logger,
            env=env,
            buffer_info=buffer_info,
            network=self.network,
            variable_source=self.network,
            actor_buffer_shape=actor_buffer_shape,
            n_envs=self.n_envs,
            actor_class=actor_class)

        # for single actor in training or inference time
        self.actor = self.actors[0]

    def default_buffer_class(self):
        return RolloutBuffer if self.policy_type == "on_policy" else ReplayBuffer

    def default_buffer_shape(self):
        if self.policy_type == "on_policy":
            actor_buffer_size = self.default_actor_buffer_shape()[0]
            return [actor_buffer_size * self.n_envs]

        return [self.config.replay_buffer_size]

    def default_actor_buffer_shape(self):
        if self.config.n_steps != 0:
            return [self.config.n_steps]

        max_len_episode = self.env.max_episode_limit()
        return [max_len_episode*self.config.n_episodes]

    def make_actors(self,
                     config: SimpleNamespace,
                     logger: Logger,
                     env: Environment,
                     buffer_info: BufferInfo,
                     network: NetworkBase,
                     variable_source: VariableSource,
                     actor_buffer_shape: Tuple,
                     n_envs: int,
                     actor_class=Actor):

        actors = []
        for i in range(n_envs):
            actor = actor_class(config=config,
                                logger=logger,
                                env=env,
                                buffer_info=buffer_info,
                                network=network,
                                variable_source=variable_source,
                                actor_buffer_shape=actor_buffer_shape,
                                actor_id=i)
            actors.append(actor)

        return actors

    def add_rollouts(self, list_of_buffers: Union[RolloutBuffer, List[RolloutBuffer]]):
        if isinstance(list_of_buffers, RolloutBuffer):
            list_of_buffers = [list_of_buffers]

        for buffer in list_of_buffers:
            self.buffer += buffer

    def update(self, total_n_timesteps: int, total_n_episodes: int, force_update: bool = False):
        learner_update = False
        if self.config.training_mode:
            learner_update = self.learner.update(total_n_timesteps, total_n_episodes)

        # force_update=True: after checkpoint or trained model loading
        if force_update or learner_update:
            for actor in self.actors:
                actor.update()

        return force_update or learner_update

    def save(self, checkpoint_path):
        self.learner.save(checkpoint_path)

    def restore(self, path):
        self.learner.restore(path)

    def load(self, model_path):
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        self.update(total_n_timesteps=0, total_n_episodes=0, force_update=True)

    def get_variables(self) -> List[List[np.ndarray]]:
        if self.config.training_mode:
            return self.learner.get_variables()
        # inference time
        return self.network.get_variables()

    def cuda(self):
        if self.config.training_mode:
            self.learner.cuda()
            for actor in self.actors:
                actor.cuda()

        # inference time
        self.actor.cuda()


