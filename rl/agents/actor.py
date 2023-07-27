import torch
from typing import Tuple, Dict
from copy import deepcopy

from types import SimpleNamespace
from rl.utils.logging import Logger
from rl.envs.environment import Environment
from rl.datasets.buffer_info import BufferInfo

from rl.agents.base import NetworkBase
from rl.utils.util import to_tensor, to_device, to_numpy
from rl.datasets.rollout_buffer import RolloutBuffer
from rl.agents.base import ActorBase, VariableSource


class Actor(ActorBase):

    def __init__(self,
                 config: SimpleNamespace,
                 logger: Logger,
                 env: Environment,
                 buffer_info: BufferInfo,
                 network: NetworkBase,
                 variable_source: VariableSource,
                 actor_buffer_shape: Tuple,
                 actor_id: int = 0):

        self.config = config
        self.logger = logger
        self.env = env
        self._variable_source = variable_source
        self.actor_id = actor_id

        # create actor's network
        self.network = deepcopy(network)
        state_dict = self._variable_source.get_variables()
        self.network.load_state_dict(state_dict)

        self.buffer = None
        if self.config.training_mode:
            # create an actor's rollout buffer (that is different from the agent's buffer)
            self.buffer_info = buffer_info
            buffer_class = self.default_buffer_class()

            self.buffer = buffer_class(config=self.config,
                                       buffer_info=self.buffer_info,
                                       buffer_shape=actor_buffer_shape)

    def default_buffer_class(self):
        return RolloutBuffer

    def reset(self):
        pass

    def select_action(self,
                      state: torch.Tensor,
                      total_n_timesteps: int) -> torch.Tensor:

        """Samples from the network and returns an action."""
        # 1. Convert numpy type to tensor type and insert batch dimension to them
        state = to_device(to_tensor(state), self.config).unsqueeze(dim=0)

        # 2. Action Selection
        action = self.network.select_action(state=state,
                                            total_n_timesteps=total_n_timesteps)

        # 3. Convert tensor type to numpy type
        action = to_numpy(action, self.config).squeeze()
        return action

    def observe(self, rollout: Dict):
        if not self.config.training_mode: return
        self.buffer += rollout

    def update(self, state_dict=None):
        # Update the actor weights when learner updates.
        if state_dict is None:
            state_dict = self._variable_source.get_variables()
        self.network.load_state_dict(state_dict)

    def cuda(self):
        self.network.cuda()

    def rollouts(self):
        return self.buffer

    def clear_rollouts(self):
        if not self.config.training_mode: return
        self.buffer.clear()