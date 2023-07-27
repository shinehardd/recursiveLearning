import abc
import torch
import torch.nn as nn
import numpy as np
from typing import Generic, TypeVar
from types import SimpleNamespace

import rl.utils.array_types as types
from rl.utils.logging import Logger
from rl.envs.environment import EnvironmentSpec

TimeStep = int
T = TypeVar('T')


class ActorBase(abc.ABC):
    """Interface for an agent that can act.

    This interface defines an API for an Actor to interact with an EnvironmentLoop
    (see runner.environment_loop), e.g. a simple RL loop where each step is of the
    form:

      # Make the first observation.
      timestep = env.reset()
      actor.reset()

      # Take a step and observe.
      action = actor.select_action(timestep.observation)
      next_timestep = env.step(action)
      actor.observe(action, next_timestep)

      # Update the actor policy/parameters.
      actor.update()
    """

    @abc.abstractmethod
    def reset(self):
        """Initialize policy """

    @abc.abstractmethod
    def select_action(self, state: torch.Tensor, n_timesteps: int) -> torch.Tensor:
        """Samples from the policy and returns an action."""

    @abc.abstractmethod
    def observe(
            self,
            action: types.Array,
            next_timestep: TimeStep,
    ):
        """Make an observation of timestep data from the environment.

        Args:
          action: action taken in the environment.
          next_timestep: timestep produced by the environment given the action.
        """

    @abc.abstractmethod
    def update(self, wait: bool = False):
        """Perform an update of the actor parameters from past observations.

        Args:
          wait: if True, the update will be blocking.
        """

class VariableSource(abc.ABC):
    """Abstract source of variables.

    Objects which implement this interface provide a source of variables, returned
    as a collection of (nested) numpy arrays. Generally this will be used to
    provide variables to some learned policy/etc.
    """

    @abc.abstractmethod
    def get_variables(self) -> dict:
        """Return the named variables as a collection of (nested) numpy arrays.

        Args:
          names: args where each name is a string identifying a predefined subset of
            the variables.

        Returns:
          A list of (nested) numpy arrays `variables` such that `variables[i]`
          corresponds to the collection named by `names[i]`.
        """

class Saveable(abc.ABC, Generic[T]):
    """An interface for saveable objects."""

    @abc.abstractmethod
    def save(self, checkpoint_path: str):
        """Returns the state from the object to be saved."""

    @abc.abstractmethod
    def restore(self, checkpoint_path: str):
        """Given the state, restores the object."""

class Learner(VariableSource, Saveable):
    """Abstract learner object.

    This corresponds to an object which implements a learning loop. A single step
    of learning can be implemented via the `step` method and this step
    is generally interacted with via the `update` method which runs update
    continuously.
    """

    def update(self, total_n_timesteps: int, total_n_episodes:int) -> None:
        """Run the update loop"""
        raise NotImplementedError('Method "update" is not implemented.')

    def save(self, checkpoint_path: str):
        raise NotImplementedError('Method "save" is not implemented.')

    def restore(self, checkpoint_path: str):
        raise NotImplementedError('Method "restore" is not implemented.')

    def cuda(self):
        self.network.cuda()

    def get_variables(self) -> dict:
        return self.network.get_variables()

class NetworkBase(nn.Module, VariableSource, Saveable):
    """ Network base object.

    This corresponds to an object which has policy and critic networks.
    """

    def __init__(self,
                 config: SimpleNamespace,
                 logger: Logger,
                 environment_spec: EnvironmentSpec):

        super(NetworkBase, self).__init__()

        self.config = config
        self.logger = logger
        self.environment_spec = environment_spec

        self.b_continuous_action = self.environment_spec.b_continuous_action

        self.action_dim = self.environment_spec.action_dim
        self.state_dim = np.array(self.environment_spec.state_spec.shape).prod()
        self.last_logging_step = 0

    @abc.abstractmethod
    def select_action(self,
                      state: torch.Tensor,
                      total_n_timesteps: int,
                      avail_action: torch.Tensor = None) -> torch.Tensor:
        """Sample an action from policy

        Args:
          state: state of environment
          exploration: exploration class that can apply when sampling
        """

    def forward(self, state, action):
        pass

    @abc.abstractmethod
    def cuda(self):
        """Set network device to cuda
        """

    def save(self, checkpoint_path: str):
        torch.save(self.state_dict(), "{}/network.th".format(checkpoint_path))

    def restore(self, checkpoint_path: str):
        state_dict = torch.load("{}/network.th".format(checkpoint_path), map_location=torch.device(self.config.device))
        self.load_state_dict(state_dict)

    def get_variables(self) -> dict:
        return self.state_dict()