import abc
import math
from rl.utils.logging import Logger

class LRScheduler(abc.ABC):
    @abc.abstractmethod
    def step(self, current_timestep: int = 0):
        """ close environment """
        raise NotImplementedError

class MultiStepLR(LRScheduler):
    def __init__(self,
                 logger: Logger,
                 param_groups,
                 timestep_milestones: list,
                 lr_milestones: list,
                 name: str='',
                 verbose:bool=False):

        self.logger = logger
        self.param_groups = param_groups
        self.timestep_milestones = timestep_milestones
        self.lr_milestones = lr_milestones
        self.name = name
        self.verbose = verbose
        self.current_index = -1

    def step(self, current_timestep: int = 0):

        # last milestone
        next_index = self.current_index+1
        if next_index == len(self.timestep_milestones): return

        # check arrival of next milestone
        if current_timestep >= self.timestep_milestones[next_index]:
            self.current_index += 1
            self.param_groups['lr'] = self.lr_milestones[self.current_index]
            if self.verbose:
                logging_msg = "{} MultiStepLR annealing : milestone {} lr {:.6f} ".format(
                              self.name,
                              self.timestep_milestones[next_index],
                              self.lr_milestones[self.current_index])
                self.logger.console_logger.info(logging_msg)

class LinearLR(LRScheduler):
    def __init__(self,
                 logger: Logger,
                 param_groups,
                 start_lr: float,
                 end_timesteps: int,
                 interval: int = 1,
                 start_timesteps: int = 0,
                 name: str = '',
                 verbose: bool = False):

        self.logger = logger
        self.param_groups = param_groups
        self.start_lr = start_lr
        self.start_timesteps = start_timesteps
        self.end_timesteps = end_timesteps
        self.interval = interval
        self.name = name
        self.verbose = verbose
        self.last_timestep = 0

    def step(self, current_timestep: int = 0):

        if current_timestep < self.start_timesteps: return

        if (current_timestep - self.last_timestep) >= self.interval:
            fraction = 1.0 - (current_timestep - self.start_timesteps)/(self.end_timesteps - self.start_timesteps)
            self.param_groups['lr'] = self.start_lr*fraction
            self.last_timestep = current_timestep

            if self.verbose:
                logging_msg = "{} LinearLR annealing : remainder_frac {:.3f} lr {:.5f} ".format(
                               self.name,
                               fraction,
                               self.start_lr*fraction)
                self.logger.console_logger.info(logging_msg)

class CosineLR(LRScheduler):
    def __init__(self,
                 logger: Logger,
                 param_groups,
                 start_lr: float,
                 end_timesteps: int,
                 interval: int = 1,
                 start_timesteps: int = 0,
                 name: str = '',
                 verbose: bool = False):

        self.logger = logger
        self.param_groups = param_groups
        self.start_lr = start_lr
        self.start_timesteps = start_timesteps
        self.end_timesteps = end_timesteps
        self.interval = interval
        self.name = name
        self.verbose = verbose
        self.last_timestep = 0

    def step(self, current_timestep: int = 0):

        if current_timestep < self.start_timesteps: return

        if (current_timestep - self.last_timestep) >= self.interval:
            rate = (current_timestep - self.start_timesteps)/(self.end_timesteps - self.start_timesteps)
            fraction = math.cos((math.pi/2.)*rate)

            self.param_groups['lr'] = self.start_lr*fraction
            self.last_timestep = current_timestep

            if self.verbose:
                logging_msg = "{} LinearLR annealing : remainder_frac {:.3f} lr {:.5f} ".format(
                               self.name,
                               fraction,
                               self.start_lr*fraction)
                self.logger.console_logger.info(logging_msg)