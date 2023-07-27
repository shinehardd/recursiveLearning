import abc


class Scheduler(abc.ABC):
    @abc.abstractmethod
    def value(self, current_timestep: int = 0):
        """ close environment """
        raise NotImplementedError


class LinearScheduler(Scheduler):
    def __init__(self,
                 start_value: float,
                 end_value: float=0,
                 start_timesteps: int=1,
                 end_timesteps: int=-1,
                 interval: int = 1):

        self.start_value = start_value
        self.end_value = end_value
        self.current_value = start_value
        self.start_timesteps = start_timesteps
        self.end_timesteps = end_timesteps
        self.interval = interval
        self.last_timestep = 0

    def value(self, current_timestep: int) -> float:

        # No Scheduling
        if self.end_timesteps == -1:
            return self.current_value

        if current_timestep > self.end_timesteps: return self.current_value

        assert self.start_timesteps <= current_timestep <= self.end_timesteps

        # Warm-up time
        if current_timestep < self.start_timesteps:
            return self.current_value

        # Decay
        if (current_timestep - self.last_timestep) >= self.interval:
            fraction = (current_timestep - self.start_timesteps) / (self.end_timesteps - self.start_timesteps)
            self.current_value = self.start_value + (self.end_value-self.start_value)*fraction
            self.last_timestep = current_timestep
            return self.current_value

        print("LinearScheduler exceptional case - current_timestep=", current_timestep)
        return self.current_value
