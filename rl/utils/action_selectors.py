import numpy as np
import torch
from types import SimpleNamespace
from torch.distributions import Categorical


class DecayThenFlatSchedule():

    def __init__(self,
                 start: int,
                 finish: int,
                 time_length: int,
                 decay: str = "exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass


class EpsilonGreedyActionSelector():

    def __init__(self, config: SimpleNamespace):
        self.config = config

        self.schedule = DecayThenFlatSchedule(
            start=config.epsilon_start,
            finish=config.epsilon_finish,
            time_length=config.epsilon_anneal_time,
            decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_input, total_n_timesteps: int):

        # Assuming agent_input is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(total_n_timesteps)

        if not self.config.training_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        random_actions = Categorical(torch.ones_like(agent_input).float()).sample().long()
        selected_action = agent_input.max(dim=-1)[1]

        random_numbers = torch.rand_like(agent_input[:, 0])
        pick_random = (random_numbers < self.epsilon).long()
        picked_actions = pick_random * random_actions + (1 - pick_random) * selected_action

        return picked_actions