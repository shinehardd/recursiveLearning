import torch
from torch import nn
from types import SimpleNamespace
from rl.datasets.rollout_buffer import RolloutBuffer

from rl.agents.reinforce.reinforce_network import REINFORCENetwork
from rl.envs.environment import EnvironmentSpec
from rl.utils.logging import Logger
from rl.agents.base import Learner
from rl.utils.lr_scheduler import CosineLR
from rl.utils.value_util import monte_carlo_returns


class REINFORCELearner(Learner):
    def __init__(self,
                 config: SimpleNamespace,
                 logger: Logger,
                 environment_spec: EnvironmentSpec,
                 network: REINFORCENetwork,
                 buffer: RolloutBuffer):

        self.config = config
        self.logger = logger
        self.environment_spec = environment_spec
        self.b_continuous_action = self.environment_spec.b_continuous_action
        self.buffer = buffer
        self.network = network

        self.baseline = self.network.baseline

        self.optimizer = torch.optim.Adam([
                        {'params': self.network.policy.parameters(), 'lr': self.config.lr_policy},
                        {'params': self.network.baseline.parameters(), 'lr': self.config.lr_policy},
                    ])

        if self.config.lr_annealing:
            self.policy_lr_scheduler = CosineLR(logger=self.logger,
                                                param_groups=self.optimizer.param_groups[0],
                                                start_lr=self.config.lr_policy,
                                                end_timesteps=self.config.max_environment_steps,
                                                name="policy lr"
                                                )

            self.baseline_lr_scheduler = CosineLR(logger=self.logger,
                                                  param_groups=self.optimizer.param_groups[1],
                                                  start_lr=self.config.lr_policy,
                                                  end_timesteps=self.config.max_environment_steps,
                                                  name="policy lr"
                                                  )

            self.MSELoss = nn.MSELoss()

        self.learner_step = 0

    def _calc_returns(self):

        if len(self.buffer) == 0: return

        # calculate return
        returns, advantage = monte_carlo_returns(
            self.config,
            self.buffer['state'],
            self.buffer['next_state'],
            self.buffer['reward'],
            self.buffer['done'],
        )

        with torch.no_grad():
            baseline = self.baseline(self.buffer['state'])
            advantage = returns - baseline

        if self.buffer["returns"] is None:
            scheme = {'returns': {'shape': (1,)},
                      'advantage': {'shape': (1,)},
                      }
            self.buffer.extend_scheme(scheme)

        self.buffer['returns'] = returns
        self.buffer['advantage'] = advantage

    def update(self, total_n_timesteps: int, total_n_episodes: int):

        if len(self.buffer) == 0: return False

        # calculate mc return
        self._calc_returns()

        num_batch_times = (len(self.buffer)-1)//self.config.batch_size+1

        # Optimize policy for n epochs
        for epoch in range(0, self.config.n_epochs):
            for i in range(num_batch_times):
                sample_batched = self.buffer.sample(self.config.batch_size)

                state = sample_batched["state"]
                action = sample_batched["action"]
                returns = sample_batched["returns"]
                advantage = sample_batched["advantage"]

                self.learner_step += 1

                # Evaluating old actions and values
                log_probs, baseline = self.network(state, action)

                # baseline loss
                baseline_loss = self.MSELoss(baseline, returns)

                # policy loss
                policy_loss = -(log_probs * advantage).mean()

                total_loss = baseline_loss + policy_loss

                # take gradient step
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.grad_norm_clip
                )
                self.optimizer.step()

                # Performance Logging
                self.logger.log_stat("policy_loss", policy_loss.item(), self.learner_step)
                self.logger.log_stat("baseline_loss", policy_loss.item(), self.learner_step)

        if self.config.lr_annealing:
            self.policy_lr_scheduler.step(total_n_timesteps)
            self.baseline_lr_scheduler.step(total_n_timesteps)
            self.logger.log_stat("policy learning rate", self.optimizer.param_groups[0]['lr'], total_n_timesteps)
            self.logger.log_stat("baseline learning rate", self.optimizer.param_groups[1]['lr'], total_n_timesteps)

        # clear buffer
        self.buffer.clear()

        return True

    def save(self, checkpoint_path):
        # model
        self.network.save(checkpoint_path)
        # optimizer
        torch.save(self.optimizer.state_dict(), "{}/opt.th".format(checkpoint_path))

    def restore(self, checkpoint_path):
        # model
        self.network.restore(checkpoint_path)
        # optimizer
        self.optimizer.load_state_dict(
            torch.load("{}/opt.th".format(checkpoint_path), map_location=lambda storage, loc: storage))