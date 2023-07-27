import torch
import torch.nn as nn
import math
import numpy as np
from typing import Tuple, Dict
from types import SimpleNamespace

from rl.datasets.rollout_buffer import RolloutBuffer
from rl.agents.ppo.ppo_network import PPONetwork
from rl.envs.environment import EnvironmentSpec
from rl.utils.logging import Logger
from rl.agents.base import Learner
from rl.utils.lr_scheduler import CosineLR
from rl.utils.value_util import REGISTRY as RETURN_REGISTRY
from rl.utils.schduler import LinearScheduler


class PPOLearner(Learner):
    def __init__(self,
                 config: SimpleNamespace,
                 logger: Logger,
                 environment_spec: EnvironmentSpec,
                 network: PPONetwork,
                 buffer: RolloutBuffer):

        self.config = config
        self.logger = logger
        self.environment_spec = environment_spec
        self.b_continuous_action = self.environment_spec.b_continuous_action
        self.buffer = buffer
        self.network = network
        self.policy = self.network.policy
        self.critic = self.network.critic

        self.optimizer = torch.optim.Adam([
                        {'params': self.network.policy.parameters(), 'lr': self.config.lr_policy},
                        {'params': self.network.critic.parameters(), 'lr': self.config.lr_critic}
                    ])

        if self.config.lr_annealing:
            self.policy_lr_scheduler = CosineLR(logger=self.logger,
                                                param_groups=self.optimizer.param_groups[0],
                                                start_lr=self.config.lr_policy,
                                                end_timesteps=self.config.max_environment_steps,
                                                name="policy lr"
                                                )
            self.critic_lr_scheduler = CosineLR(logger=self.logger,
                                                param_groups=self.optimizer.param_groups[1],
                                                start_lr=self.config.lr_critic,
                                                end_timesteps=self.config.max_environment_steps,
                                                name="critic lr"
                                                )

        self.MSELoss = nn.MSELoss()

        end_timesteps = self.config.max_environment_steps if self.config.clip_schedule else -1

        self.clip_scheduler = LinearScheduler(start_value=self.config.ppo_clipping_epsilon,
                                              start_timesteps=1,
                                              end_timesteps=end_timesteps)
        self.learner_step = 0

    def _loss(self,
              states: torch.FloatTensor,
              actions: torch.FloatTensor,
              target_values: torch.FloatTensor,
              log_probs_old: torch.FloatTensor,
              advantages: torch.FloatTensor=None,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:

        # Evaluating old actions and values
        log_probs, entropy, values = self.network(states, actions)

        ratios = torch.exp(log_probs - log_probs_old)

        assert not torch.any(ratios == math.inf)

        # Finding the ratio
        ratios = ratios.prod(1, keepdim=True)

        # policy loss
        surrogate1 = # your code
        surrogate2 = # your code

        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # value loss
        value_loss = self.MSELoss(values, target_values)

        # entropy loss
        entropy_loss = -entropy.mean()

        # final loss of clipped objective PPO
        total_loss = (
                policy_loss
                + self.config.vloss_coef * value_loss
                + self.config.eloss_coef * entropy_loss
        )

        return total_loss, {
          'total_loss': total_loss.item(),
          'policy_loss': policy_loss.item(),
          'value_loss': value_loss.item(),
          'entropy_loss': entropy_loss.item(),
        }

    def _calc_target_value(self):
        if len(self.buffer) == 0: return

        target_value, advantage = RETURN_REGISTRY[self.config.advantage_type](
            self.config,
            self.buffer['state'],
            self.buffer['next_state'],
            self.buffer['reward'],
            self.buffer['done'],
            self.critic
        )

        # make input tensor
        state = self.buffer['state']
        action = self.buffer['action']

        # calculate objective ratio
        with torch.no_grad():
            log_probs_old, _, _ = self.network(state, action)

        if self.buffer["advantage"] is None:
            scheme = {
                'advantage': {'shape': (1,)},
                'target_value': {'shape': (1,)},
                'log_probs_old': {'shape': (log_probs_old.shape[-1],),},
            }
            self.buffer.extend_scheme(scheme)

        self.buffer['advantage'] = advantage
        self.buffer['target_value'] = target_value
        self.buffer['log_probs_old'] = log_probs_old

    def update(self, total_n_timesteps: int, total_n_episodes: int):

        if len(self.buffer) == 0: return False

        # calculate gae or mc return
        self._calc_target_value()
        self.clipping_epsilon = self.clip_scheduler.value(total_n_timesteps)

        num_batch_times = (len(self.buffer)-1)//self.config.batch_size+1

        # Optimize policy for K epochs
        for epoch in range(0, self.config.n_epochs):
            for i in range(num_batch_times):
                sample_batched = self.buffer.sample(self.config.batch_size)

                state = sample_batched["state"]
                action = sample_batched["action"]
                advantage = sample_batched["advantage"]
                target_value = sample_batched["target_value"]
                log_probs_old = sample_batched["log_probs_old"]

                self.learner_step += 1
                total_loss, loss_results = self._loss(state, action, target_value, log_probs_old, advantage)

                # take gradient step
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.grad_norm_clip
                )
                self.optimizer.step()

                # Performance Logging
                self.logger.log_stat("total_loss", loss_results['total_loss'], self.learner_step)
                self.logger.log_stat("policy_loss", loss_results['policy_loss'], self.learner_step)
                self.logger.log_stat("value_loss", loss_results['value_loss'], self.learner_step)
                self.logger.log_stat("entropy_loss", loss_results['entropy_loss'], self.learner_step)

        if self.config.lr_annealing:
            self.policy_lr_scheduler.step(total_n_timesteps)
            self.critic_lr_scheduler.step(total_n_timesteps)
            self.logger.log_stat("policy learning rate", self.optimizer.param_groups[0]['lr'], total_n_timesteps)
            self.logger.log_stat("critic learning rate", self.optimizer.param_groups[1]['lr'], total_n_timesteps)

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
