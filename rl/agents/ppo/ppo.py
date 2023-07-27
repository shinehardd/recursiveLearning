from types import SimpleNamespace
from rl.utils.logging import Logger
from rl.envs.environment import Environment
from rl.datasets.buffer_info import BufferInfo

from rl.agents.agent import Agent
from rl.agents.ppo.ppo_network import PPONetwork
from rl.agents.ppo.ppo_learner import PPOLearner


class PPO(Agent):
    def __init__(self,
                 config: SimpleNamespace,
                 logger: Logger,
                 env: Environment,
                 buffer_info: BufferInfo,):

        super(PPO, self).__init__(
            config=config,
            logger=logger,
            env=env,
            buffer_info=buffer_info,
            network_class=PPONetwork,
            learner_class=PPOLearner,
            policy_type="on_policy")