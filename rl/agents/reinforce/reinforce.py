from types import SimpleNamespace
from rl.utils.logging import Logger
from rl.envs.environment import Environment
from rl.datasets.buffer_info import BufferInfo

from rl.agents.agent import Agent
from rl.agents.reinforce.reinforce_network import REINFORCENetwork
from rl.agents.reinforce.reinforce_learner import REINFORCELearner


class REINFORCE(Agent):
    def __init__(self,
                 config: SimpleNamespace,
                 logger: Logger,
                 env: Environment,
                 buffer_info: BufferInfo,):

        super(REINFORCE, self).__init__(
            config=config,
            logger=logger,
            env=env,
            buffer_info=buffer_info,
            network_class=REINFORCENetwork,
            learner_class=REINFORCELearner,
            policy_type="on_policy")