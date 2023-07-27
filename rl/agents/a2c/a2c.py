from types import SimpleNamespace
from rl.utils.logging import Logger
from rl.envs.environment import Environment
from rl.datasets.buffer_info import BufferInfo

from rl.agents.agent import Agent
from rl.agents.a2c.a2c_network import A2CNetwork
from rl.agents.a2c.a2c_learner import A2CLearner


class A2C(Agent):
    def __init__(self,
                 config: SimpleNamespace,
                 logger: Logger,
                 env: Environment,
                 buffer_info: BufferInfo,):

        super(A2C, self).__init__(
            config=config,
            logger=logger,
            env=env,
            buffer_info=buffer_info,
            network_class=A2CNetwork,
            learner_class=A2CLearner,
            policy_type="on_policy")