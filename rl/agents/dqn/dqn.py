from types import SimpleNamespace
from rl.utils.logging import Logger
from rl.envs.environment import Environment
from rl.datasets.buffer_info import BufferInfo

from rl.agents.agent import Agent
from rl.agents.dqn.dqn_network import DQNNetwork
from rl.agents.dqn.dqn_learner import DQNLearner


class DQN(Agent):

    def __init__(self,
                 config: SimpleNamespace,
                 logger: Logger,
                 env: Environment,
                 buffer_info: BufferInfo,):

        if env.environment_spec().b_continuous_action:
            raise Exception("DQN doesn't support continuous action space")

        super(DQN, self).__init__(
            config=config,
            logger=logger,
            env=env,
            buffer_info=buffer_info,
            network_class=DQNNetwork,
            learner_class=DQNLearner,
            policy_type="off_policy")