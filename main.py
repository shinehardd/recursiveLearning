import torch
import numpy as np
import random
import rl.utils.config as cu
import argparse
from rl.runner.runner import Runner

if __name__ == '__main__':

    desc = 'RL Framework'
    parser = argparse.ArgumentParser(description=desc)
    # agent
    parser.add_argument('-a',
                        '--agent',
                        help='agent name {reinforce, reinforce_baseline, a2c, ppo, dqn, ddqn}',
                        type=str,
                        default='reinforce')

    # environment name
    parser.add_argument('-e',
                        '--env',
                        help='run type {CartPole-v1, Acrobot-v1, LunarLander-v2, LunarLanderContinuous-v2}',
                        type=str,
                        default='CartPole-v1')

    args = parser.parse_args()

    # random number initialization
    random_seed = random.randrange(0, 16546)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print("random_seed=", random_seed)

    agent_name = args.agent
    env_name = args.env

    # read configuration file
    config: dict = cu.config_copy(cu.get_config(agent_name, env_name))
    config['agent'] = agent_name
    config['env_name'] = env_name
    config['random_seed'] = random_seed  # add random seed into config object
    if config.get('env_args', None) is None:
        config['env_args'] = {}

    # run model
    Runner(config).run()