import argparse
import gym
# import pybullet_envs

def run_gym(env_name, n_steps=100):

    env = gym.make(env_name, render_mode="human")

    env.action_space.seed(42)
    observation, info = env.reset(seed=42)

    for _ in range(n_steps):
        observation, reward, terminated, truncated, info = \
            env.step(env.action_space.sample())

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == '__main__':

    desc = 'RL Framework'
    parser = argparse.ArgumentParser(description=desc)
    # environment name
    parser.add_argument('-e',
                        '--env',
                        help='run type {CartPole-v1, AntBulletEnv-v0, LunarLanderContinuous-v2, ALE/Breakout-v5, CarRacing-v2}',
                        type=str,
                        default='CartPole-v1')

    parser.add_argument('-s',
                        '--steps',
                        help='Number of environment step executions',
                        type=int,
                        default=1000)

    args = parser.parse_args()
    print(args)

    # run gym
    run_gym(args.env, n_steps=args.steps)
