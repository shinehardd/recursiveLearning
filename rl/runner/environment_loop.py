from types import SimpleNamespace
from collections import defaultdict
from rl.utils.logging import Logger
from rl.agents.actor import Actor
from rl.envs.environment import Environment

class EnvironmentLoop:

    """ Environment loop class has role of collecting trajectory data
        from the interaction of an actor and an environment.
        It has a data collect type
            1) Collect fixed-length trajectory data
    """


    def __init__(self,
                 config: SimpleNamespace,
                 logger: Logger,
                 actor: Actor,
                 env: Environment):

        self.config = config
        self.logger = logger
        self.actor = actor
        self.env = env

        # initialize global counters and stats
        self.total_n_timesteps = 0
        self.total_n_episodes = 0
        self.init_stats()

        # initialize episode
        self.reset_episode(seed=self.config.random_seed)
        # render always in test mode
        self.b_render = self.config.render if self.config.training_mode else True

    def run(self, max_n_timesteps: int = 0, max_n_episodes: int = 0):
        """ Running environment loop according to
        Arguments
            max_n_timesteps : the length of trajectory data to collect
            max_n_episodes : # of episode of trajectory data to collect
        """

        if max_n_timesteps: max_n_episodes = 0
        if max_n_episodes: self.reset_episode()

        self.init_run()
        # interaction loop
        while self.n_timesteps < max_n_timesteps or self.n_episodes < max_n_episodes:

            pre_transition_data = self.pre_transition_data()

            # select action with policy
            action = self.select_action()

            # do action in the environment
            next_state, reward, done, truncated, env_info = self.env.step(action)
            done = done or truncated

            # save rollout (train mode)
            post_transition_data = self.post_transition_data(action, reward, next_state, done)
            transition_data = {**pre_transition_data, **post_transition_data}
            self.actor.observe(transition_data)

            # move to next step
            self.state = next_state

            # episode return
            self.return_of_an_episode += reward

            # increment global, run and episode related timestep counters
            self.total_n_timesteps += 1             # global counter
            self.n_timesteps += 1                   # counter per run
            self.n_timesteps_of_an_episode += 1     # counter per episode

            # break; if the episode is over
            if done:
                # increment episode related stats count
                self.total_n_episodes += 1          # global counter
                self.n_episodes += 1                # counter per run

                # update stats
                self.update_stats()

                # reset episode
                self.reset_episode()

        return self.final_result()

    def init_stats(self):
        self.stats = defaultdict(float)
        self.result = None

    def reset_stats(self):
        self.stats.clear()

    def update_stats(self):
        self.stats['n_episodes'] += 1
        self.stats['return'] += self.return_of_an_episode
        self.stats['len_episodes'] += self.n_timesteps_of_an_episode

    def final_result(self):
        result = {
            'rollouts': self.actor.rollouts(),
            'total_n_timesteps': self.total_n_timesteps,
            'total_n_episodes': self.total_n_episodes,
            'stats': self.stats
        }
        return result

    def init_run(self):
        # initialize run counters
        self.n_timesteps = 0
        self.n_episodes = 0

    def reset_episode(self, seed=None):
        self.state, _ = self.env.reset(seed=seed)    # return state, info
        self.actor.reset()
        self.return_of_an_episode = 0
        self.n_timesteps_of_an_episode = 0

    def select_action(self):
        # within a warming up period
        if self.total_n_timesteps < self.config.warmup_step:
            return self.env.select_action()

        # determine actor's action
        action = self.actor.select_action(self.state, self.total_n_timesteps)

        return action

    def pre_transition_data(self):
        pre_transition_data = {
            "state": self.state,
        }

        return pre_transition_data

    def post_transition_data(self, action, reward, next_state, done):
        post_transition_data = {
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        }
        return post_transition_data
