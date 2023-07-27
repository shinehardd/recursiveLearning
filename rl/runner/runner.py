import abc
import os
import pprint
import datetime
import time
import torch

from types import SimpleNamespace
from rl.utils.logging import get_console_logger
from rl.utils.logging import Logger
from rl.utils.config import save_config

from rl.agents import REGISTRY as agent_REGISTRY
from rl.envs import REGISTRY as env_REGISTRY
from rl.datasets.buffer_info import BufferInfo
from rl.runner.environment_loop import EnvironmentLoop
from rl.utils.timehelper import time_left, time_str

class Runner:
    def __init__(self,
                 config: dict,
                 console_logger: Logger = None,
                 logger: Logger = None,
                 verbose: bool = False):
        # Console Logger
        if console_logger is None:
            self.console_logger = get_console_logger()

        # Create self.config from config
        config = self._sanity_check_config(config)
        self.config = SimpleNamespace(**config)
        self.config.device = (
           "cuda:{}".format(self.config.device_num) if self.config.use_cuda else "cpu"
        )

        # Unique Token
        unique_token = "{}_{}_{}".format(
            self.config.agent, self.config.env_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        self.config.unique_token = unique_token

        # logger
        if logger is None:
            logger = Logger(self.console_logger)

            # tensorboard self.logger
            if self.config.use_tensorboard:
                tb_logs_dir = os.path.join(
                    os.getcwd(),
                    self.config.local_results_path,
                    "tb_logs",
                    unique_token,
                )
                logger.setup_tensorboard(tb_logs_dir)

        self.logger = logger

        # Print experiment parameters
        if verbose:
            self.logger.console_logger.info("Experiment Parameters:")
            experiment_params = pprint.pformat(config, indent=4, width=1)
            self.logger.console_logger.info("\n\n" + experiment_params + "\n")

        torch.backends.cudnn.deterministic = self.config.torch_deterministic

        if self.config.training_mode and self.config.save_model:
            save_config(self.config)

        self.total_n_timesteps = 0
        self.total_n_episodes = 0

    def _sanity_check_config(self, config):

        # set CUDA flags
        if config["use_cuda"] and not torch.cuda.is_available():
            config["use_cuda"] = False
            warning_msg = "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
            self.console_logger.warning(warning_msg)

        return config

    def run(self):

        self.env = env_REGISTRY[self.config.env_wrapper](self.config, **self.config.env_args)
        self.buffer_info = BufferInfo(self.config, self.env)

        # Agent
        self.agent = agent_REGISTRY[self.config.agent](
            config=self.config,
            logger=self.logger,
            env=self.env,
            buffer_info=self.buffer_info,
        )

        # Cuda
        if self.config.use_cuda: self.agent.cuda()

        if self.config.training_mode:
            # Training Mode
            # Restore the checkpoint
            if self.config.checkpoint_path != "" and self.restore() is False: return False
            self.train()
        else:
            # Inference Mode
            # Restore the trained model
            if self.load() is False: return False
            self.test()

        self.env.close()
        return True

    def train(self):

        self.logger.console_logger.info("training environment name : " + self.config.env_name)
        last_model_save_timestep = 0
        last_logging_step = 0

        # track total training time
        start_time = datetime.datetime.now().replace(microsecond=0)

        environment_loop = EnvironmentLoop(
            config=self.config,
            logger=self.logger,
            actor=self.agent.actor,
            env=self.env)

        # training loop
        while self.total_n_timesteps < self.config.max_environment_steps:

            result = environment_loop.run(max_n_timesteps=self.config.n_steps,
                                          max_n_episodes=self.config.n_episodes)

            # logging message
            self.total_n_timesteps = result['total_n_timesteps']
            self.total_n_episodes = result['total_n_episodes']

            # update buffer
            self.agent.add_rollouts(result['rollouts'])

            # train
            if self.total_n_timesteps >= self.config.warmup_step:
                self.agent.update(self.total_n_timesteps, self.total_n_episodes)

            # clear actor's local buffer
            self.agent.actor.clear_rollouts()

            logging_msg = f"timesteps: {self.total_n_timesteps}, "
            for key, value in result['stats'].items():
                if key == 'n_episodes': continue
                mean_value = value / result['stats']['n_episodes']
                logging_msg += f"{key}: {mean_value:.4f}, "
                self.logger.log_stat(f"{key}_mean", mean_value, self.total_n_timesteps)

            # logging stats
            if (self.total_n_timesteps - last_logging_step) >= self.config.log_interval:

                environment_loop.reset_stats()

                # update last logging time step
                last_logging_step = self.total_n_timesteps
                self.logger.log_stat("episode", self.total_n_episodes, self.total_n_timesteps)
                self.logger.print_recent_stats()

            # checkpoint
            if self.config.save_model and \
                    ((self.total_n_timesteps - last_model_save_timestep) >= self.config.save_model_interval):
                # learner should handle saving/loading
                self.save(self.total_n_timesteps)
                last_model_save_timestep = self.total_n_timesteps

        # print total training time
        end_time = datetime.datetime.now().replace(microsecond=0)
        self.logger.console_logger.info("Started training at (GMT) : {} ".format(start_time))
        self.logger.console_logger.info("Finished training at (GMT) : {} ".format(end_time))
        self.logger.console_logger.info("Total training time  : {} ".format(end_time - start_time))

    def test(self):

        self.logger.console_logger.info("environment name : " + self.config.env_name)

        environment_loop = EnvironmentLoop(
            config=self.config,
            logger=self.logger,
            actor=self.agent.actor,
            env=self.env)

        result = environment_loop.run(max_n_episodes=self.config.test_mode_max_episodes)
        self.logger.console_logger.info("Result: {} ".format(result))

    def load(self):
        self.agent.load(self.config.trained_model_path)
        return True

    def restore(self):
        timesteps = []

        # if checkpoint path is invalid
        if not os.path.isdir(self.config.checkpoint_path):
            self.logger.console_logger.info(
                "Checkpoint directory {} doesn't exist".format(
                    self.config.checkpoint_path
                )
            )
            return False

        # Go through all files in self.config.checkpoint_path
        for name in os.listdir(self.config.checkpoint_path):
            full_name = os.path.join(self.config.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if self.config.load_step == 0:
            # choose the max timestep (last saved step)
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(
                timesteps, key=lambda x: abs(x - self.config.load_step)
            )

        model_path = os.path.join(self.config.checkpoint_path, str(timestep_to_load))
        self.total_time_step = timestep_to_load

        self.logger.console_logger.info("Loading model from {}".format(model_path))
        self.agent.restore(model_path)
        self.agent.update(self.total_time_step, force_update=True)

        return True

    def save(self, time_step):
        # make directory for model saving
        checkpoint_path = os.path.join(
            os.getcwd(),
            self.config.local_results_path,
            "models",
            self.config.unique_token,
            str(time_step),
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        self.logger.console_logger.info("Saving models to {}".format(checkpoint_path))

        if self.agent is not None:
            self.agent.save(checkpoint_path)