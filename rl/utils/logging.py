import torch
import numpy as np
from collections import defaultdict
import logging
import os
from os.path import dirname, abspath

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False

        self.stats = defaultdict(lambda: [])

    def setup_tensorboard(self, dirpath):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(dirpath)
        self.tb_logger = log_value
        self.use_tb = True

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))
        if self.use_tb: self.tb_logger(key, value, t)

    def print_recent_stats(self, item_per_line:int=4):
        log_str = "Steps: {:>8} | Episode: {:>6}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode": continue
            i += 1
            window = 5 if k != "epsilon" else 1
            stats_list = [x[1].cpu() if isinstance(x[1], torch.Tensor) else x[1] for x in self.stats[k][-window:]]
            item = "{:.6f}".format(np.mean(stats_list))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % item_per_line == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_console_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

def get_logger(use_tensorboard, model_name, unique_token):
    console_logger = get_console_logger()
    logger = Logger(console_logger)
    # tensorboard self.logger
    if use_tensorboard:
        tb_logs_dir = os.path.join(
            dirname(abspath(__file__)),
            "..",
            "results",
            "tb_logs",
            model_name,
            unique_token,
        )
        logger.setup_tensorboard(tb_logs_dir)
    return console_logger, logger