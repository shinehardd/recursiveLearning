import abc
import torch
from types import SimpleNamespace
from typing import Dict, Any
from rl.envs.environment import Environment
from rl.utils.util import to_torch_type


class BufferInfo():
    def __init__(self,
                 config: SimpleNamespace,
                 env: Environment,
                 scheme: Dict[str, Any] = None):

        self.config = config
        self.env = env
        self.scheme = self.create() if scheme is None else scheme


    def create(self):
        env_spec = self.env.environment_spec()
        scheme = {
            "state": {"shape": env_spec.state_spec.shape},
            "action": {"shape": env_spec.action_spec.shape, "dtype": to_torch_type(env_spec.action_spec.dtype)},
            "next_state": {"shape": env_spec.state_spec.shape},
            "reward": {"shape": env_spec.reward_spec.shape},
            "done": {"shape": (1,), "dtype": to_torch_type(int)},
        }

        return scheme