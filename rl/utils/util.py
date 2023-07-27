"""util.py.

This file includes utility functions.
"""
import torch
import numpy as np

def to_torch_type(dtype):
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, int):
        return torch.int
    return torch.float32


def to_device(a_tensor, config):
    if a_tensor.device != config.device:
        a_tensor = a_tensor.to(config.device)
    return a_tensor


def to_numpy(a_tensor, config):
    """Type conversion of tensor to numpy accoding to cuda use.
    """
    if config.use_cuda: return a_tensor.cpu().data.numpy()
    return a_tensor.data.numpy()


def to_tensor(data, dtype=torch.float32):
    """Type conversion of numpy array to tensor
    """
    if isinstance(data, (list, tuple)):  data = np.array(data)
    if not isinstance(data , (np.ndarray, int, float)): return data
    return torch.tensor(data, dtype=dtype)


def scale_bias(high, low):
    scale = (high - low) / 2.
    bias = (high + low) / 2.
    return scale, bias


def soft_update(source, target, tau):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(target_param.data*(1.0-tau) + source_param.data*tau)


def hard_update(source, target):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(source_param.data)