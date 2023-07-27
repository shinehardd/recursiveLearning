import torch
import numpy as np
from typing import Dict, Any, Union, Tuple
from types import SimpleNamespace
from rl.datasets.buffer import Buffer
from rl.datasets.buffer_info import BufferInfo

class TransitionBuffer(Buffer):
    """ 1D Tensor Type Buffer. It has transition data"""

    def __init__(self,
                 config: SimpleNamespace,
                 buffer_info: BufferInfo,
                 buffer_shape: Tuple,
                 data: Dict = None):

        super(TransitionBuffer, self).__init__(
            config=config,
            buffer_info=buffer_info,
            buffer_shape=buffer_shape,
            data=data)

    """
    def update(self, data: Dict, time_slice):

        slices = self._parse_slices((time_slice))
        super().update(data, slices)
    """
    def __getitem__(self, item: Union[str, tuple, slice]):
        # Single key
        if isinstance(item, str):
            if item in self.data:
                return self.data[item]
            else:
                return None

        # Slice
        slices = self._parse_slices(item)
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = v[slices]

        buffer_shape = [self._get_num_items(slices[0], self.buffer_size)]

        ret = TransitionBuffer(config=self.config,
                               buffer_info=self.buffer_info,
                               buffer_shape=buffer_shape,
                               data=new_data)
        return ret

    def __setitem__(self, item: Union[str, tuple, slice], value: Union[torch.tensor, Any]):
        # Single key
        if isinstance(item, str):
            if item in self.data:
                if not self._same_shape(self.data[item].shape, value.shape):
                    raise IndexError("Data shape {} is not proper to buffer shape {}".format(value.shape,
                                                                                             self.data[item].shape))
                self.data[item] = value
                return
            else:   # prevent direct buffer creation
                raise ValueError

        # Slice
        self.update(value.data, slice=item)

    def can_sample(self, batch_size):
        return len(self) >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        # When buffer has minimum data
        if len(self) == batch_size: return self[:batch_size]

        # Uniform sampling only atm
        time_ids = np.random.choice(len(self), batch_size, replace=False)
        return self[time_ids]

    def __repr__(self):
        return "Transition Buffer. Buffer Size:{} Keys:{} ".format(self.buffer_size, self.buffer_info.scheme.keys())

    def append_one_transition(self, data: Dict):
        """append one transition"""
        self.update(data, slices=len(self))
        self.increment_transition_index(offset=1)

        return self

    def append_from_other_buffer(self, other: Buffer):
        return self

    def __add__(self, other: Union[Buffer, Dict]):
        """ 1. Append one transition"""
        if isinstance(other, Dict):
            return self.append_one_transition(other)

        """ 2. Append the other buffer's data"""
        return self.append_from_other_buffer(other)