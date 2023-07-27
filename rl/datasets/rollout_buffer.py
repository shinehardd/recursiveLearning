import numpy as np
from typing import Dict, Any, Union, Tuple
from types import SimpleNamespace
from rl.datasets.transition_buffer import TransitionBuffer
from rl.datasets.buffer_info import BufferInfo


class RolloutBuffer(TransitionBuffer):
    """ Buffer for storing trajectories sequentially"""
    def __init__(self,
                 config: SimpleNamespace,
                 buffer_info: BufferInfo,
                 buffer_shape: Tuple,
                 data: Dict = None):

        super(RolloutBuffer, self).__init__(
            config=config,
            buffer_info=buffer_info,
            buffer_shape=buffer_shape,
            data=data)

        self.transition_index = 0 if data is None else self.buffer_size

    def append_from_other_buffer(self, other: TransitionBuffer):
        """append the other buffer's data after the last transition data"""
        if self.is_full(): return self

        if other.buffer_size != len(other):
            other = other[:len(other)]
        self.update(other.data, slices=slice(self.transition_index, self.transition_index + other.buffer_size))
        self.increment_transition_index(offset=other.buffer_size)
        return self

    def increment_transition_index(self, offset):
        self.transition_index += offset
        assert self.transition_index <= self.buffer_size

    def sample(self, batch_size=-1, allow_remainder: bool = True):
        # Returns whole data
        if batch_size == -1:
            if len(self) != self.buffer_size:
                return self[:len(self)]
            return self

        # Returns even if the data size is less than the batch size
        if allow_remainder and len(self) < batch_size:
            batch_size = len(self)

        assert self.can_sample(batch_size)

        # When buffer has minimum data
        if len(self) == batch_size: return self[:batch_size]

        # Uniform sampling only atm
        time_ids = np.random.choice(len(self), batch_size, replace=False)
        return self[time_ids]

    def clear(self):
        super(RolloutBuffer, self).clear()
        self.transition_index = 0

    def __getitem__(self, item: Union[str, tuple, slice]):

        ret = super(RolloutBuffer, self).__getitem__(item)

        if isinstance(ret, TransitionBuffer):
            ret = RolloutBuffer(config=self.config,
                                buffer_info=ret.buffer_info,
                                buffer_shape=ret.buffer_shape,
                                data=ret.data)
        return ret

    def __len__(self):
        return self.transition_index

    def __repr__(self):
        return "RolloutBuffer. Batch Size:{} Keys:{}".format(self.buffer_size, self.buffer_info.scheme.keys())