from typing import Dict, Any, Union, Tuple
from types import SimpleNamespace
from rl.datasets.transition_buffer import TransitionBuffer
from rl.datasets.buffer_info import BufferInfo


class ReplayBuffer(TransitionBuffer):
    """ Cycle buffer for storing trajectories sequentially"""
    def __init__(self,
                 config: SimpleNamespace,
                 buffer_info: BufferInfo,
                 buffer_shape: Tuple,
                 data: Dict = None):

        super(ReplayBuffer, self).__init__(config, buffer_info, buffer_shape, data)
        self.transition_index = 0 if data is None else self.buffer_size
        self.n_transitions = self.transition_index

    def append_from_other_buffer(self, other):
        """Append the other buffer's data"""
        buffer_left = self.buffer_size - self.transition_index
        if other.buffer_size <= buffer_left:
            self.update(other.data, slices=slice(self.transition_index, self.transition_index + other.buffer_size))
            self.increment_transition_index(offset=other.buffer_size)
        else:
            # self += other[:, 0:buffer_left]
            # self += other[:, buffer_left:]
            self += other[0:buffer_left]
            self += other[buffer_left:]
        return self

    def increment_transition_index(self, offset):
        self.transition_index += offset
        self.n_transitions = min(self.buffer_size, max(self.n_transitions, self.transition_index))

        # Circular buffer
        self.transition_index %= self.buffer_size
        assert self.transition_index < self.buffer_size

    def clear(self):
        super(ReplayBuffer, self).clear()
        self.transition_index = 0
        self.n_transitions = 0

    def __getitem__(self, item: Union[str, tuple, slice]):

        ret = super(ReplayBuffer, self).__getitem__(item)

        if isinstance(ret, TransitionBuffer):
            ret = ReplayBuffer(config=self.config,
                               buffer_info=ret.buffer_info,
                               buffer_shape=ret.buffer_shape,
                               data=ret.data)
        return ret

    def __len__(self):
        return self.n_transitions

    def __repr__(self):
        return "Replay Buffer. Buffer Size:{} Keys:{} ".format(self.buffer_size, self.buffer_info.scheme.keys())