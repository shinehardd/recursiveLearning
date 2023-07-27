import torch
import numpy as np
import copy
from types import SimpleNamespace
from typing import Dict, Any, Union, Tuple
from rl.utils.util import to_tensor, to_device
from rl.datasets.buffer_info import BufferInfo

class Buffer():
    """Tensor Type Buffer. It has memory array but, don't have rollout or episode counter"""

    def __init__(self,
                 config: SimpleNamespace,
                 buffer_info: BufferInfo,
                 buffer_shape: Tuple[int],
                 data: Dict = None):

        assert config
        assert buffer_info
        assert buffer_shape

        self.config = config
        self.buffer_info = copy.deepcopy(buffer_info)
        self.buffer_shape = buffer_shape
        self.buffer_size = self.buffer_shape[0]
        self.data = data

        if self.data is None:
            self._setup_data(self.buffer_info, self.buffer_shape)

    def __len__(self):
        """return length"""

    def is_full(self):
        return len(self) == self.buffer_size

    def clear(self):
        for key in self.data.keys():
            self.data[key][:, :] = 0

    def _setup_data(self,
                    buffer_info: BufferInfo,
                    buffer_shape: Tuple[int]):

        assert buffer_info
        assert buffer_shape
        self.data = self._create_buffer_from_schema(buffer_info.scheme, buffer_shape)

    def _create_buffer_from_schema(self,
                                   scheme: Dict[str, Any],
                                   buffer_shape: Tuple[int],):

        assert scheme
        assert buffer_shape

        # Allocate new data space
        data = {}

        # Make keys from the scheme
        for key, info in scheme.items():
            assert "shape" in info, "Scheme must define shape for {}".format(key)

            data_shape, dtype = info["shape"], info.get("dtype", torch.float32)
            if len(data_shape) == 0: data_shape = 1   # convert scalar to 1 dim tensor
            if isinstance(data_shape, int): data_shape = (data_shape,)

            # alloc buffer
            data[key] = self._alloc_buffer(buffer_shape, data_shape, dtype)

        return data

    def _alloc_buffer(self,
                      buffer_shape: Tuple[int],
                      data_shape: tuple,
                      dtype: torch.tensor):

        return to_device(torch.zeros((*buffer_shape, *data_shape), dtype=dtype), self.config)

    def _check_safe_view(self, src: Any, dst: Any):
        """
            Check the value has same view as the dest.
            For example, if src_tensor = [1,2,3], dst_tensor = [1,1,2,3],
            the dimension of dest with size = 1 may not be exist in the value.
        """

        # Check from the last dimension to the first dimension.
        idx = len(src.shape) - 1
        for dst_size in dst.shape[::-1]:
            # Source index must be positive
            src_size = src.shape[idx] if idx >= 0 else 0

            # When the dimension size  of value and dest is different
            if src_size != dst_size:
                # if the size of dest dimension is 1, pass
                if dst_size != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(src.shape, dst.shape))

            # if the size of source dimension and the size of dest dimension is same, move to prev dim
            else:
                idx -= 1

    def _parse_slices(self, slides: Union[int, list, slice, tuple]) -> Tuple[slice, slice]:
        '''
            arguments
                slides : (batch slice, time slice)
                        batch slice and time slice can be a index slice, a index, a index list.
            returns
                (batch slice, time slice)
        '''

        # Only batch slice given, add full time slice 이것을 왜 넣지?
        if (isinstance(slides, slice)  # slice a:b
           or isinstance(slides, int)  # int i
           or (isinstance(slides, (list, np.ndarray, torch.LongTensor, torch.cuda.LongTensor)))):  # [a,b,c]
            slides = (slides, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(slides[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        parsed = []
        for a_slice in slides:
            if isinstance(a_slice, int):   # Convert single indices to slices
                parsed.append(slice(a_slice, a_slice+1))
            else:   # Leave slices and lists as is
                parsed.append(a_slice)
        return parsed

    def update(self, data: Dict, slices):
        """
            Update buffer slices with data
        """
        slices = self._parse_slices((slices))

        for key, value in data.items():

            # 1. Key validation and make slice
            if key not in self.data:
                raise KeyError("{} not found in data".format(key))

            # 2. Transform value to tensor
            dtype = self.buffer_info.scheme[key].get("dtype", torch.float32)
            value = to_device(to_tensor(value, dtype=dtype), self.config)
            if len(value.shape) == 0: value = value.view((1,))

            # 3. Reshape value according to target shape
            self._check_safe_view(value, self.data[key][slices])
            self.data[key][slices] = value.view_as(self.data[key][slices])  # make input data as target view

    def extend_scheme(self, scheme: Dict[str, Any]):
        self.data.update(self._create_buffer_from_schema(scheme, self.buffer_shape))
        self.buffer_info.scheme.update(scheme)

    def _get_num_items(self, item: Union[list, np.ndarray, slice], max_size: int):
        """ check item size from numpy array size or slice information"""

        if isinstance(item, (list, np.ndarray)):
            return len(item)

        if isinstance(item, slice):
            _range = item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _same_shape(self, this_shape, that_shape):
        """ check if two shape is same"""
        if all((i == j for i, j in zip(this_shape, that_shape))): return True
        return False