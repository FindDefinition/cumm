# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Type
from typing import Dict, List, Optional, Tuple, Union, overload
import builtins

import numpy as np


class CUDAKernelTimer:
    def __init__(self, enable: bool) -> None:
        ...

    def push(self, name: str) -> None:
        ...

    def pop(self) -> None:
        ...

    def record(self, name: str, stream: int = 0) -> None:
        ...

    def insert_pair(self, name: str, start: str, stop: str) -> None:
        ...

    def has_pair(self, name: str) -> bool:
        ...

    def sync_all_event(self) -> None:
        ...

    def get_all_pair_duration(self) -> Dict[str, float]:
        ...

    @property
    def enable(self) -> bool:
        ...


class Tensor:
    @overload
    def __init__(self):
        ...

    @overload
    def __init__(self,
                 shape: Union[List[int], Tuple[int]],
                 dtype: int = 0,
                 device: int = -1,
                 pinned: bool = False,
                 managed: bool = False):
        ...

    @property
    def shape(self) -> List[int]:
        ...

    @property
    def stride(self) -> List[int]:
        ...

    @property
    def dtype(self) -> int:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def itemsize(self) -> int:
        ...

    @property
    def ndim(self) -> int:
        ...

    @property
    def device(self) -> int:
        ...

    def pinned(self) -> bool:
        ...

    def is_contiguous(self) -> bool:
        ...

    def byte_offset(self) -> int:
        ...

    def empty(self) -> bool:
        ...

    def dim(self, axis: int) -> int:
        ...

    def slice_first_axis(self, start: int, end: int) -> "Tensor":
        ...

    def view(self, views: List[int]) -> "Tensor":
        ...

    def clone(self,
              pinned: bool = False,
              use_cpu_copy: bool = False) -> "Tensor":
        ...

    def clone_whole_storage(self) -> "Tensor":
        ...

    def zero_whole_storage_(self) -> None:
        ...

    def unsqueeze(self, axis: int) -> "Tensor":
        ...

    @overload
    def squeeze(self) -> "Tensor":
        ...

    @overload
    def squeeze(self, axis: int) -> "Tensor":
        ...

    @overload
    def __getitem__(self, idx: int) -> "Tensor":
        ...

    @overload
    def __getitem__(self, idx: slice) -> "Tensor":
        ...

    @overload
    def __getitem__(
        self, idx: Tuple[Union[int, None, slice, builtins.ellipsis],
                         ...]) -> "Tensor":
        ...

    def as_strided(self, shape: List[int], stride: List[int],
                   storage_byte_offset: int) -> "Tensor":
        ...

    def slice_axis(self,
                   dim: int,
                   start: Optional[int],
                   stop: Optional[int],
                   step: Optional[int] = None) -> "Tensor":
        ...

    def select(self, dim: int, index: int) -> "Tensor":
        ...

    def numpy(self) -> np.ndarray:
        ...

    def numpy_view(self) -> np.ndarray:
        ...

    @overload
    def cpu(self) -> "Tensor":
        ...

    @overload
    def cpu(self, stream_handle: int) -> "Tensor":
        ...

    @overload
    def copy_(self, other: "Tensor") -> None:
        ...

    @overload
    def copy_(self, other: "Tensor", stream_handle: int) -> None:
        ...

    @overload
    def zero_(self) -> "Tensor":
        ...

    @overload
    def zero_(self, stream_handle: int) -> "Tensor":
        ...

    @overload
    def cuda(self) -> "Tensor":
        ...

    @overload
    def cuda(self, stream_handle: int) -> "Tensor":
        ...

    @overload
    def fill_int_(self, val: Union[int, float]) -> "Tensor":
        ...

    @overload
    def fill_int_(self, val: Union[int, float],
                  stream_handle: int) -> "Tensor":
        ...

    @overload
    def fill_float_(self, val: Union[int, float]) -> "Tensor":
        ...

    @overload
    def fill_float_(self, val: Union[int, float],
                    stream_handle: int) -> "Tensor":
        ...

    def byte_pointer(self) -> int:
        ...


def zeros(shape: List[int],
          dtype: Union[np.dtype, int] = np.float32,
          device: int = -1,
          pinned: bool = False,
          managed: bool = False) -> Tensor:
    ...

@overload
def from_blob(ptr: int,
                      shape: List[int],
                      stride: List[int],
                      dtype: Union[np.dtype, int] = np.float32,
                      device: int = -1) -> Tensor:
    ...

@overload
def from_const_blob(ptr: int,
                            shape: List[int],
                            stride: List[int],
                            dtype: Union[np.dtype, int] = np.float32,
                            device: int = -1) -> Tensor:
    ...

@overload
def from_blob(ptr: int,
              shape: List[int],
              dtype: Union[np.dtype, int] = np.float32,
              device: int = -1) -> Tensor:
    ...

@overload
def from_const_blob(ptr: int,
                    shape: List[int],
                    dtype: Union[np.dtype, int] = np.float32,
                    device: int = -1) -> Tensor:
    ...


def empty(shape: List[int],
          dtype: Union[np.dtype, int] = np.float32,
          device: int = -1,
          pinned: bool = False,
          managed: bool = False) -> Tensor:
    ...


def full(shape: List[int],
         val: Union[int, float],
         dtype: Union[np.dtype, int] = np.float32,
         device: int = -1,
         pinned: bool = False,
         managed: bool = False) -> Tensor:
    ...


def zeros_managed(shape: List[int],
                  dtype: Union[np.dtype, int] = np.float32) -> Tensor:
    ...


def from_numpy(arr: np.ndarray) -> Tensor:
    ...
