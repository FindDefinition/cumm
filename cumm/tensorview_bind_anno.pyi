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

import builtins
from typing import Dict, List, Optional, Tuple, Type, Union, overload

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


class NVRTCProgram:
    def __init__(self,
                 code: str,
                 headers: Dict[str, str] = {},
                 opts: List[str] = [],
                 program_name: str = "kernel") -> None:
        ...

    def ptx(self) -> str:
        ...

    def compile_log(self) -> str:
        ...

    def get_lowered_name(self, name: str) -> str:
        ...


class NVRTCModule:
    kTensor = 0
    kArray = 1
    kScalar = 2

    @overload
    def __init__(self,
                 code: str,
                 headers: Dict[str, str] = {},
                 opts: List[str] = [],
                 program_name: str = "kernel",
                 cudadevrt_path: str = "") -> None:
        ...

    @overload
    def __init__(self, prog: NVRTCProgram) -> None:
        ...

    def load(self) -> "NVRTCModule":
        ...

    def run_kernel(self, name: str, blocks: List[int], threads: List[int],
                   smem_size: int, stream: int, args: List[Tuple[Tensor,
                                                                 int]]):
        ...


    @property 
    def program(self) -> NVRTCProgram:
        ...

    def get_lowered_name(self, name: str) -> str:
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

def get_compute_capability(index: int) -> Tuple[int, int]:
    ...
    
def is_cpu_only() -> bool:
    ...
