from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
class TensorViewBind:
    @staticmethod
    def hello() -> None: ...
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
from enum import Enum 

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

    def insert_pair(self, name: str, start: str, stop: str) -> str:
        ...

    def get_pair_duration(self, name: str) -> float:
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

    def to_string(self) -> str:
        ...

    @staticmethod
    def from_string(json_string: str) -> "NVRTCProgram":
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
                 name_exprs: List[str] = [],
                 cudadevrt_path: str = "") -> None:
        ...

    @overload
    def __init__(self, prog: NVRTCProgram, cudadevrt_path: str = "") -> None:
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

    def get_kernel_attributes(self, name: str) -> Dict[str, int]:
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

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
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

def cufilt(name: str) -> str:
    ...

class ConvOpType(Enum):
    Forward = 0
    BackwardInput = 1
    BackwardWeight = 2

class ConvIterAlgo(Enum):
    Analytic = 0
    Optimized = 1

class ConvMode(Enum):
    Convolution = 0
    CrossCorrelation = 1


class ConvLayoutType(Enum):
    ChannelFirst = 0
    ChannelLast = 1
    SpatialFirst = 2

class ShuffleStrideType(Enum):
    NoShuffle = 0
    ShuffleAC = 1
    ShuffleAB = 2

class NVRTCParams:
    cumodule: NVRTCModule
    kernel_name: str 
    init_kernel_name: str 
    constant_name: str 
    param_size: int 
    param_storage: Tensor 
    param_storage_cpu: Tensor 
    num_threads: int 
    smem_size: int 
    mode: int

class GemmAlgoDesp:
    dtype_a: int
    dtype_b: int
    dtype_c: int
    tile_shape: Tuple[int, int, int]
    warp_tile_shape: Tuple[int, int, int]
    num_stage: int
    dacc: int
    dcomp: int
    algo: str
    tensorop: List[int]
    split_k_serial_: int
    split_k_parallel_: int
    shuffle_type: ShuffleStrideType
    element_per_access_a: int
    element_per_access_b: int
    element_per_access_c: int
    access_per_vector: int
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def split_k_serial(self) -> bool: ...
    @split_k_serial.setter
    def split_k_serial(self, val: bool) -> None: 
        """
        Args:
            val: 
        """
        ...
    @property
    def split_k_parallel(self) -> bool: ...
    @split_k_parallel.setter
    def split_k_parallel(self, val: bool) -> None: 
        """
        Args:
            val: 
        """
        ...
    def check_valid(self) -> None: ...
    @property
    def trans_a(self) -> bool: ...
    @trans_a.setter
    def trans_a(self, val: bool) -> None: 
        """
        Args:
            val: 
        """
        ...
    @property
    def trans_b(self) -> bool: ...
    @trans_b.setter
    def trans_b(self, val: bool) -> None: 
        """
        Args:
            val: 
        """
        ...
    @property
    def trans_c(self) -> bool: ...
    @trans_c.setter
    def trans_c(self, val: bool) -> None: 
        """
        Args:
            val: 
        """
        ...
    def query_workspace_size(self, m: int, n: int, k: int, split_k_slices: int) -> int: 
        """
        Args:
            m: 
            n: 
            k: 
            split_k_slices: 
        """
        ...
    def supported(self, m: int, n: int, k: int) -> bool: 
        """
        Args:
            m: 
            n: 
            k: 
        """
        ...
    def supported_ldx(self, lda: int, ldb: int, ldc: int) -> bool: 
        """
        Args:
            lda: 
            ldb: 
            ldc: 
        """
        ...

class ConvAlgoDesp(GemmAlgoDesp):
    ndim: int
    op_type: ConvOpType
    iter_algo: ConvIterAlgo
    layout_i: ConvLayoutType
    layout_w: ConvLayoutType
    layout_o: ConvLayoutType
    interleave_i: int
    interleave_w: int
    interleave_o: int
    mask_sparse: bool
    increment_k_first: bool
    def __init__(self, ndim: int, op_type: ConvOpType) -> None: 
        """
        Args:
            ndim: 
            op_type: 
        """
        ...
    def __repr__(self) -> str: ...
    @staticmethod
    def conv_iwo_012_to_abc(op_type: ConvOpType) -> List[int]: 
        """
        Args:
            op_type: 
        """
        ...
    @staticmethod
    def gemm_abc_012_to_iwo(op_type: ConvOpType) -> List[int]: 
        """
        Args:
            op_type: 
        """
        ...
    @property
    def dtype_input(self) -> int: ...
    @property
    def dtype_weight(self) -> int: ...
    @property
    def dtype_output(self) -> int: ...
    def supported(self, m: int, n: int, k: int, C: int, K: int, mask_width: int) -> bool: 
        """
        Args:
            m: 
            n: 
            k: 
            C: 
            K: 
            mask_width: 
        """
        ...
    def query_conv_workspace_size(self, m: int, n: int, k: int, split_k_slices: int, kv: int) -> int: 
        """
        Args:
            m: 
            n: 
            k: 
            split_k_slices: 
            kv: 
        """
        ...
    def supported_ldx_conv(self, ldi: int, ldw: int, ldo: int) -> bool: 
        """
        Args:
            ldi: 
            ldw: 
            ldo: 
        """
        ...

class GemmParams:
    algo_desp: GemmAlgoDesp
    split_k_slices: int
    workspace: Tensor =  Tensor()
    a_inds: Tensor =  Tensor()
    b_inds: Tensor =  Tensor()
    c_inds: Tensor =  Tensor()
    alpha: float
    beta: float
    stream: int
    timer: CUDAKernelTimer
    nvrtc_params: NVRTCParams
    def __init__(self, timer: CUDAKernelTimer =  CUDAKernelTimer(False)) -> None: 
        """
        Args:
            timer: 
        """
        ...
    def check_valid(self) -> None: ...
    @property
    def a(self) -> Tensor: ...
    @a.setter
    def a(self, val: Tensor) -> None: 
        """
        Args:
            val: 
        """
        ...
    @property
    def b(self) -> Tensor: ...
    @b.setter
    def b(self, val: Tensor) -> None: 
        """
        Args:
            val: 
        """
        ...
    @property
    def c(self) -> Tensor: ...
    @c.setter
    def c(self, val: Tensor) -> None: 
        """
        Args:
            val: 
        """
        ...

class ConvParams:
    conv_algo_desp: ConvAlgoDesp
    input: Tensor
    weight: Tensor
    output: Tensor
    split_k_slices: int
    padding: List[int]
    stride: List[int]
    dilation: List[int]
    alpha: float
    beta: float
    mask_width: int
    mask_filter: int
    reverse_mask: bool
    verbose: bool
    timer: CUDAKernelTimer
    workspace: Tensor =  Tensor()
    mask: Tensor =  Tensor()
    mask_argsort: Tensor =  Tensor()
    indices: Tensor =  Tensor()
    mask_output: Tensor =  Tensor()
    stream: int
    nvrtc_params: NVRTCParams
    def __init__(self, ndim: int, op_type: ConvOpType, timer: CUDAKernelTimer =  CUDAKernelTimer(False)) -> None: 
        """
        Args:
            ndim: 
            op_type: 
            timer: 
        """
        ...

def run_nvrtc_gemm_kernel(params: GemmParams) -> None: ...

def run_nvrtc_conv_kernel(params: ConvParams) -> None: ...