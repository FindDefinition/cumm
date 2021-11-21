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

from enum import Enum
from typing import List, Tuple, Union

import numpy as np

from cumm.core_cc import tensorview_bind
from cumm.core_cc.tensorview_bind import Tensor

from cumm.core_cc.tensorview_bind import CUDAKernelTimer


def get_numpy_view(ten: Tensor) -> np.ndarray:
    if not ten.is_contiguous():
        raise NotImplementedError(
            "numpy_view only support contiguous tv::Tensor")
    buf = ten.get_memoryview()
    return np.frombuffer(buf, dtype=TENSOR_TO_NPDTYPE_MAP[ten.dtype]).reshape(
        ten.shape)


def numpy_view(self):
    return get_numpy_view(self)


Tensor.numpy_view = numpy_view

bool_ = 0
float16 = 1
float32 = 2
float64 = 3
int8 = 4
int16 = 5
int32 = 6
int64 = 7
uint8 = 8
uint16 = 9
uint32 = 10
uint64 = 11
tf32 = 13

custom16 = 100
custom32 = 101
custom48 = 102
custom64 = 103
custom80 = 104
custom96 = 105
custom128 = 106

NPDTYPE_TO_TENSOR_MAP = {
    np.dtype(np.float32): float32,
    np.dtype(np.int32): int32,
    np.dtype(np.int16): int16,
    np.dtype(np.int8): int8,
    np.dtype(np.float64): float64,
    np.dtype(np.bool_): bool_,
    np.dtype(np.uint8): uint8,
    np.dtype(np.float16): float16,
    np.dtype(np.int64): int64,
    np.dtype(np.uint16): uint16,
    np.dtype(np.uint32): uint32,
    np.dtype(np.uint64): uint64,
}

ALL_TV_TENSOR_DTYPES = set([
    bool_, float16, float32, float64, int8, int16, int32, int64, uint8, uint16,
    uint32, uint64, tf32, custom16, custom32, custom48, custom64, custom80,
    custom96, custom128
])

TENSOR_TO_NPDTYPE_MAP = {v: k for k, v in NPDTYPE_TO_TENSOR_MAP.items()}
TENSOR_TO_NPDTYPE_MAP[tf32] = np.dtype(np.float32)

_SUPPORTED_FILL_INT = {int32, int16, int8, uint32, uint16, uint8}


def zeros(shape: List[int],
          dtype: Union[np.dtype, int] = np.float32,
          device: int = -1,
          pinned: bool = False,
          managed: bool = False) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.zeros(shape, tv_dtype, device, pinned, managed)


def from_blob_strided(ptr: int, shape: List[int], stride: List[int],
                      dtype: Union[np.dtype, int], device: int) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.from_blob(ptr, shape, stride, tv_dtype,
                                             device)


def from_const_blob_strided(ptr: int, shape: List[int], stride: List[int],
                            dtype: Union[np.dtype,
                                         int], device: int) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.from_const_blob(ptr, shape, stride,
                                                   tv_dtype, device)


def from_blob(ptr: int, shape: List[int], dtype: Union[np.dtype, int],
              device: int) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.from_blob(ptr, shape, tv_dtype, device)


def from_const_blob(ptr: int, shape: List[int], dtype: Union[np.dtype, int],
                    device: int) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.from_const_blob(ptr, shape, tv_dtype, device)


def empty(shape: List[int],
          dtype: Union[np.dtype, int] = np.float32,
          device: int = -1,
          pinned: bool = False,
          managed: bool = False) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in TENSOR_TO_NPDTYPE_MAP
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.empty(shape, tv_dtype, device, pinned, managed)


def full(shape: List[int],
         val: Union[int, float],
         dtype: Union[np.dtype, int] = np.float32,
         device: int = -1,
         pinned: bool = False,
         managed: bool = False) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in TENSOR_TO_NPDTYPE_MAP
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    if tv_dtype == float32:
        return tensorview_bind.full_float(shape, val, tv_dtype, device, pinned,
                                          managed)
    elif tv_dtype in _SUPPORTED_FILL_INT:
        return tensorview_bind.full_int(shape, val, tv_dtype, device, pinned,
                                        managed)
    else:
        raise NotImplementedError


def zeros_managed(shape: List[int],
                  dtype: Union[np.dtype, int] = np.float32) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in TENSOR_TO_NPDTYPE_MAP
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.zeros_managed(shape, tv_dtype)


def from_numpy(arr: np.ndarray) -> Tensor:
    return tensorview_bind.from_numpy(arr)
