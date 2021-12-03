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

import enum
from typing import Dict, Tuple, List
from cumm.gemm.core import MetaArray, metaseq, seq


class GemmAlgo(enum.Enum):
    Simt = "Simt"
    SimtDP4A = "SimtDP4A"
    SimtDP2A = "SimtDP2A"
    Volta = "Volta"
    Turing = "Turing"
    Ampere = "Ampere"

_GEMM_MIN_ARCH_TO_ALGO : List[Tuple[Tuple[int, int], List[str]]] = [
    ((3, 5), [GemmAlgo.Simt.value]),
    ((6, 1), [GemmAlgo.SimtDP4A.value, GemmAlgo.SimtDP2A.value]),
    ((7, 0), [GemmAlgo.Volta.value]),
    ((7, 5), [GemmAlgo.Turing.value]),
    ((8, 0), [GemmAlgo.Ampere.value]),
]

_GEMM_ALGO_TO_MIN_ARCH: Dict[str, Tuple[int, int]] = {}

for min_arch, algos in _GEMM_MIN_ARCH_TO_ALGO:
    for algo in algos:
        _GEMM_ALGO_TO_MIN_ARCH[algo] = min_arch

def get_min_arch_of_algo(algo: GemmAlgo):
    return _GEMM_ALGO_TO_MIN_ARCH[algo.value]

def get_min_arch_of_algo_str(algo_str: str):
    return _GEMM_ALGO_TO_MIN_ARCH[algo_str]

def get_available_algo_str_from_arch(arch: Tuple[int, int]):
    res: List[str] = []
    for i in range(len(_GEMM_MIN_ARCH_TO_ALGO) - 1, -1, -1):
        arch_cur, algos = _GEMM_MIN_ARCH_TO_ALGO[i]
        if arch >= arch_cur:
            res.extend(algos)
    return res

class ShuffleStrideType(enum.Enum):
    NoShuffle = "NS"
    # A and C have indices, for spatial spconv forward and backward input
    ShuffleAC = "SAC"
    # A and B have indices, for spatial spconv backward weight
    ShuffleAB = "SAB"


class TensorOpParams(object):
    def __init__(self, shape: Tuple[int, int, int]):
        self.shape = seq(*shape)

    def to_string(self):
        return f"{self.shape[0]}{self.shape[1]}{self.shape[2]}"

    def __getitem__(self, val: int):
        return self.shape[val]
