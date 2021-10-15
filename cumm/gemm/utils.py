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

import pccm

from cumm.common import TensorView
from cumm.gemm.core.metaarray import MetaArray


class GemmUtils(pccm.ParameterizedClass):
    def __init__(self, tile_shape: MetaArray[int]):
        super().__init__()
        self.add_dependency(TensorView)
        self.tile_shape = tile_shape

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def get_gemm_k_size_per_split(self):
        """get gemm per split k
        first we need to get iterations by tile shape k,
        
        """
        code = pccm.FunctionCode()
        code.arg("k, split_k", "int")
        code.raw(f"""
        int total_gemm_k_iterations = tv::div_up(k, {self.tile_shape[2]});
        int gemm_k_iterations_per_split =
            tv::div_up(total_gemm_k_iterations, split_k);
        auto gemm_k_size_per_split = gemm_k_iterations_per_split * {self.tile_shape[2]}; 
        return gemm_k_size_per_split;
        """)
        return code.ret("int")

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def get_gemm_k_bound(self):
        code = pccm.FunctionCode()
        code.arg("k, gemm_k_size_per_split, tile_offset_k", "int")
        code.raw(f"""
        int k_bound = min(k, (tile_offset_k + 1) * gemm_k_size_per_split);
        return k_bound;
        """)
        return code.ret("int")

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def get_gemm_iterations(self):
        code = pccm.FunctionCode()
        code.arg("k_bound, gemm_k_size_per_split, tile_offset_k", "int")
        code.raw(f"""
        int gemm_k_iterations =
            tv::div_up(k_bound - tile_offset_k * gemm_k_size_per_split, {self.tile_shape[2]});
        return gemm_k_iterations;
        """)
        return code.ret("int")
