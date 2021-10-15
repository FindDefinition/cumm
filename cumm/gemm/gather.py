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

from cumm import dtypes
from cumm.common import GemmBasic, TensorView, TensorViewKernel
from cumm.gemm import (constants, gemmmath, layout, mask_iters, out_iters,
                       output_op, thread_map, volta_iters, volta_out_iters,
                       wmma)
from cumm.gemm.algospec.core import GemmAlgo, ShuffleStrideType
from cumm.gemm.core import MetaArray, metaseq, seq


class GatherKernel(pccm.ParameterizedClass):
    def __init__(self, dtype: dtypes.DType, tile_shape: MetaArray[int],
                 epa: int, num_threads: int):
        super().__init__()
        self.add_dependency(TensorView, TensorViewKernel)
        self.dtype = dtype
        self.tile_shape = tile_shape
        self.epa = epa
        sub_tile_shape = seq(1, epa)
        self.tmap = thread_map.PitchLinear(tile_shape, sub_tile_shape,
                                           num_threads)
        self.num_threads = num_threads
        self.inp_iter_in_param = mask_iters.MaskTileIteratorParams(
            dtype, tile_shape, sub_tile_shape, self.tmap, 1, True)

        self.inp_iter_out_param = mask_iters.MaskTileIteratorParams(
            dtype, tile_shape, sub_tile_shape, self.tmap, 1, False)

        self.inp_iter_in = mask_iters.MaskTileIterator(
            dtype, tile_shape, sub_tile_shape, self.tmap,
            self.inp_iter_in_param, 1, epa, False, False, True)

        self.inp_iter_out = mask_iters.MaskTileIterator(
            dtype,
            tile_shape,
            sub_tile_shape,
            self.tmap,
            self.inp_iter_out_param,
            1,
            epa,
            False,
            False,
            False,
            read_only=False)

        self.add_param_class("inpp1", self.inp_iter_in_param, "InputParams")
        self.add_param_class("outp1", self.inp_iter_out_param, "OutputParams")
        self.add_param_class("inpiter1", self.inp_iter_in, "InputIter")
        self.add_param_class("outiter1", self.inp_iter_out, "OutputIter")

    @pccm.cuda.cuda_global_function(inline=True)
    def kernel(self):
        code = pccm.FunctionCode()
        code.arg("input_ptr", f"const {self.dtype}*")
        code.arg("output_ptr", f"{self.dtype}*")

        code.arg("m, k, k_iterations", f"int")

        code.arg("inp_params", "InputParams")
        code.arg("out_params", "OutputParams")
        code.raw(f"""
        int tile_offset_m = blockIdx.x;
        int block_offset_m = tile_offset_m * {self.tile_shape[0]};
        InputIter input_iter(
            inp_params, input_ptr,
            {{m, k}},
            threadIdx.x,
            {{block_offset_m, 0}});
        OutputIter output_iter(
            out_params, output_ptr,
            {{m, k}},
            threadIdx.x,
            {{block_offset_m, 0}});
        {self.inp_iter_in.fragment_t} input_frag;
        // input_frag.clear();
        for (; k_iterations > 0; --k_iterations){{
            input_iter.load(input_frag);
            output_iter.store(input_frag);
            ++input_iter;
            ++output_iter;
        }}
        """)
        return code

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def gather(self):
        code = pccm.FunctionCode()
        code.arg("output", "tv::Tensor")
        code.arg("input", "tv::Tensor")
        code.arg("indices", "tv::Tensor")

        code.raw(f"""
        int m = input.dim(0);
        auto timer = tv::CudaContextTimer<>();
        InputParams parmas_inp(input.dim(1), indices.data_ptr<const int>());
        OutputParams parmas_out(input.dim(1));
        int k_iterations = tv::div_up(input.dim(1), {self.tile_shape[1]});
        dim3 grid(tv::div_up(m, {self.tile_shape[0]}));
        tv::cuda::Launch launcher(grid, dim3({self.num_threads}, 1, 1));
        tv::ssprint(grid.x, grid.y, grid.z, k_iterations, m);
        launcher(kernel, input.data_ptr<const {self.dtype}>(), output.data_ptr<{self.dtype}>(), m, 
            input.dim(1), k_iterations, parmas_inp, parmas_out);
        tv::ssprint("gather time", timer.report() / 1000.0);
        """)
        return code
