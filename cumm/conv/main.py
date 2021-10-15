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

import asyncio
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pccm
from ccimport import compat
from pccm.core import CodeFormatter

# from myclang import clangformat
from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.common import GemmBasic, TensorView, TensorViewKernel
from cumm.constants import CUTLASS_MODE
from cumm.conv import kernel
from cumm.conv.bases import (NCHW, NHWC, ConvEnum, ConvIterAlgo, ConvLayout,
                             ConvLayoutType, ConvMode, ConvOpType)
from cumm.conv.params import (ConvProblem, conv_iwo_to_gemm_abc_indices,
                              gemm_abc_to_conv_iwo_indices)
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm.algospec import GemmAlgo
from cumm.gemm.algospec.core import TensorOpParams
from cumm.gemm.core.metaarray import MetaArray


def seq(*vals):
    return np.array([*vals], dtype=np.int64)


class ConvAlgoParams(object):
    def __init__(self,
                 ndim: int,
                 op_type: ConvOpType,
                 iter_algo: ConvIterAlgo,
                 ts: Tuple[int, int, int],
                 wts: Tuple[int, int, int],
                 num_stage: int,
                 dtype_shorts: str,
                 layout_desp_input: ConvLayout,
                 layout_desp_weight: ConvLayout,
                 layout_desp_output: ConvLayout,
                 algo: GemmAlgo,
                 tensorop: Optional[TensorOpParams] = None,
                 splitk_serial: bool = False,
                 splitk_parallel: bool = False,
                 mask_sparse: bool = False,
                 increment_k_first: bool = False,
                 mask_width: int = -1):
        self.ts = MetaArray(*ts)
        self.wts = MetaArray(*wts)
        self.num_stage = num_stage
        self.ndim = ndim
        dtype_abcac = [
            dtypes.get_dtype_by_shortcut(s.strip())
            for s in dtype_shorts.split(",")
        ]
        assert len(dtype_abcac) == 5
        self.op_type = op_type
        self.iter_algo = iter_algo
        self.dtype_a = dtype_abcac[0]
        self.dtype_b = dtype_abcac[1]
        self.dtype_c = dtype_abcac[2]

        self.dtype_acc = dtype_abcac[3]
        self.dtype_comp = dtype_abcac[4]
        self.mask_sparse = mask_sparse
        self.increment_k_first = increment_k_first

        indices = gemm_abc_to_conv_iwo_indices(op_type)
        dtypes_abc = [self.dtype_a, self.dtype_b, self.dtype_c]
        self.dtype_input = dtypes_abc[indices[0]]
        self.dtype_weight = dtypes_abc[indices[1]]
        self.dtype_output = dtypes_abc[indices[2]]

        self.layout_desp_input = layout_desp_input
        self.layout_desp_weight = layout_desp_weight
        self.layout_desp_output = layout_desp_output
        self.algo = algo
        self.tensorop = tensorop
        self.splitk_serial = splitk_serial
        self.splitk_parallel = splitk_parallel
        self.mask_width = mask_width

    def support_splitk(self):
        return self.splitk_serial or self.splitk_parallel

    def skipped(self):
        if self.op_type != ConvOpType.kForward and self.dtype_a.itemsize(
        ) == 1:
            return True
        return False

    def get_algo_name(self):
        res = f"{self.algo.value}_{self.dtype_a.shortcut()}{self.dtype_b.shortcut()}{self.dtype_c.shortcut()}"
        res += f"{self.dtype_acc.shortcut()}{self.dtype_comp.shortcut()}"
        res += f"_m{self.ts[0]}n{self.ts[1]}k{self.ts[2]}"
        res += f"m{self.wts[0]}n{self.wts[1]}k{self.wts[2]}"
        if self.tensorop is not None:
            tss = self.tensorop.shape
            res += f"T{tss[0]}{tss[1]}{tss[2]}"
        res += f"_{self.num_stage}"
        res += f"_C{self.ndim}_{self.op_type.value}{self.iter_algo.value}"
        if self.mask_sparse:
            res += "_F" if not self.increment_k_first else "_K"

        return res


def gen_gemm_params(ts,
                    wts,
                    ndim: int,
                    iter_algo: ConvIterAlgo,
                    stage: int,
                    dtypes_string: str,
                    li: ConvLayout,
                    lw: ConvLayout,
                    lo: ConvLayout,
                    algo: GemmAlgo,
                    tensorop: Optional[TensorOpParams],
                    splitk_serial: bool = False,
                    splitk_parallel: bool = False,
                    mask_sparse: bool = False,
                    increment_k_first: bool = False,
                    mask_width: int = -1):
    res = []
    # for ta in [False, True]:
    #     for tb in [False, True]:
    #         for tc in [False, True]:
    #             p = GemmAlgoParams(ts, wts, stage, dtypes_string, ta, tb, tc, algo, tensorop)
    #             if not p.skipped():
    #                 res.append(p)
    op_types = [
        ConvOpType.kForward, ConvOpType.kBackwardInput,
        ConvOpType.kBackwardWeight
    ]
    op_types = [
        ConvOpType.kForward, ConvOpType.kBackwardInput,
        ConvOpType.kBackwardWeight
    ]
    # if mask_sparse:
    #     op_types = [ConvOpType.kForward, ConvOpType.kBackwardInput]

    for op_type in op_types:
        if op_type == ConvOpType.kBackwardWeight:
            p = ConvAlgoParams(ndim,
                               op_type,
                               iter_algo,
                               ts,
                               wts,
                               stage,
                               dtypes_string,
                               li,
                               lw,
                               lo,
                               algo,
                               tensorop,
                               True,
                               splitk_parallel,
                               mask_sparse,
                               increment_k_first,
                               mask_width=mask_width)
        else:
            p = ConvAlgoParams(ndim, op_type, iter_algo, ts, wts, stage,
                               dtypes_string, li, lw, lo, algo, tensorop,
                               splitk_serial, splitk_parallel, mask_sparse,
                               increment_k_first)

        if not p.skipped():
            res.append(p)
    return res


def gen_spwgrad_params(ts,
                       wts,
                       ndim: int,
                       iter_algo: ConvIterAlgo,
                       stage: int,
                       dtypes_string: str,
                       li: ConvLayout,
                       lw: ConvLayout,
                       lo: ConvLayout,
                       algo: GemmAlgo,
                       tensorop: Optional[TensorOpParams],
                       splitk_serial: bool = False,
                       splitk_parallel: bool = False,
                       mask_sparse: bool = False,
                       increment_k_first: bool = False,
                       mask_width: int = -1):
    p = ConvAlgoParams(ndim,
                       ConvOpType.kBackwardWeight,
                       iter_algo,
                       ts,
                       wts,
                       stage,
                       dtypes_string,
                       li,
                       lw,
                       lo,
                       algo,
                       tensorop,
                       True,
                       splitk_parallel,
                       mask_sparse,
                       increment_k_first,
                       mask_width=mask_width)
    return [p]


def gen_gemm_kernels(params: ConvAlgoParams):
    return kernel.ConvKernel(params.ndim,
                             params.op_type,
                             params.iter_algo,
                             params.ts,
                             params.wts,
                             params.num_stage,
                             dtype_a=params.dtype_a,
                             dtype_b=params.dtype_b,
                             dtype_c=params.dtype_c,
                             dtype_acc=params.dtype_acc,
                             dtype_comp=params.dtype_comp,
                             layout_desp_input=params.layout_desp_input,
                             layout_desp_output=params.layout_desp_output,
                             layout_desp_weight=params.layout_desp_weight,
                             algo=params.algo,
                             tensorop=params.tensorop,
                             splitk_serial=params.splitk_serial,
                             splitk_parallel=params.splitk_parallel,
                             mask_sparse=params.mask_sparse,
                             increment_k_first=params.increment_k_first,
                             mask_width=params.mask_width)


class ConvMainUnitTest(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, GemmBasic, ConvEnum)
        # unit test params: [ts, wts, stage, dtypes, trans, algo, tensorop]
        self.simt_params = [
            # *gen_gemm_params((64, 128, 32), (32, 64, 32), 2, ConvIterAlgo.Optimized, 2, "s8,s8,s32,s32,s32",
            #     NHWC, NHWC, NHWC, GemmAlgo.SimtDP4A, None),
            # *gen_gemm_params((32, 128, 16), (32, 32, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
            #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True),
            # *gen_spwgrad_params((128, 128, 8), (32, 64, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
            #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True, mask_width=32),
            # *gen_gemm_params((32, 128, 16), (32, 32, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
            #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True, mask_width=32),
            *gen_gemm_params(
                (32, 32, 32), (32, 32, 8), 3, ConvIterAlgo.Optimized, 2,
                "f32,f32,f32,f32,f32", NHWC, NHWC, NHWC, GemmAlgo.Simt, None),
            # *gen_gemm_params_rowmajor_c((8, 32, 8), (8, 32, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
            # *gen_gemm_params_rowmajor_c((16, 32, 8), (16, 32, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
            # *gen_gemm_params_rowmajor_c((8, 32, 8), (8, 16, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
            # *gen_gemm_params_rowmajor_c((8, 64, 8), (8, 32, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
            # *gen_gemm_params_rowmajor_c((16, 32, 8), (16, 16, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
            # *gen_gemm_params_rowmajor_c((64, 128, 8), (32, 32, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
            # *gen_gemm_params_rowmajor_c((128, 128, 8), (32, 64, 8), 2, "f16,f16,f16,f16,f16", GemmAlgo.Simt, None),
            # *gen_gemm_params_rowmajor_c((128, 128, 8), (32, 64, 8), 2, "f16,f16,f16,f32,f32", GemmAlgo.Simt, None),
            # *gen_gemm_params_rowmajor_c((128, 128, 16), (32, 64, 16), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
            # *gen_gemm_params_rowmajor_c((128, 128, 16), (32, 64, 16), 2, "f16,f16,f16,f32,f32", GemmAlgo.Simt, None),
            # *gen_gemm_params((128, 128, 32), (32, 64, 32), 2, "s8,s8,s32,s32,s32", GemmAlgo.SimtDP4A, None),
        ]  # type: List[ConvAlgoParams]
        self.volta_params = [
            # *gen_gemm_params_rowmajor_c((64, 64, 64), (32, 32, 32), 2, "f16,f16,f16,f16,f16", GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
            # *gen_gemm_params((64, 64, 32),
            #                  (32, 32, 32), 2, "f16,f16,f32,f32,f32",
            #                  GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
            # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
            # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
            # *gen_gemm_params_rowmajor_c((128, 128, 32), (64, 64, 32), 2, "f16,f16,f16,f32,f32", GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
            # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f32,f32", GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
            # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f32,f32", GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
        ]
        self.turing_params = [
            # interleave = 4:
            # *gen_gemm_params((128, 64, 32), (64, 32, 32), 2, "s8,s8,s8,s32,s32", GemmAlgo.Turing, TensorOpParams((16, 8, 16))),

            # *gen_gemm_params((64, 64, 32), (32, 32, 32), 2, "tf32,tf32,tf32,tf32,tf32", GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
            # *gen_gemm_params((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", GemmAlgo.Turing, TensorOpParams([16, 8, 16])),

            # *gen_gemm_params_rowmajor_c((64, 128, 32), (32, 64, 32), 2, "f16,f16,f32,f32,f16", GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
            # *gen_gemm_params_rowmajor_c((128, 256, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
            # *gen_gemm_params_rowmajor_c((256, 128, 32), (64, 64, 32), 2, "f16,f16,f32,f32,f16", GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
            # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
            # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", GemmAlgo.Volta, None),
        ]
        # self.turing_s8_params = [
        #     *gen_gemm_params((128, 128, 64), (64, 64, 64), 2, "s8,s8,s8,s32,f32", GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
        #     # *gen_gemm_params_rowmajor_c((64, 128, 32), (32, 64, 32), 2, "f16,f16,f32,f32,f16", GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
        #     # *gen_gemm_params_rowmajor_c((128, 256, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
        #     # *gen_gemm_params_rowmajor_c((256, 128, 32), (64, 64, 32), 2, "f16,f16,f32,f32,f16", GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
        #     # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
        #     # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", GemmAlgo.Volta, None),
        # ]
        # print(self.simt_params[0].dtype_b)
        # raise NotImplementedError
        self.all_params = self.simt_params + self.volta_params + self.turing_params
        self.all_kernels = [gen_gemm_kernels(p) for p in self.all_params]
        for p, ker in zip(self.all_params, self.all_kernels):
            self.add_impl_only_param_class(self.implicit_gemm,
                                           "cp" + p.get_algo_name(),
                                           ker.gemm_params,
                                           "ConvParams" + p.get_algo_name())

            self.add_impl_only_param_class(self.implicit_gemm,
                                           p.get_algo_name(), ker,
                                           "Conv" + p.get_algo_name())

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def implicit_gemm(self):
        code = pccm.FunctionCode()
        code.arg("input,weight,output",
                 "tv::Tensor",
                 pyanno="cumm.tensorview.Tensor")
        code.arg("padding,stride,dilation",
                 f"std::vector<int>",
                 pyanno="List[int]")

        code.arg("ndim, iter_algo_, op_type_", "int")
        code.arg("i_ltype_,w_ltype_,o_ltype_", "int")
        code.arg("ts,wts", "std::array<int, 3>", pyanno="Tuple[int, int, int]")
        code.arg("num_stage", "int", pyanno="int")
        code.arg("dacc,dcomp", "int", pyanno="int")
        code.arg("algo", "std::string", pyanno="str")
        code.arg("tensorop", "std::array<int, 3>", "std::array<int, 3>{}")
        code.arg("i_interleave", "int", "1")
        code.arg("w_interleave", "int", "1")
        code.arg("o_interleave", "int", "1")
        code.arg("alpha", "float", "1")
        code.arg("beta", "float", "0")

        code.arg("split_k_slices", "int", "1")
        code.arg("workspace",
                 "tv::Tensor",
                 "tv::Tensor()",
                 pyanno="cumm.tensorview.Tensor = Tensor()")
        code.arg("mask_sparse", "bool", "false")
        code.arg("increment_k_first", "bool", "false")

        code.arg("mask",
                 "tv::Tensor",
                 "tv::Tensor()",
                 pyanno="cumm.tensorview.Tensor = Tensor()")
        code.arg("mask_argsort",
                 "tv::Tensor",
                 "tv::Tensor()",
                 pyanno="cumm.tensorview.Tensor = Tensor()")
        code.arg("indices",
                 "tv::Tensor",
                 "tv::Tensor()",
                 pyanno="cumm.tensorview.Tensor = Tensor()")
        code.arg("mask_output",
                 "tv::Tensor",
                 "tv::Tensor()",
                 pyanno="cumm.tensorview.Tensor = Tensor()")

        code.raw(f"""
        int groups = 1;
        bool found = false;
        ConvEnum::IterAlgo iter_algo = static_cast<ConvEnum::IterAlgo>(iter_algo_);
        ConvEnum::OpType op_type = static_cast<ConvEnum::OpType>(op_type_);
        ConvEnum::LayoutType i_ltype = static_cast<ConvEnum::LayoutType>(i_ltype_);
        ConvEnum::LayoutType w_ltype = static_cast<ConvEnum::LayoutType>(w_ltype_);
        ConvEnum::LayoutType o_ltype = static_cast<ConvEnum::LayoutType>(o_ltype_);
        """)

        for p, ker in zip(self.all_params, self.all_kernels):
            indices = gemm_abc_to_conv_iwo_indices(p.op_type)
            inv_indices = conv_iwo_to_gemm_abc_indices(p.op_type)
            dtypes_abc = [p.dtype_a, p.dtype_b, p.dtype_c]
            dtypes_iwo = [dtypes_abc[i] for i in indices]
            param_cls_name = "ConvParams" + p.get_algo_name()
            param_cls_ns = "cp" + p.get_algo_name()

            if_tests = [
                f"tv::type_v<{dtypes_iwo[0]}> == input.dtype()",
                f"tv::type_v<{dtypes_iwo[1]}> == weight.dtype()",
                f"tv::type_v<{dtypes_iwo[2]}> == output.dtype()",
            ]
            if_tests.extend([
                f"{p.iter_algo.value} == iter_algo_",
                f"{p.op_type.value} == op_type_",
                f"{p.ndim} == ndim",
                f"{p.layout_desp_input.layout_type.value} == i_ltype_",
                f"{p.layout_desp_weight.layout_type.value} == w_ltype_",
                f"{p.layout_desp_output.layout_type.value} == o_ltype_",
                f"{p.layout_desp_input.interleave} == i_interleave",
                f"{p.layout_desp_weight.interleave} == w_interleave",
                f"{p.layout_desp_output.interleave} == o_interleave",
                f"std::array<int, 3>{{{code.unpack(list(p.ts))}}} == ts",
                f"std::array<int, 3>{{{code.unpack(list(p.wts))}}} == wts",
                f"{p.num_stage} == num_stage",
                f"{p.dtype_acc.tv_dtype} == dacc",
                f"{p.dtype_comp.tv_dtype} == dcomp",
                f"\"{p.algo.value}\" == algo",
                f"{pccm.boolean(p.mask_sparse)} == mask_sparse",
                f"{pccm.boolean(p.increment_k_first)} == increment_k_first",
            ])
            if p.tensorop is not None:
                if_tests.append(
                    f"std::array<int, 3>{{{code.unpack(list(p.tensorop.shape))}}} == tensorop"
                )
            if_test = " && ".join(if_tests)
            param_type_str = "ConvParams" + p.get_algo_name()
            with code.if_(if_test):
                if not p.support_splitk():
                    code.raw(f"""
                    TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
                    """)
                # TODO if input is NCxHWx
                # TODO if input weight and output have different layout
                dim_start = 2 if p.layout_desp_weight.is_channel_first() else 1
                io_ndim = 2 if p.mask_sparse else p.ndim + 2
                code.raw(f"""
                found = true;
                TV_ASSERT_RT_ERR(input.ndim() == {io_ndim}, "error");
                TV_ASSERT_RT_ERR(weight.ndim() == {p.ndim + 2}, "error");
                TV_ASSERT_RT_ERR(output.ndim() == {io_ndim}, "error");
                int N = input.dim(0);
                int C = {'input.dim(1)' if p.layout_desp_input.is_channel_first() else f'input.dim({io_ndim - 1})'};
                int K = weight.dim(0);
                int K2 = {'output.dim(1)' if p.layout_desp_output.is_channel_first() else f'output.dim({io_ndim - 1})'};
                TV_ASSERT_RT_ERR(K2 == K, "error");
                tv::array<int, {p.ndim}> ksize, input_dims, output_dims;
                """)
                if p.mask_sparse:
                    if p.op_type == ConvOpType.kForward:
                        code.raw(
                            f"TV_ASSERT_RT_ERR(!mask_output.empty(), \"error\");"
                        )
                    elif p.op_type == ConvOpType.kBackwardWeight:
                        assert p.mask_width > 0 and p.mask_width % p.ts[2] == 0
                        code.raw(f"""
                        TV_ASSERT_RT_ERR(C % {p.ts[1]} == 0, "error");
                        """)
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(!indices.empty(), "error");
                    TV_ASSERT_RT_ERR(!mask.empty(), "error");
                    TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");

                    for (int i = {dim_start}; i < {dim_start + p.ndim}; ++i){{
                        ksize[i - {dim_start}] = weight.dim(i);
                    }}
                    {param_cls_ns}::ConvProblem problem(N, C, K, ksize, 
                        ConvEnum::Mode::kCrossCorrelation, split_k_slices, groups);
                    """)
                else:
                    code.raw(f"""
                    for (int i = {dim_start}; i < {dim_start + p.ndim}; ++i){{
                        ksize[i - {dim_start}] = weight.dim(i);
                        input_dims[i - {dim_start}] = input.dim(i);
                        output_dims[i - {dim_start}] = output.dim(i);
                    }}
                    tv::array<int, {p.ndim}> padding_arr{{{code.unpack([f"padding[{i}]" for i in range(p.ndim)])}}};
                    tv::array<int, {p.ndim}> stride_arr{{{code.unpack([f"stride[{i}]" for i in range(p.ndim)])}}};
                    tv::array<int, {p.ndim}> dilation_arr{{{code.unpack([f"dilation[{i}]" for i in range(p.ndim)])}}};
                    auto output_dims_check_again = {param_cls_ns}::ConvProblem::calc_output_dims(input_dims, ksize, padding_arr, stride_arr, dilation_arr);
                    for (int i = 0; i < {p.ndim}; ++i){{
                        TV_ASSERT_RT_ERR(output_dims_check_again[i] == output_dims[i], "error");
                    }}
                    {param_cls_ns}::ConvProblem problem(N, C, K, input_dims, output_dims, ksize, padding_arr, stride_arr, dilation_arr, 
                        ConvEnum::Mode::kCrossCorrelation, split_k_slices, groups);
                    """)
                code.raw(f"""
                auto mnk = problem.implicit_gemm_mnk(op_type);
                int workspace_size = 0;
                auto logical_tile_count = {param_type_str}::get_logical_tile_count(mnk[0], mnk[1], mnk[2], split_k_slices);
                if (split_k_slices > 1){{
                    if ({pccm.boolean(p.splitk_serial)}){{
                        workspace_size = sizeof(int) * logical_tile_count.x * logical_tile_count.y;
                    }} else if ({pccm.boolean(p.splitk_parallel)}){{
                        workspace_size = {p.dtype_acc.itemsize()} * mnk[0] * mnk[1] * logical_tile_count.z;
                    }} else{{
                        TV_THROW_INVALID_ARG("not impemented");
                    }}
                }}

                if (workspace_size > 0){{
                    if (!workspace.empty()){{
                        TV_ASSERT_RT_ERR(workspace.nbytes() >= workspace_size, 
                            "workspace at least", workspace_size, "bytes.");
                    }}else{{
                        workspace = tv::zeros({{workspace_size}}, tv::uint8, 0);
                    }}
                }}

                """)
                input_names = ["input", "weight", "output"]
                input_names = [input_names[i] for i in inv_indices]
                code.raw(f"""
                auto a_ten = {input_names[0]};
                auto b_ten = {input_names[1]};
                auto c_ten = {input_names[2]};
                """)
                if p.mask_sparse:
                    mask_out_ptr = "mask_output.data_ptr<uint32_t>(), "
                    if p.op_type != ConvOpType.kForward:
                        mask_out_ptr = ""
                    code.raw(f"""
                    {param_type_str} params(
                        problem, a_ten.data_ptr<{ker.dtype_a}>(), b_ten.data_ptr<{ker.dtype_b}>(),
                        c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(),
                        mask.data_ptr<uint32_t>(), mask_argsort.data_ptr<int32_t>(),
                        indices.data_ptr<int32_t>(), {mask_out_ptr}
                        {p.dtype_comp}(alpha), {p.dtype_comp}(beta){", split_k_slices, workspace.raw_data()" if p.support_splitk() else ""});
                    """)
                else:
                    code.raw(f"""
                    {param_type_str} params(
                        problem, a_ten.data_ptr<{ker.dtype_a}>(), b_ten.data_ptr<{ker.dtype_b}>(),
                        c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(), 
                        {p.dtype_comp}(alpha), {p.dtype_comp}(beta){", split_k_slices, workspace.raw_data()" if p.support_splitk() else ""});
                    """)

                code.raw(f"""
                tv::cuda::Launch launcher(params.grid_dims, dim3({ker.num_threads}),
                                            {ker.smem_size});
                cudaError_t result;
                if ({ker.smem_size} >= (48 << 10)) {{
                    result = cudaFuncSetAttribute({p.get_algo_name()}::conv_kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    {ker.smem_size});
                    TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                    result = cudaFuncSetAttribute(
                        {p.get_algo_name()}::conv_kernel,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                    TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                }}
                // auto timer = tv::CudaContextTimer<>();

                launcher({p.get_algo_name()}::conv_kernel, params);
                cudaFuncAttributes attr;
                checkCudaErrors(
                    cudaFuncGetAttributes(&attr, {p.get_algo_name()}::conv_kernel));
                // tv::ssprint("{p.get_algo_name()} kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);

                // tv::ssprint("{p.get_algo_name()} kernel num regs:", attr.numRegs, "my conv time", timer.report() / 1000.0);

                TV_CHECK_CUDA_ERR_V2("???");
                """)
        code.raw("""
        if (!found){
            TV_THROW_INVALID_ARG("Can't Found Algorithm for params:", ts, wts, num_stage, tv::dtype_str(input.dtype()), 
                tv::dtype_str(weight.dtype()), tv::dtype_str(output.dtype()), tv::dtype_str(dacc), 
                tv::dtype_str(dcomp), algo, tensorop);
        }
        """)
        return code

    # @lineprof.lineprof_wrapper_cpp
    def implicit_gemm_python(self,
                             input_: np.ndarray,
                             weight: np.ndarray,
                             output: np.ndarray,
                             input_meta: np.ndarray,
                             weight_meta: np.ndarray,
                             output_meta: np.ndarray,
                             padding: List[int],
                             stride: List[int],
                             dilation: List[int],
                             ndim: int,
                             iter_algo: ConvIterAlgo,
                             op_type: ConvOpType,
                             i_ltype: ConvLayoutType,
                             w_ltype: ConvLayoutType,
                             o_ltype: ConvLayoutType,
                             ts: np.ndarray,
                             wts: np.ndarray,
                             num_stage: int,
                             dacc: dtypes.DType,
                             dcomp: dtypes.DType,
                             algo: str,
                             tensorop: np.ndarray,
                             i_interleave: int = 1,
                             w_interleave: int = 1,
                             o_interleave: int = 1):
        found = False
        for p, ker in zip(self.all_params, self.all_kernels):
            indices = gemm_abc_to_conv_iwo_indices(p.op_type)
            inv_indices = conv_iwo_to_gemm_abc_indices(p.op_type)
            dtypes_abc = [p.dtype_a, p.dtype_b, p.dtype_c]
            dtypes_iwo = [dtypes_abc[i] for i in indices]

            if_tests = [
                dtypes_iwo[0].npdtype() == input_.dtype,
                dtypes_iwo[1].npdtype() == weight.dtype,
                dtypes_iwo[2].npdtype() == output.dtype,
                p.layout_desp_input.layout_type == i_ltype,
                p.layout_desp_weight.layout_type == w_ltype,
                p.layout_desp_output.layout_type == o_ltype,
                p.layout_desp_input.interleave == i_interleave,
                p.layout_desp_weight.interleave == w_interleave,
                p.layout_desp_output.interleave == o_interleave,
                p.ts[0] == ts[0] and p.ts[1] == ts[1] and p.ts[2] == ts[2],
                p.wts[0] == wts[0] and p.wts[1] == wts[1]
                and p.wts[2] == wts[2],
                p.num_stage == num_stage,
                p.dtype_acc == dacc,
                p.dtype_comp == dcomp,
                algo == p.algo.value,
            ]
            if all(if_tests):
                found = True
                assert input_.ndim == p.ndim + 2
                assert weight.ndim == p.ndim + 2
                assert output.ndim == p.ndim + 2
                N = input_.shape[0]
                if p.layout_desp_input.is_channel_first():
                    C = input_.shape[1]
                else:
                    C = input_.shape[p.ndim + 1]
                K = weight.shape[0]
                if p.layout_desp_output.is_channel_first():
                    K2 = output.shape[1]
                else:
                    K2 = output.shape[p.ndim + 1]
                assert K == K2
                ksize = [0] * p.ndim
                input_dims = [0] * p.ndim
                output_dims = [0] * p.ndim
                dim_start = 2 if p.layout_desp_weight.is_channel_first() else 1
                for i in range(dim_start, dim_start + p.ndim):
                    ksize[i - dim_start] = weight.shape[i]
                    input_dims[i - dim_start] = input_.shape[i]
                    output_dims[i - dim_start] = output.shape[i]

                output_dims_check_again = ConvProblem.calc_output_dims_python(
                    input_dims, ksize, padding, stride, dilation)
                assert output_dims_check_again == output_dims
                problem = ker.problem.python_ctor(N, C, K, input_dims,
                                                  output_dims, ksize, padding,
                                                  stride, dilation,
                                                  ConvMode.kCrossCorrelation,
                                                  1, 1)
                print(problem.N_, problem.C_, problem.K_, problem.output_dims_)
                inputs = [input_, weight, output]
                input_metas = [input_meta, weight_meta, output_meta]
                input_abcs = [inputs[i] for i in inv_indices]
                input_meta_abcs = [input_metas[i] for i in inv_indices]

                a_ten = input_abcs[0]
                b_ten = input_abcs[1]
                c_ten = input_abcs[2]
                a_meta_ten = input_meta_abcs[0]
                b_meta_ten = input_meta_abcs[1]

                if cudasim.enable_debug():
                    a_ptr = ArrayPtr(p.dtype_a.tv_dtype,
                                     a_ten.size,
                                     external_data=tv.from_numpy(a_ten),
                                     meta_data=tv.from_numpy(a_meta_ten))
                    b_ptr = ArrayPtr(p.dtype_b.tv_dtype,
                                     b_ten.size,
                                     external_data=tv.from_numpy(b_ten),
                                     meta_data=tv.from_numpy(b_meta_ten))
                else:
                    a_ptr = ArrayPtr(p.dtype_a.tv_dtype,
                                     a_ten.size,
                                     external_data=tv.from_numpy(a_ten),
                                     meta_data=tv.Tensor())
                    b_ptr = ArrayPtr(p.dtype_b.tv_dtype,
                                     b_ten.size,
                                     external_data=tv.from_numpy(b_ten),
                                     meta_data=tv.Tensor())

                c_ptr = ArrayPtr(p.dtype_c.tv_dtype,
                                 c_ten.size,
                                 external_data=tv.from_numpy(c_ten))
                params = ker.gemm_params.python_ctor(problem, a_ptr, b_ptr,
                                                     c_ptr, c_ptr, 1.0, 0.0)
                func = partial(ker.conv_kernel_python, params=params)
                blocks = params.grid_dims
                threads = cudasim.Dim3(ker.num_threads, 1, 1)
                return asyncio.run(
                    cudasim.kernel_launch(func, blocks, threads,
                                          ker.smem_size)), blocks, threads
        raise NotImplementedError
