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
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pccm
from ccimport import compat
from pccm.core import CodeFormatter, FunctionCode

# from myclang import clangformat
from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.common import GemmBasic, PyBind11, TensorView, TensorViewKernel
from cumm.gemm.utils import GemmUtilsCPU
from cumm.constants import CUTLASS_MODE
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import codeops, kernel
from cumm.gemm.algospec.core import GemmAlgo, ShuffleStrideType, TensorOpParams
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.core.metaarray import MetaArray


class GemmAlgoParams(object):
    def __init__(
            self,
            ts: Tuple[int, int, int],
            wts: Tuple[int, int, int],
            num_stage: int,
            dtype_shorts: str,
            trans_a: bool,
            trans_b: bool,
            trans_c: bool,
            algo: kernel.GemmAlgo,
            tensorop: Optional[kernel.TensorOpParams] = None,
            splitk_serial: bool = False,
            splitk_parallel: bool = False,
            shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle,
            access_per_vector: int = 1):
        self.ts = MetaArray(*ts)
        self.wts = MetaArray(*wts)
        self.num_stage = num_stage
        dtype_abcac = [
            dtypes.get_dtype_by_shortcut(s.strip())
            for s in dtype_shorts.split(",")
        ]
        assert len(dtype_abcac) == 5

        self.dtype_a = dtype_abcac[0]
        self.dtype_b = dtype_abcac[1]
        self.dtype_c = dtype_abcac[2]

        self.dtype_acc = dtype_abcac[3]
        self.dtype_comp = dtype_abcac[4]
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.algo = algo
        self.tensorop = tensorop
        self.splitk_serial = splitk_serial
        self.splitk_parallel = splitk_parallel
        self.shuffle_stride = shuffle_stride
        self.access_per_vector = access_per_vector

    def support_splitk(self):
        return self.splitk_serial or self.splitk_parallel

    def skipped(self):
        if self.dtype_a == dtypes.int8:
            if self.tensorop is not None:
                if (self.trans_a or not self.trans_b):
                    return True
        return False

    def get_algo_name(self):
        res = f"{self.algo.value}_{self.dtype_a.shortcut()}{self.dtype_b.shortcut()}{self.dtype_c.shortcut()}"
        res += f"{self.dtype_acc.shortcut()}{self.dtype_comp.shortcut()}"
        las = "n" if self.trans_a else "t"
        lbs = "n" if self.trans_b else "t"
        lcs = "n" if self.trans_c else "t"
        res += f"{las}{lbs}{lcs}"
        res += f"_m{self.ts[0]}n{self.ts[1]}k{self.ts[2]}"
        res += f"m{self.wts[0]}n{self.wts[1]}k{self.wts[2]}"
        res += f"A{self.access_per_vector}"
        if self.tensorop is not None:
            tes = self.tensorop.shape
            res += f"T{tes[0]}{tes[1]}{tes[2]}"
        res += f"_{self.num_stage}"
        if self.shuffle_stride != ShuffleStrideType.NoShuffle:
            res += f"_{self.shuffle_stride.value}"
        if self.splitk_serial:
            res += "1"
        else:
            res += "0"
        if self.splitk_parallel:
            res += "1"
        else:
            res += "0"
        return res


def gen_gemm_params(
        ts,
        wts,
        stage: int,
        dtypes_string: str,
        algo: kernel.GemmAlgo,
        tensorop: Optional[kernel.TensorOpParams],
        splitk_serial: bool = False,
        splitk_parallel: bool = False,
        shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle):
    res = []
    # for ta in [False, True]:
    #     for tb in [False, True]:
    #         for tc in [False, True]:
    #             p = GemmAlgoParams(ts, wts, stage, dtypes_string, ta, tb, tc, algo, tensorop)
    #             if not p.skipped():
    #                 res.append(p)
    for ta in [False]:
        for tb in [True]:
            for tc in [False]:
                p = GemmAlgoParams(ts, wts, stage, dtypes_string, ta, tb, tc,
                                   algo, tensorop, splitk_serial,
                                   splitk_parallel, shuffle_stride)
                if not p.skipped():
                    res.append(p)

    return res


def gen_shuffle_params(ts, wts, dss: List[str], stage: int,
                       algo: kernel.GemmAlgo,
                       tensorop: Optional[kernel.TensorOpParams]):
    res = []
    for ds in dss:
        for tb in [False, True]:
            p = GemmAlgoParams(ts, wts, stage, ds, False, tb, False, algo,
                               tensorop, False, False,
                               ShuffleStrideType.ShuffleAC)
            if not p.skipped():
                res.append(p)
        p = GemmAlgoParams(ts, wts, stage, ds, True, False, False, algo,
                           tensorop, True, False, ShuffleStrideType.ShuffleAB)
        if not p.skipped():
            res.append(p)
    return res

def gen_shuffle_params_v2(ts, wts, dss: List[str], ds_for_sab: str, stage: int,
                       algo: kernel.GemmAlgo,
                       tensorop: Optional[kernel.TensorOpParams]):
    res = []
    for ds in dss:
        for tb in [False, True]:
            p = GemmAlgoParams(ts, wts, stage, ds, False, tb, False, algo,
                               tensorop, False, False,
                               ShuffleStrideType.ShuffleAC)
            if not p.skipped():
                res.append(p)
    if ds_for_sab:
        p = GemmAlgoParams(ts, wts, stage, ds_for_sab, True, False, False, algo,
                            tensorop, True, False, ShuffleStrideType.ShuffleAB)
        if not p.skipped():
            res.append(p)
    return res

def gen_gemm_params_rowmajor_c(ts,
                               wts,
                               stage: int,
                               dtypes_string: str,
                               algo: kernel.GemmAlgo,
                               tensorop: Optional[kernel.TensorOpParams],
                               splitk_serial: bool = False,
                               splitk_parallel: bool = False):
    res = []
    for ta in [False]:
        for tb in [False]:
            for tc in [False]:
                p = GemmAlgoParams(ts, wts, stage, dtypes_string, ta, tb, tc,
                                   algo, tensorop, splitk_serial,
                                   splitk_parallel)
                if not p.skipped():
                    res.append(p)
    return res


def gen_gemm_kernels(params: GemmAlgoParams):
    return kernel.GemmKernel(params.ts,
                             params.wts,
                             params.num_stage,
                             dtype_a=params.dtype_a,
                             dtype_b=params.dtype_b,
                             dtype_c=params.dtype_c,
                             dtype_acc=params.dtype_acc,
                             dtype_comp=params.dtype_comp,
                             trans_a=params.trans_a,
                             trans_b=params.trans_b,
                             trans_c=params.trans_c,
                             algo=params.algo,
                             tensorop=params.tensorop,
                             splitk_serial=params.splitk_serial,
                             splitk_parallel=params.splitk_parallel,
                             shuffle_stride=params.shuffle_stride,
                             access_per_vector=params.access_per_vector)


class SpconvKernel(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("spconv/minkowski.cu.h")


class IGemmMaskIterator(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_member("k_idx, filter_idx", "int")
        self.add_member("mask", "const int&")

    @pccm.cuda.member_function(device=True, forceinline=True)
    def increment(self):
        pass

@pccm.pybind.bind_class_module_local
class GemmAlgoDesp(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, GemmUtilsCPU)
        # why not std::optional?
        # c++17 requires cuda11. to support cuda 10.2, we can't use c++17 for now.
        self.add_pybind_member("dtype_a,dtype_b,dtype_c", "int")  # -1 means unset
        self.add_member("trans_a_,trans_b_,trans_c_", "int")  # -1 means unset
        self.add_pybind_member("tile_shape,warp_tile_shape",
                               "std::array<int, 3>",
                               pyanno="Tuple[int, int, int]")
        self.add_pybind_member("num_stage", "int")
        self.add_pybind_member("dacc,dcomp", "int")
        self.add_pybind_member("algo", "std::string")
        self.add_pybind_member("tensorop", "std::array<int, 3>",
                               "std::array<int, 3>{}")
        self.add_pybind_member("split_k_serial_", "int", "0")  # -1 means unset
        self.add_pybind_member("split_k_parallel_", "int",
                               "0")  # -1 means unset
        self.add_pybind_member("shuffle_type", "std::string",
                               f"\"{ShuffleStrideType.NoShuffle.value}\"")
        self.add_pybind_member("element_per_access_a", "int", "-1")
        self.add_pybind_member("element_per_access_b", "int", "-1")
        self.add_pybind_member("element_per_access_c", "int", "-1")
        self.add_pybind_member("access_per_vector", "int", "1")

    @pccm.pybind.mark
    @pccm.constructor
    def default_ctor(self):
        code = pccm.FunctionCode()
        code.ctor_init("dtype_a", "int(tv::unknown)")
        code.ctor_init("dtype_b", "int(tv::unknown)")
        code.ctor_init("dtype_c", "int(tv::unknown)")

        code.ctor_init("trans_a_", "-1")
        code.ctor_init("trans_b_", "-1")
        code.ctor_init("trans_c_", "-1")
        code.ctor_init("tile_shape", "{-1, -1, -1}")
        code.ctor_init("warp_tile_shape", "{-1, -1, -1}")
        code.ctor_init("num_stage", "-1")
        code.ctor_init("dacc", "int(tv::unknown)")
        code.ctor_init("dcomp", "int(tv::unknown)")
        code.ctor_init("algo", "\"\"")
        code.ctor_init("tensorop", "{-1, -1, -1}")
        code.ctor_init("shuffle_type",
                       f"\"{ShuffleStrideType.NoShuffle.value}\"")
        code.ctor_init("split_k_serial_", "0")
        code.ctor_init("split_k_parallel_", "0")
        code.ctor_init("element_per_access_a", "-1")
        code.ctor_init("element_per_access_b", "-1")
        code.ctor_init("element_per_access_c", "-1")
        code.ctor_init("access_per_vector", "1")

        return code

    @pccm.pybind.mark
    @pccm.member_function(name="__repr__")
    def repr(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        check_valid();
        std::stringstream ss;
        ss << algo << "_" << tv::dtype_short_str(dtype_a) << tv::dtype_short_str(dtype_b)
            << tv::dtype_short_str(dtype_c) << tv::dtype_short_str(dacc) << tv::dtype_short_str(dcomp);
        ss << (trans_a() ? "n" : "t") << (trans_b() ? "n" : "t") << (trans_c() ? "n" : "t");
        ss << "_m" << tile_shape[0] << "n" << tile_shape[1] << "k" << tile_shape[2];
        ss << "m" << warp_tile_shape[0] << "n" << warp_tile_shape[1] << "k" << warp_tile_shape[2];
        ss << "A" << access_per_vector;
        if (tensorop[0] != -1){{
            ss << "T" << tensorop[0] << tensorop[1] << tensorop[2];
        }}
        if (shuffle_type != "{ShuffleStrideType.NoShuffle}"){{
            ss << "_" << shuffle_type;
        }}
        ss << (split_k_serial() ? 1 : 0) << (split_k_parallel() ? 1 : 0);
        return ss.str();
        """)
        return code.ret("std::string")

    @pccm.pybind.mark_prop_getter(prop_name="split_k_serial")
    @pccm.member_function
    def split_k_serial(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        return split_k_serial_ == 1;
        """)
        return code.ret("bool")

    @pccm.pybind.mark_prop_setter(prop_name="split_k_serial")
    @pccm.member_function
    def split_k_serial_set(self):
        code = pccm.FunctionCode()
        code.arg("val", "bool")
        code.raw(f"""
        split_k_serial_ = val ? 1 : 0;
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="split_k_parallel")
    @pccm.member_function
    def split_k_parallel(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        return split_k_parallel_ == 1;
        """)
        return code.ret("bool")

    @pccm.pybind.mark_prop_setter(prop_name="split_k_parallel")
    @pccm.member_function
    def split_k_parallel_set(self):
        code = pccm.FunctionCode()
        code.arg("val", "bool")
        code.raw(f"""
        split_k_parallel_ = val ? 1 : 0;
        """)
        return code

    @pccm.pybind.mark
    @pccm.member_function
    def check_valid(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        TV_ASSERT_RT_ERR(trans_a_ != -1 && !trans_b_ != -1 && trans_c_ != -1 && !algo.empty(), 
            "trans_a, trans_b, trans_c and algo must be set");
        for (int i = 0; i < 3; ++i){{
            TV_ASSERT_RT_ERR(tile_shape[i] > 0 && warp_tile_shape[i] > 0, 
                "tile_shape and warp_tile_shape must be set, but they are", tile_shape, warp_tile_shape);
        }}
        if (algo != "{GemmAlgo.Simt.value}" && algo != "{GemmAlgo.SimtDP4A.value}" && algo != "{GemmAlgo.SimtDP2A.value}"){{
            // tensor op must not empty
            for (int i = 0; i < 3; ++i){{
                TV_ASSERT_RT_ERR(tensorop[i] > 0, 
                    "tensorop must be set, but they are", tensorop);
            }}
        }}
        TV_ASSERT_RT_ERR(dtype_a != int(tv::unknown) && dtype_b != int(tv::unknown) && dtype_c != int(tv::unknown), 
            "dacc and dcomp must be set to valid value");

        TV_ASSERT_RT_ERR(dacc != int(tv::unknown) && dcomp != int(tv::unknown), "dacc and dcomp must be set to valid value");
        TV_ASSERT_RT_ERR(num_stage > 0, "num_stage must larger than zero");
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="trans_a")
    @pccm.member_function
    def trans_a(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        return trans_a_ == 1;
        """)
        return code.ret("bool")

    @pccm.pybind.mark_prop_setter(prop_name="trans_a")
    @pccm.member_function
    def trans_a_set(self):
        code = pccm.FunctionCode()
        code.arg("val", "bool")
        code.raw(f"""
        trans_a_ = val ? 1 : 0;
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="trans_b")
    @pccm.member_function
    def trans_b(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        return trans_b_ == 1;
        """)
        return code.ret("bool")

    @pccm.pybind.mark_prop_setter(prop_name="trans_b")
    @pccm.member_function
    def trans_b_set(self):
        code = pccm.FunctionCode()
        code.arg("val", "bool")
        code.raw(f"""
        trans_b_ = val ? 1 : 0;
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="trans_c")
    @pccm.member_function
    def trans_c(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        return trans_c_ == 1;
        """)
        return code.ret("bool")

    @pccm.pybind.mark_prop_setter(prop_name="trans_c")
    @pccm.member_function
    def trans_c_set(self):
        code = pccm.FunctionCode()
        code.arg("val", "bool")
        code.raw(f"""
        trans_c_ = val ? 1 : 0;
        """)
        return code

    @pccm.pybind.mark 
    @pccm.member_function
    def query_workspace_size(self):
        code = pccm.FunctionCode()
        code.arg("m,n,k,split_k_slices", "int")
        code.raw(f"""
        auto logical_tile_count =  GemmUtilsCPU::get_logical_tile_count(m, n, k, tile_shape[0], tile_shape[1], split_k_slices);
        int workspace_size = 0;
        if (split_k_slices > 1){{
            if (split_k_serial()){{
                workspace_size = sizeof(int) * logical_tile_count[0] * logical_tile_count[1];
            }} else if (split_k_parallel()){{
                workspace_size = tv::detail::sizeof_dtype(tv::DType(dacc)) * m * n * logical_tile_count[2];
            }} else{{
                TV_THROW_INVALID_ARG("not impemented");
            }}
        }}
        return workspace_size;
        """)
        return code.ret("int")

    @pccm.pybind.mark 
    @pccm.member_function
    def supported(self):
        code = pccm.FunctionCode()
        code.arg("m,n,k", "int")
        code.raw(f"""
        bool res = true;
        auto lda = trans_a() ? m : k;
        auto ldb = trans_b() ? k : n;
        auto ldc = trans_c() ? m : n;
        if (element_per_access_a > 0){{
            res &= lda % element_per_access_a == 0;
        }}
        if (element_per_access_b > 0){{
            res &= ldb % element_per_access_b == 0;
        }}
        if (element_per_access_c > 0){{
            res &= ldc % element_per_access_c == 0;
        }}
        return res;
        """)
        return code.ret("bool")

    @pccm.pybind.mark 
    @pccm.member_function
    def supported_ldx(self):
        code = pccm.FunctionCode()
        code.arg("lda, ldb, ldc", "int")
        code.raw(f"""
        bool res = true;
        if (element_per_access_a > 0){{
            res &= lda % element_per_access_a == 0;
        }}
        if (element_per_access_b > 0){{
            res &= ldb % element_per_access_b == 0;
        }}
        if (element_per_access_c > 0){{
            res &= ldc % element_per_access_c == 0;
        }}
        return res;
        """)
        return code.ret("bool")

class GemmParams(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, GemmAlgoDesp)
        # why not std::optional?
        # c++17 requires cuda11. to support cuda 10.2, we can't use c++17 for now.
        self.add_pybind_member("algo_desp",
                               "GemmAlgoDesp",
                               pyanno="GemmAlgoDesp")
        self.add_member("a,b,c", "tv::Tensor", pyanno="cumm.tensorview.Tensor")
        self.add_pybind_member("split_k_slices", "int", "1")
        self.add_pybind_member("workspace",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        # for spatial sparse convolution (split kernel algorithm)
        self.add_pybind_member("a_inds",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("b_inds",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("c_inds",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("alpha,beta", "float")
        self.add_pybind_member("stream", "std::uintptr_t", pyanno="int")
        self.add_pybind_member("timer", "tv::CUDAKernelTimer", "tv::CUDAKernelTimer(false)", pyanno="cumm.tensorview.CUDAKernelTimer")


    @pccm.pybind.mark
    @pccm.constructor
    def default_ctor(self):
        code = pccm.FunctionCode()
        code.arg("timer", "tv::CUDAKernelTimer", "tv::CUDAKernelTimer(false)", pyanno="cumm.tensorview.CUDAKernelTimer = CUDAKernelTimer(False)")
        code.ctor_init("a", "tv::Tensor()")
        code.ctor_init("b", "tv::Tensor()")
        code.ctor_init("c", "tv::Tensor()")
        code.ctor_init("split_k_slices", "1")
        code.ctor_init("workspace", "tv::Tensor()")
        code.ctor_init("a_inds", "tv::Tensor()")
        code.ctor_init("b_inds", "tv::Tensor()")
        code.ctor_init("c_inds", "tv::Tensor()")
        code.ctor_init("alpha", "1.0")
        code.ctor_init("beta", "0.0")
        code.ctor_init("stream", "0")
        code.ctor_init("timer", "timer")

        return code

    @pccm.pybind.mark
    @pccm.member_function
    def check_valid(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        algo_desp.check_valid();
        TV_ASSERT_RT_ERR(!a.empty() && !b.empty() && !c.empty(), 
            "a,b,c must not empty");
        if (algo_desp.shuffle_type == "{ShuffleStrideType.ShuffleAC.value}"){{
            TV_ASSERT_RT_ERR(!c_inds.empty(), "a_inds,c_inds tensor must not empty");
        }}else if (algo_desp.shuffle_type == "{ShuffleStrideType.ShuffleAB.value}"){{
            TV_ASSERT_RT_ERR(!a_inds.empty() && !b_inds.empty(), "a_inds,b_inds tensor must not empty");
        }}
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="a")
    @pccm.member_function
    def a_get(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        return a;
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark_prop_setter(prop_name="a")
    @pccm.member_function
    def a_set(self):
        code = pccm.FunctionCode()
        code.arg("val", "tv::Tensor")
        code.raw(f"""
        a = val;
        algo_desp.dtype_a = int(a.dtype());
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="b")
    @pccm.member_function
    def b_get(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        return b;
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark_prop_setter(prop_name="b")
    @pccm.member_function
    def b_set(self):
        code = pccm.FunctionCode()
        code.arg("val", "tv::Tensor")
        code.raw(f"""
        b = val;
        algo_desp.dtype_b = int(b.dtype());
        """)
        return code

    @pccm.pybind.mark_prop_getter(prop_name="c")
    @pccm.member_function
    def c_get(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        return c;
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark_prop_setter(prop_name="c")
    @pccm.member_function
    def c_set(self):
        code = pccm.FunctionCode()
        code.arg("val", "tv::Tensor")
        code.raw(f"""
        c = val;
        algo_desp.dtype_c = int(c.dtype());
        """)
        return code


SHUFFLE_SIMT_PARAMS: List[GemmAlgoParams] = [
    *gen_shuffle_params(
        (64, 128, 32), (32, 64, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"],
        2, kernel.GemmAlgo.SimtDP4A, None),
    *gen_shuffle_params(
        (128, 64, 32), (64, 32, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"],
        2, kernel.GemmAlgo.SimtDP4A, None),
    *gen_shuffle_params(
        (128, 128, 32),
        (32, 64, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2,
        kernel.GemmAlgo.SimtDP4A, None),
    *gen_shuffle_params(
        (128, 128, 32),
        (64, 32, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2,
        kernel.GemmAlgo.SimtDP4A, None),
    *gen_shuffle_params(
        (64, 64, 32), (32, 32, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"],
        2, kernel.GemmAlgo.SimtDP4A, None),
    *gen_shuffle_params(
        (64, 256, 8),
        (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (64, 256, 8),
        (64, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (32, 128, 16),
        (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (32, 512, 8),
        (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (128, 128, 8),
        (64, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (128, 128, 8),
        (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (64, 128, 8),
        (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (64, 128, 8),
        (64, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (128, 64, 8),
        (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (128, 64, 8),
        (64, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (64, 64, 8),
        (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (32, 64, 16),
        (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (64, 32, 16),
        (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (32, 32, 32),
        (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    # fall back kernels if mat is misaligned for half
    *gen_shuffle_params(
        (128, 128, 8),
        (32, 64, 8), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (32, 64, 32),
        (32, 32, 8), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (32, 32, 32),
        (32, 32, 8), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (64, 64, 16),
        (32, 32, 8), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (64, 128, 16),
        (32, 64, 8), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (64, 64, 8),
        (32, 32, 8), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
]

SHUFFLE_VOLTA_PARAMS: List[GemmAlgoParams] = [
    *gen_shuffle_params(
        (64, 64, 32),
        (32, 32, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
    *gen_shuffle_params(
        (128, 128, 32),
        (64, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
    *gen_shuffle_params(
        (128, 256, 32),
        (64, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
    *gen_shuffle_params(
        (256, 128, 32),
        (64, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
    *gen_shuffle_params(
        (128, 64, 32),
        (64, 32, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
    *gen_shuffle_params(
        (64, 128, 32),
        (32, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
]
# SHUFFLE_VOLTA_PARAMS = []
SHUFFLE_TURING_PARAMS: List[GemmAlgoParams] = [
    *gen_shuffle_params(
        (64, 64, 32),
        (32, 32, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    *gen_shuffle_params(
        (128, 128, 32),
        (32, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    *gen_shuffle_params(
        (128, 128, 32),
        (64, 32, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    *gen_shuffle_params(
        (64, 64, 64),
        (32, 32, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    *gen_shuffle_params(
        (64, 128, 64),
        (32, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    *gen_shuffle_params(
        (128, 256, 32),
        (64, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    *gen_shuffle_params(
        (256, 128, 32),
        (64, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    *gen_shuffle_params(
        (128, 64, 32),
        (64, 32, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    *gen_shuffle_params(
        (64, 128, 32),
        (32, 64, 32), ["f16,f16,f16,f16,f16", "f16,f16,f16,f32,f32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
    *gen_shuffle_params(
        (64, 64, 32), (32, 32, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"],
        2, kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
    *gen_shuffle_params(
        (128, 128, 32),
        (32, 64, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
    *gen_shuffle_params(
        (128, 128, 32),
        (64, 32, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
    *gen_shuffle_params(
        (128, 256, 32),
        (64, 64, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
    *gen_shuffle_params(
        (256, 128, 32),
        (64, 64, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2,
        kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
    *gen_shuffle_params(
        (128, 64, 32), (64, 32, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"],
        2, kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
    *gen_shuffle_params(
        (64, 128, 32), (32, 64, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"],
        2, kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
]
# SHUFFLE_TURING_PARAMS = []

class GemmMainUnitTest(pccm.ParameterizedClass):
    def __init__(self, gemm_params: Optional[List[GemmAlgoParams]] = None):
        super().__init__()
        self.add_dependency(TensorView, GemmBasic, GemmParams)
        if gemm_params is None:
            is_debug = os.getenv("CUMM_DEBUG", None)
            if is_debug is not None and is_debug == "1":
                simt_params = [
                    # *gen_gemm_params((64, 128, 32), (32, 64, 32), 2, "s8,s8,s32,s32,s32", kernel.GemmAlgo.SimtDP4A, None),
                    # *gen_gemm_params((64, 64, 16),
                    #                  (32, 32, 16), 2, "f16,f16,f16,f16,f16",
                    #                  kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params((64, 64, 32),
                    #                  (32, 32, 32), 2, "s8,s8,s32,s32,s32",
                    #                  kernel.GemmAlgo.SimtDP4A, None),
                    # *gen_gemm_params((128, 128, 8),
                    #                 (32, 64, 8), 2, "f32,f32,f32,f32,f32",
                    #                 kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params((128, 128, 8),
                    #                 (32, 64, 8), 2, "f32,f32,f32,f32,f32",
                    #                 kernel.GemmAlgo.Simt, None, shuffle_stride=ShuffleStrideType.ShuffleAB, splitk_serial=True),
                    # *gen_gemm_params((32, 128, 16),
                    #                 (32, 32, 8), 2, "f32,f32,f32,f32,f32",
                    #                 kernel.GemmAlgo.Simt, None, splitk_serial=True),
                    # *gen_shuffle_params(
                    #     (64, 32, 16),
                    #     (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
                    # *gen_shuffle_params(
                    #     (32, 512, 8),
                    #     (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
                    # *gen_shuffle_params(
                    #     (64, 32, 16),
                    #     (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params((32, 512, 8),
                    #                 (32, 64, 8), 2, "f32,f32,f32,f32,f32",
                    #                 kernel.GemmAlgo.Simt, None, splitk_serial=True),

                    # *gen_gemm_params((32, 128, 16),
                    #                  (32, 32, 8), 2, "f32,f32,f32,f32,f32",
                    #                  kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params((64, 128, 32),
                    #                  (32, 64, 32), 2, "s8,s8,s32,s32,s32",
                    #                  kernel.GemmAlgo.SimtDP4A, None, shuffle_stride=ShuffleStrideType.ShuffleAC),
                    # *gen_gemm_params((32, 32, 32),
                    #                  (32, 32, 8), 2, "f32,f32,f32,f32,f32",
                    #                  kernel.GemmAlgo.Simt, None, shuffle_stride=ShuffleStrideType.ShuffleAC, splitk_serial=False),
                    # *gen_gemm_params((128, 64, 32),
                    #                  (64, 32, 32), 2, "s8,s8,s32,s32,s32",
                    #                  kernel.GemmAlgo.SimtDP4A, None, shuffle_stride=ShuffleStrideType.ShuffleAC, splitk_serial=False),

                    # *gen_gemm_params_rowmajor_c((8, 32, 8), (8, 32, 8), 2, "f32,f32,f32,f32,f32", kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((16, 32, 8), (16, 32, 8), 2, "f32,f32,f32,f32,f32", kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((8, 32, 8), (8, 16, 8), 2, "f32,f32,f32,f32,f32", kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((8, 64, 8), (8, 32, 8), 2, "f32,f32,f32,f32,f32", kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((16, 32, 8), (16, 16, 8), 2, "f32,f32,f32,f32,f32", kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((64, 128, 8), (32, 32, 8), 2, "f32,f32,f32,f32,f32", kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((128, 128, 8), (32, 64, 8), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((128, 128, 8), (32, 64, 8), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((128, 128, 16), (32, 64, 16), 2, "f32,f32,f32,f32,f32", kernel.GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((128, 128, 16), (32, 64, 16), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Simt, None),
                ]  # type: List[GemmAlgoParams]
                volta_params = [
                    # *gen_gemm_params_rowmajor_c((128, 64, 32), (64, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params((64, 64, 32),
                    #                  (32, 32, 32), 2, "f16,f16,f32,f32,f32",
                    #                  kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params_rowmajor_c((128, 128, 32), (64, 64, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                ]
                turing_params = [
                    # *gen_gemm_params_rowmajor_c((128, 64, 32), (64, 32, 32),
                    #                             2,
                    #                             "f16,f16,f16,f32,f32",
                    #                             kernel.GemmAlgo.Turing,
                    #                             TensorOpParams((16, 8, 8)),
                    #                             splitk_serial=True),
                    # *gen_gemm_params(
                    #     (64, 64, 32),
                    #     (64, 64, 16), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing,
                    #     TensorOpParams((16, 8, 8))),
                    # interleave = 4:
                    # *gen_gemm_params((128, 64, 32), (64, 32, 32), 2, "s8,s8,s8,s32,s32", kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 16))),

                    *gen_gemm_params((64, 64, 16), (32, 32, 16), 2, "tf32,tf32,tf32,tf32,tf32", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                    # *gen_gemm_params((64, 128, 64), (32, 64, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),

                    # *gen_gemm_params_rowmajor_c((64, 128, 32), (32, 64, 32), 2, "f16,f16,f32,f32,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                    # *gen_gemm_params_rowmajor_c((128, 256, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                    # *gen_gemm_params_rowmajor_c((256, 128, 32), (64, 64, 32), 2, "f16,f16,f32,f32,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                    # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                    # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, None),
                ]
                # self.turing_s8_params = [
                #     *gen_gemm_params((128, 128, 64), (64, 64, 64), 2, "s8,s8,s8,s32,f32", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                #     # *gen_gemm_params_rowmajor_c((64, 128, 32), (32, 64, 32), 2, "f16,f16,f32,f32,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                #     # *gen_gemm_params_rowmajor_c((128, 256, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                #     # *gen_gemm_params_rowmajor_c((256, 128, 32), (64, 64, 32), 2, "f16,f16,f32,f32,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                #     # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                #     # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, None),
                # ]
                # print(self.simt_params[0].dtype_b)
                # raise NotImplementedError
            else:
                simt_params = [
                    *SHUFFLE_SIMT_PARAMS,
                    # *gen_gemm_params_rowmajor_c((128, 128, 8), (32, 64, 8), 2, "f32,f32,f32,f32,f32", kernel.GemmAlgo.Simt, None, splitk_serial=True),

                    # *gen_gemm_params((64, 128, 32), (32, 64, 32), 2, "s8,s8,s32,s32,s32", kernel.GemmAlgo.SimtDP4A, None),
                ]  # type: List[GemmAlgoParams]
                volta_params = [
                    *SHUFFLE_VOLTA_PARAMS,

                    # *gen_gemm_params_rowmajor_c((128, 128, 32), (32, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params_rowmajor_c((64, 64, 64), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params((128, 256, 32),
                    #                  (64, 64, 32), 2, "f16,f16,f16,f16,f16",
                    #                  kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params_rowmajor_c((128, 128, 32), (64, 64, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                    # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                ]
                turing_params = [
                    *SHUFFLE_TURING_PARAMS,
                    # *gen_gemm_params_rowmajor_c((128, 256, 32), (64, 64, 32), 2, "s8,s8,s8,s32,s32", kernel.GemmAlgo.Turing, TensorOpParams((8, 8, 16))),
                    # *gen_gemm_params_rowmajor_c((128, 64, 32), (64, 32, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 8)), splitk_serial=True),

                    # *gen_gemm_params(
                    #     (64, 64, 32),
                    #     (64, 64, 16), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing,
                    #     TensorOpParams((16, 8, 8))),
                    # interleave = 4:
                    # *gen_gemm_params((128, 64, 32), (64, 32, 32), 2, "s8,s8,s8,s32,s32", kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 16))),

                    # *gen_gemm_params((64, 64, 32), (32, 32, 32), 2, "tf32,tf32,tf32,tf32,tf32", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                    # *gen_gemm_params((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 16])),

                    # *gen_gemm_params_rowmajor_c((64, 128, 32), (32, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                    # *gen_gemm_params_rowmajor_c((128, 256, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                    # *gen_gemm_params_rowmajor_c((256, 128, 32), (64, 64, 32), 2, "f16,f16,f32,f32,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                    # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                    # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, None),
                ]

            self.all_params = simt_params + volta_params + turing_params
            self.all_kernels = [gen_gemm_kernels(p) for p in self.all_params]
        else:
            assert len(gemm_params) > 0
            self.all_params = gemm_params
            self.all_kernels = [gen_gemm_kernels(p) for p in self.all_params]
        self.ker_names = [k.get_algo_name() for k in self.all_kernels]
        assert len(set(self.ker_names)) == len(self.ker_names), "kernel must unique"
        # self.add_impl_only_dependency(self.shuffle_matmul_ref, SpconvKernel)

    @staticmethod
    def matmul_select_helper_base(kernels: List[Union[kernel.GemmKernel, Any]], code: FunctionCode):
        """if based algorithm selector
        """
        tabc_to_kers = codeops.group_by(
            lambda x: (x.trans_a, x.trans_b, x.trans_c), kernels)
        func: Callable[[kernel.GemmKernel], Tuple[int, int, int]] = lambda x: (
            x.warp_tile_shape[0], x.warp_tile_shape[1], x.warp_tile_shape[2])
        func2: Callable[[kernel.GemmKernel], Tuple[
            int, int, int]] = lambda x: (x.num_stage, x.dtype_acc.tv_dtype, x.
                                         dtype_comp.tv_dtype)

        for tabc, tabc_kers in tabc_to_kers.items():
            if_tests = [
                f"algo_desp.trans_a() == {pccm.boolean(tabc[0])}",
                f"algo_desp.trans_b() == {pccm.boolean(tabc[1])}",
                f"algo_desp.trans_c() == {pccm.boolean(tabc[2])}",
            ]
            with code.if_(" && ".join(if_tests)):
                ts_to_kers = codeops.group_by(
                    lambda x:
                    (x.tile_shape[0], x.tile_shape[1], x.tile_shape[2]),
                    tabc_kers)
                for ts, ts_kers in ts_to_kers.items():
                    with code.if_(
                            f"algo_desp.tile_shape == std::array<int, 3>{{{ts[0]}, {ts[1]}, {ts[2]}}}"
                    ):
                        wts_to_kers = codeops.group_by(func, ts_kers)
                        for wts, wts_kers in wts_to_kers.items():
                            with code.if_(
                                    f"algo_desp.warp_tile_shape == std::array<int, 3>{{{wts[0]}, {wts[1]}, {wts[2]}}}"
                            ):
                                saccomp_to_kers = codeops.group_by(
                                    func2, wts_kers)
                                for saccomp, saccomp_kers in saccomp_to_kers.items(
                                ):
                                    saccomp_if_tests = [
                                        f"algo_desp.num_stage == {saccomp[0]}",
                                        f"algo_desp.dacc == {saccomp[1]}",
                                        f"algo_desp.dcomp == {saccomp[2]}",
                                    ]
                                    with code.if_(
                                            " && ".join(saccomp_if_tests)):
                                        spks_to_kers = codeops.group_by(
                                            lambda k: k.splitk_serial,
                                            saccomp_kers)
                                        for spks, spks_kers in spks_to_kers.items(
                                        ):
                                            if spks:
                                                spks_if_test = f"(params.split_k_slices > 1)"
                                            else:
                                                spks_if_test = f"(params.split_k_slices == 1)"
                                            with code.if_(spks_if_test):
                                                yield spks_kers
                                        # iterate spkers again, run splitk>1 algo for splitk==1 inputs
                                        for spks, spks_kers in spks_to_kers.items(
                                        ):
                                            if spks:
                                                yield spks_kers
    @staticmethod
    def matmul_select_helper_stage2(kernels: List[Union[kernel.GemmKernel, Any]], code: FunctionCode, has_shuffle: bool = True, is_end: bool = True):
        func3: Callable[[kernel.GemmKernel], Optional[Tuple[
            int, int, int]]] = lambda x: (x.tensorop[0], x.tensorop[
                1], x.tensorop[2]) if x.tensorop is not None else None

        for spks_kers in GemmMainUnitTest.matmul_select_helper_base(kernels, code):
            apv_to_kers = codeops.group_by(lambda x: x.access_per_vector, spks_kers)
            for apv, apv_kers in apv_to_kers.items():
                if_test = f"algo_desp.access_per_vector == {apv}"
                with code.if_(if_test):
                    top_to_kers = codeops.group_by(func3, apv_kers)
                    for top, top_kers in top_to_kers.items():
                        algo_to_kers = codeops.group_by(
                            lambda x: (x.algo, x.shuffle_stride), top_kers)

                        if top is None:
                            for (algo, shuf), algo_kers in algo_to_kers.items():
                                assert algo == GemmAlgo.Simt or algo == GemmAlgo.SimtDP4A or algo == GemmAlgo.SimtDP2A
                                if_test = f"algo_desp.algo == \"{algo.value}\""
                                if has_shuffle:
                                    if_test += f"&& algo_desp.shuffle_type == \"{shuf.value}\""
                                with code.if_(if_test):
                                    dabc_to_kers = codeops.group_by(
                                        lambda x: (x.dtype_a.tv_dtype, x.dtype_b.
                                                tv_dtype, x.dtype_c.tv_dtype),
                                        algo_kers)
                                    for dabc, dabc_kers in dabc_to_kers.items():
                                        if is_end:
                                            assert len(
                                                dabc_kers
                                            ) == 1, "find multiple kernels for one configuration"
                                        dtype_if_tests = [
                                            f"algo_desp.dtype_a == tv::DType({dabc[0]})",
                                            f"algo_desp.dtype_b == tv::DType({dabc[1]})",
                                            f"algo_desp.dtype_c == tv::DType({dabc[2]})",
                                        ]
                                        with code.if_(" && ".join(dtype_if_tests)):
                                            yield dabc_kers
                        else:
                            with code.if_(
                                    f"algo_desp.tensorop == std::array<int, 3>{{{top[0]}, {top[1]}, {top[2]}}}"
                            ):
                                for (algo, shuf), algo_kers in algo_to_kers.items():
                                    assert algo != GemmAlgo.Simt and algo != GemmAlgo.SimtDP4A and algo != GemmAlgo.SimtDP2A
                                    if_test = f"algo_desp.algo == \"{algo.value}\""
                                    if has_shuffle:
                                        if_test += f"&& algo_desp.shuffle_type == \"{shuf.value}\""
                                    with code.if_(if_test):
                                        dabc_to_kers = codeops.group_by(
                                            lambda x: (x.dtype_a.tv_dtype, x.dtype_b.
                                                    tv_dtype, x.dtype_c.tv_dtype),
                                            algo_kers)
                                        for dabc, dabc_kers in dabc_to_kers.items():
                                            if is_end:
                                                assert len(
                                                    dabc_kers
                                                ) == 1, "find multiple kernels for one configuration"
                                            dtype_if_tests = [
                                                f"algo_desp.dtype_a == tv::DType({dabc[0]})",
                                                f"algo_desp.dtype_b == tv::DType({dabc[1]})",
                                                f"algo_desp.dtype_c == tv::DType({dabc[2]})",
                                            ]
                                            with code.if_(" && ".join(dtype_if_tests)):
                                                yield dabc_kers

    @pccm.pybind.mark
    @pccm.static_function
    def get_all_algo_desp(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        std::vector<GemmAlgoDesp> desps;
        """)
        for ker in self.all_kernels:
            code.raw("{")
            code.raw(f"""
            GemmAlgoDesp desp;
            desp.dtype_a = {ker.dtype_a.tv_dtype};
            desp.dtype_b = {ker.dtype_b.tv_dtype};
            desp.dtype_c = {ker.dtype_c.tv_dtype};
            desp.dacc = {ker.dtype_acc.tv_dtype};
            desp.dcomp = {ker.dtype_comp.tv_dtype};

            desp.trans_a_set({pccm.boolean(ker.trans_a)});
            desp.trans_b_set({pccm.boolean(ker.trans_b)});
            desp.trans_c_set({pccm.boolean(ker.trans_c)});
            desp.tile_shape = {{{ker.tile_shape[0]}, {ker.tile_shape[1]}, {ker.tile_shape[2]}}};
            desp.warp_tile_shape = {{{ker.warp_tile_shape[0]}, {ker.warp_tile_shape[1]}, {ker.warp_tile_shape[2]}}};
            """)
            if ker.tensorop is not None:
                code.raw(
                    f"desp.tensorop = {{{ker.tensorop[0]}, {ker.tensorop[1]}, {ker.tensorop[2]}}};"
                )
            else:
                code.raw(f"desp.tensorop = {{-1, -1, -1}};")
            code.raw(f"""
            desp.num_stage = {ker.num_stage};
            desp.algo = "{ker.algo.value}";
            desp.split_k_serial_set({pccm.boolean(ker.splitk_serial)});
            desp.split_k_parallel_set({pccm.boolean(ker.splitk_parallel)});
            desp.shuffle_type = "{ker.shuffle_stride.value}";
            desp.element_per_access_a = {ker.input_spec.input_iter_a.element_per_acc};
            desp.element_per_access_b = {ker.input_spec.input_iter_b.element_per_acc};
            desp.element_per_access_c = {ker.output_spec.out_iter.element_per_acc};
            desp.access_per_vector = {ker.access_per_vector};
            desps.push_back(desp);
            """)
            code.raw("}")
        code.raw(f"""
        return desps;
        """)
        return code.ret("std::vector<GemmAlgoDesp>")

    @pccm.pybind.mark
    @pccm.static_function
    def extract_mnk(self):
        code = pccm.FunctionCode()
        code.arg("a_shape,b_shape", "std::vector<int64_t>")
        code.arg("trans_a,trans_b,trans_c", "bool")
        code.arg("shuffle_type", "std::string", f"\"{ShuffleStrideType.NoShuffle.value}\"")

        code.arg("a_inds_shape", "std::vector<int64_t>", "std::vector<int64_t>{}", pyanno="List[int] = []")
        code.arg("b_inds_shape", "std::vector<int64_t>", "std::vector<int64_t>{}", pyanno="List[int] = []")
        code.arg("c_inds_shape", "std::vector<int64_t>", "std::vector<int64_t>{}", pyanno="List[int] = []")

        # TODO spatial sparse conv (implicit gemm)
        code.raw(f"""
        if (trans_c) {{
            trans_a = !trans_a;
            trans_b = !trans_b;
            std::swap(trans_a, trans_b);
            std::swap(a_shape, b_shape);
        }}
        int m, n, k, k2;
        if (shuffle_type == "{ShuffleStrideType.ShuffleAC.value}"){{
            TV_ASSERT_RT_ERR(!trans_a, "a of shuffle AB must be row major");
            TV_ASSERT_RT_ERR(!c_inds_shape.empty(), "c_inds must not empty");
            if (!a_inds_shape.empty()){{
                m = a_inds_shape[0];
            }}else{{
                m = a_shape[0];
            }}
            k = a_shape[int(!trans_a)];
            k2 = b_shape[(int(trans_b))];
            n = b_shape[(int(!trans_b) )];
        }}
        else if (shuffle_type == "{ShuffleStrideType.ShuffleAB.value}"){{
            TV_ASSERT_RT_ERR(!a_inds_shape.empty() && !b_inds_shape.empty(), "a_inds and c_inds must not empty");
            TV_ASSERT_RT_ERR(trans_a && !trans_b, "shuffle AB must be nt, i.e. backward weight");
            m = a_shape[(int(trans_a))];
            k = a_inds_shape[0];
            k2 = b_inds_shape[0];
            n = b_shape[(int(!trans_b) )];
        }}
        else{{
            m = a_shape[int(trans_a)];
            k = a_shape[(int(!trans_a))];
            k2 = b_shape[(int(trans_b))];
            n = b_shape[(int(!trans_b) )];
        }}
        return std::make_tuple(m, n, k);
        """)
        return code.ret("std::tuple<int, int, int>")

    @pccm.pybind.mark
    @pccm.static_function
    def align_to_power2(self):
        code = pccm.FunctionCode()
        code.arg("val", "int")
        code.raw(f"""
        size_t r = 0;
        size_t num_1_bit = val & 1 ? 1 : 0;
        while (val >>= 1) {{
            r++;
            if (val & 1) {{
                ++num_1_bit;
            }}
        }}
        if (num_1_bit == 1) {{
            return 1 << r;
        }} else {{
            return 1 << (r + 1);
        }}
        """)
        return code.ret("int")

    @pccm.pybind.mark
    @pccm.static_function
    def device_synchronize(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        checkCudaErrors(cudaDeviceSynchronize());
        """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def stream_synchronize(self):
        code = pccm.FunctionCode()
        code.arg("stream", "std::uintptr_t", pyanno="int")
        code.raw(f"""
        auto res = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
        if (res){{
            TV_THROW_RT_ERR("CUDA error", int(res));
        }}
        """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def simple_select_tile_shape(self):
        code = pccm.FunctionCode()
        code.arg("m,n,k", "int")
        code.arg("tile_ms", "std::vector<int>")
        code.arg("tile_ns", "std::vector<int>")
        code.arg("tile_ks", "std::vector<int>")

        code.arg("tile_shape_to_algos", "std::unordered_map<int64_t, std::vector<int>>")
        code.arg("large_k_first", "bool")

        code.raw(f"""
        auto iter_m_target = std::lower_bound(tile_ms.begin(), tile_ms.end(), m);
        auto iter_n_target = std::lower_bound(tile_ns.begin(), tile_ns.end(), n);
        auto iter_k_target = std::lower_bound(tile_ks.begin(), tile_ks.end(), k);
        if (iter_m_target == tile_ms.end()){{
            iter_m_target = tile_ms.end() - 1;
        }}
        if (iter_n_target == tile_ns.end()){{
            iter_n_target = tile_ns.end() - 1;
        }}
        if (iter_k_target == tile_ks.end()){{
            iter_k_target = tile_ks.end() - 1;
        }}
        // tv::ssprint(*iter_m_target, *iter_n_target, *iter_k_target);
        // try to find a valid configuration
        if (large_k_first){{
            for (auto iter_k = iter_k_target; iter_k != tile_ks.begin() - 1; --iter_k){{
                for (auto iter_n = iter_n_target; iter_n != tile_ns.begin() - 1; --iter_n){{
                    for (auto iter_m = iter_m_target; iter_m != tile_ms.begin() - 1; --iter_m){{
                        int64_t tm = *iter_m;
                        int64_t tn = *iter_n;
                        int64_t tk = *iter_k;
                        int64_t tile_key = tm | (tn << 20) | (tk << 40);
                        auto target_iter = tile_shape_to_algos.find(tile_key);
                        if (target_iter != tile_shape_to_algos.end()){{
                            return target_iter->second;
                        }}
                    }}
                }}
            }}
        }}
        else{{
            for (auto iter_m = iter_m_target; iter_m != tile_ms.begin() - 1; --iter_m){{
                for (auto iter_n = iter_n_target; iter_n != tile_ns.begin() - 1; --iter_n){{
                    for (auto iter_k = iter_k_target; iter_k != tile_ks.begin() - 1; --iter_k){{
                        int64_t tm = *iter_m;
                        int64_t tn = *iter_n;
                        int64_t tk = *iter_k;
                        int64_t tile_key = tm | (tn << 20) | (tk << 40);
                        auto target_iter = tile_shape_to_algos.find(tile_key);
                        if (target_iter != tile_shape_to_algos.end()){{
                            return target_iter->second;
                        }}
                    }}
                }}
            }}
        }}
        return {{}};
        """)
        return code.ret("std::vector<int>")


    @pccm.pybind.mark
    @pccm.cuda.static_function(name="matmul2")
    def matmul2(self):
        code = pccm.FunctionCode()
        for p, ker in zip(self.all_params, self.all_kernels):
            code.add_param_class("gp" + ker.get_algo_name(), ker.gemm_params,
                                 "GemmParams" + ker.get_algo_name())
            code.add_param_class(ker.get_algo_name(), ker,
                                 "Gemm" + ker.get_algo_name())
        code.arg("params", "GemmParams", pyanno="GemmParams")
        # TODO spatial sparse conv (implicit gemm)
        code.raw(f"""
        params.check_valid();
        auto& algo_desp = params.algo_desp;
        bool found = false;
        auto dacc = tv::DType(algo_desp.dacc);
        auto dcomp = tv::DType(algo_desp.dcomp);
        auto a = params.a;
        auto b = params.b;
        auto c = params.c;
        auto ta = algo_desp.trans_a();
        auto tb = algo_desp.trans_b();
        auto tc = algo_desp.trans_c();
        tv::check_shape(a, {{-1, -1}});
        tv::check_shape(b, {{-1, -1}});
        tv::check_shape(c, {{-1, -1}});
        tv::check_eq_device(a, b, c);
        tv::Tensor a_ten = a;
        tv::Tensor b_ten = b;
        tv::Tensor c_ten = c;
        auto trans_a = ta;
        auto trans_b = tb;
        auto trans_c = tc;
        if (tc) {{
            trans_a = !trans_a;
            trans_b = !trans_b;
            std::swap(trans_a, trans_b);
            std::swap(a_ten, b_ten);
        }}
        int split_k_slices = params.split_k_slices;
        auto workspace = params.workspace;
        auto a_inds = params.a_inds;
        auto c_inds = params.c_inds;
        auto b_inds = params.b_inds;
        auto& evtimer = params.timer;
        """)
        for kers in self.matmul_select_helper_stage2(self.all_kernels, code):
            ker = kers[0]
            param_type_str = "GemmParams" + ker.get_algo_name()
            code.raw(f"""
            found = true;
            """)
            if not ker.support_splitk():
                code.raw(f"""
                if (split_k_slices > 1){{
                    TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
                }}
                """)
            code.raw(f"""
            """)
            if ker.shuffle_stride == ShuffleStrideType.ShuffleAC:
                code.raw(f"""
                TV_ASSERT_RT_ERR(!trans_a, "a of shuffle AB must be row major");
                int m;
                if (!a_inds.empty()){{
                    m = a_inds.dim(0);
                }}else{{
                    m = a.dim(0);
                }}
                TV_ASSERT_RT_ERR(int64_t(a.dim(0)) * int64_t(a.dim(1)) * {ker.dtype_a.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                    "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");

                int k = a_ten.dim(int(!trans_a));
                int k2 = b_ten.dim(int(trans_b));
                int n = b_ten.dim(int(!trans_b) );
                if (trans_c){{
                    tv::check_shape(c_ten, {{-1, m}});
                }}else{{
                    tv::check_shape(c_ten, {{-1, n}});
                }}
                """)
            elif ker.shuffle_stride == ShuffleStrideType.ShuffleAB:
                code.raw(f"""
                TV_ASSERT_RT_ERR(trans_a && !trans_b, "shuffle AB must be nt, i.e. backward weight");
                auto m = a_ten.dim(int(trans_a));
                auto k = a_inds.dim(0);
                auto k2 = b_inds.dim(0);
                auto n = b_ten.dim(int(!trans_b) );
                TV_ASSERT_RT_ERR(int64_t(a.dim(0)) * int64_t(a.dim(1)) * {ker.dtype_a.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                    "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                TV_ASSERT_RT_ERR(int64_t(b.dim(0)) * int64_t(b.dim(1)) * {ker.dtype_b.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                    "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                if (trans_c){{
                    tv::check_shape(c_ten, {{n, m}});
                }}else{{
                    tv::check_shape(c_ten, {{m, n}});
                }}
                """)
            else:
                code.raw(f"""
                auto m = a_ten.dim(int(trans_a));
                auto k = a_ten.dim(int(!trans_a));
                auto k2 = b_ten.dim(int(trans_b));
                auto n = b_ten.dim(int(!trans_b) );
                if (trans_c){{
                    tv::check_shape(c_ten, {{n, m}});
                }}else{{
                    tv::check_shape(c_ten, {{m, n}});
                }}
                """)

            code.raw(f"""
            TV_ASSERT_INVALID_ARG(algo_desp.supported(m, n, k), "this m, n, k isn't supported due to misaligned contiguous dim.")
            TV_ASSERT_INVALID_ARG(k == k2, "error");
            int workspace_size = algo_desp.query_workspace_size(m, n, k, split_k_slices);
            auto ctx = tv::Context();
            ctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(params.stream));
            if (workspace_size > 0){{
                if (!workspace.empty()){{
                    workspace.zero_(ctx);
                    TV_ASSERT_RT_ERR(workspace.nbytes() >= workspace_size, 
                        "workspace at least", workspace_size, "bytes.");
                }}else{{
                    workspace = tv::empty({{workspace_size}}, tv::uint8, 0);
                    workspace.zero_(ctx);
                }}
            }}

            """)
            if CUTLASS_MODE:
                code.raw(f"""
                CutlassGemm::ThreadblockSwizzle threadblock_swizzle;

                cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
                    {{m, n, k}}, 
                    {{{ker.tile_shape[0]}, {ker.tile_shape[1]}, {ker.tile_shape[2]}}},
                    1);

                CutlassGemm::GemmKernel::Params params{{
                    {{m, n, k}},
                    grid_shape,
                    {{a_ten.data_ptr<{ker.dtype_a}>(), a_ten.stride(0)}},
                    {{b_ten.data_ptr<{ker.dtype_b}>(), b_ten.stride(0)}},
                    {{c_ten.data_ptr<{ker.dtype_c}>(), c_ten.stride(0)}},
                    {{c_ten.data_ptr<{ker.dtype_c}>(), c_ten.stride(0)}},
                }};
                dim3 grid = threadblock_swizzle.get_grid_shape(params.grid_tiled_shape);
                tv::cuda::Launch launcher(grid, dim3({ker.num_threads}, 1, 1),
                                            {ker.smem_size});
                cudaError_t result;
                if ({ker.smem_size} >= (48 << 10)) {{
                    result = cudaFuncSetAttribute({ker.get_algo_name()}::gemm_kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    {ker.smem_size});
                    TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                    result = cudaFuncSetAttribute(
                        {ker.get_algo_name()}::gemm_kernel,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                    TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                }}

                launcher({ker.get_algo_name()}::gemm_kernel, params);
                """)
            else:
                if ker.shuffle_stride == ShuffleStrideType.ShuffleAC:
                    code.raw(f"""
                    const int* a_ptr = nullptr;
                    if (!a_inds.empty()){{
                        a_ptr = a_inds.data_ptr<const int>();
                    }}
                    TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
                    {param_type_str} kernel_params(
                        m, n, k, a_ten.data_ptr<{ker.dtype_a}>(), b_ten.data_ptr<{ker.dtype_b}>(),
                        c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(), 
                        a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), c_ten.stride(0), 
                        a_ptr, c_inds.data_ptr<const int>(),
                        {ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta), split_k_slices{", workspace.raw_data()" if ker.support_splitk() else ""});
                    """)
                elif ker.shuffle_stride == ShuffleStrideType.ShuffleAB:
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(!a_inds.empty() && !b_inds.empty(), "error");
                    {param_type_str} kernel_params(
                        m, n, k, a_ten.data_ptr<{ker.dtype_a}>(), b_ten.data_ptr<{ker.dtype_b}>(),
                        c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(), 
                        a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), c_ten.stride(0), 
                        a_inds.data_ptr<const int>(), b_inds.data_ptr<const int>(),
                        {ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta), split_k_slices{", workspace.raw_data()" if ker.support_splitk() else ""});
                    """)
                else:
                    code.raw(f"""
                    {param_type_str} kernel_params(
                        m, n, k, a_ten.data_ptr<{ker.dtype_a}>(), b_ten.data_ptr<{ker.dtype_b}>(),
                        c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(), 
                        a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), c_ten.stride(0), 
                        {ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta), split_k_slices{", workspace.raw_data()" if ker.support_splitk() else ""});
                    """)

                code.raw(f"""
                tv::cuda::Launch launcher(kernel_params.grid_dims, dim3({ker.num_threads}),
                                            {ker.smem_size}, reinterpret_cast<cudaStream_t>(params.stream));
                cudaError_t result;
                if ({ker.smem_size} >= (48 << 10)) {{
                    result = cudaFuncSetAttribute({ker.get_algo_name()}::gemm_kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    {ker.smem_size});
                    TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                    result = cudaFuncSetAttribute(
                        {ker.get_algo_name()}::gemm_kernel,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                    TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                }}
                """)
                if cudasim.enable_debug():
                    code.raw(f"""
                    auto timer = tv::CudaContextTimer<>();
                    """)

                code.raw(f"""
                {{
                    tv::CUDAKernelTimerGuard timerguard(\"{ker.get_algo_name()}\", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                    launcher({ker.get_algo_name()}::gemm_kernel, kernel_params);
                }}
                TV_CHECK_CUDA_ERR_V2("{ker.get_algo_name()}", "error with params", a.shape(), b.shape(), c.shape());
                """)
                if cudasim.enable_debug():
                    code.raw(f"""
                    cudaFuncAttributes attr;
                    checkCudaErrors(
                        cudaFuncGetAttributes(&attr, {ker.get_algo_name()}::gemm_kernel));
                    tv::ssprint("{ker.get_algo_name()} kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);
                    """)
                code.raw(f"return;")
        code.raw("""
        if (!found){
            TV_THROW_INVALID_ARG("Can't Found Algorithm for params:", algo_desp.tile_shape, algo_desp.warp_tile_shape, 
                algo_desp.num_stage, tv::dtype_str(a.dtype()), 
                tv::dtype_str(b.dtype()), tv::dtype_str(c.dtype()), tv::dtype_str(dacc), 
                tv::dtype_str(dcomp), ta, tb, tc, algo_desp.algo, algo_desp.tensorop);
        }
        // return 0;
        """)
        return code# .ret("float")

    # @pccm.expose_main.mark
    # @pccm.cuda.static_function
    def main_function(self):
        code = pccm.FunctionCode()
        code.ret("int")
        code.raw(f"""
        tv::ssprint("?");
        int m = 4096;
        int n = 4096;
        int k = 4096;
        bool trans_a = false;
        bool trans_b = false;
        bool trans_c = false;

        tv::DType a_dtype = tv::float16;
        tv::DType b_dtype = tv::float16;
        tv::DType c_dtype = tv::float16;
        // 32 * 16 * (128 * 4096 + 256 * 4096) * 2
        tv::TensorShape a_shape{{m, k}};
        if (trans_a){{
            a_shape = {{k, m}};
        }}
        tv::TensorShape b_shape{{k, n}};
        if (trans_b){{
            b_shape = {{n, k}};
        }}
        tv::TensorShape c_shape{{m, n}};
        if (trans_c){{
            c_shape = {{n, m}};
        }}
        tv::Tensor a, b;
        tv::ssprint("?");

        if (a_dtype == tv::int8){{
            a = tv::Tensor(a_shape, a_dtype, -1);
            b = tv::Tensor(b_shape, b_dtype, -1);
            a.rand_int_(-2, 2);
            b.rand_int_(-2, 2);
        }}else{{
            tv::Tensor a_f32 = tv::Tensor(a_shape, tv::float32, -1);
            tv::Tensor b_f32 = tv::Tensor(b_shape, tv::float32, -1);
            a_f32.rand_();
            b_f32.rand_();
            a = tv::Tensor(a_shape, a_dtype, -1);
            b = tv::Tensor(b_shape, b_dtype, -1);
            auto a_f32_ptr = a_f32.data_ptr<float>();
            auto b_f32_ptr = b_f32.data_ptr<float>();
            auto a_ptr = a.data_ptr<tv::half_t>();
            auto b_ptr = b.data_ptr<tv::half_t>();
            // auto a_ptr = a.data_ptr<float>();
            // auto b_ptr = b.data_ptr<float>();

            for (int i = 0; i < a.size(); ++i){{
                a_ptr[i] = a_f32_ptr[i];
            }}
            for (int i = 0; i < b.size(); ++i){{
                b_ptr[i] = b_f32_ptr[i];
            }}

            // a = a.astype(a_dtype);
            // b = b.astype(b_dtype);
        }}
        tv::ssprint("?");

        auto c = tv::Tensor(c_shape, c_dtype, 0);
        a = a.cuda();
        b = b.cuda();
        try {{
            matmul(a, b, c, trans_a, trans_b, trans_c, {{128, 256, 32}}, {{64, 64, 32}}, 2, tv::float16, tv::float16, "Turing", {{16, 8, 8}});
            // matmul(a, b, c, trans_a, trans_b, trans_c, {{128, 128, 8}}, {{32, 64, 8}}, 2, tv::float32, tv::float32, "Simt", {{0, 0, 0}});
        }}
        catch (std::exception e){{
            tv::ssprint(e.what());
            return -1;
        }}
        auto c_cpu = c.cpu();
        tv::ssprint("?");
        return 0;
        """)
        return code

    # @lineprof.lineprof_wrapper_cpp
    def matmul_python(self,
                      a: np.ndarray,
                      b: np.ndarray,
                      c: np.ndarray,
                      a_meta: np.ndarray,
                      b_meta: np.ndarray,
                      ta: bool,
                      tb: bool,
                      tc: bool,
                      ts: np.ndarray,
                      wts: np.ndarray,
                      num_stage: int,
                      dacc: dtypes.DType,
                      dcomp: dtypes.DType,
                      algo: str,
                      tensorop: np.ndarray,
                      split_k_slice: int = 1):
        found = False
        for p, ker in zip(self.all_params, self.all_kernels):
            if_tests = [
                p.dtype_a.npdtype() == a.dtype,
                p.dtype_b.npdtype() == b.dtype,
                p.dtype_c.npdtype() == c.dtype,
                p.trans_a is ta,
                p.trans_b is tb,
                p.trans_c is tc,
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
                a_ten = a
                b_ten = b
                c_ten = c
                a_meta_ten = a_meta
                b_meta_ten = b_meta

                trans_a = ta
                trans_b = tb
                trans_c = tc
                if tc:
                    trans_a = not trans_a
                    trans_b = not trans_b
                    tmp = trans_a
                    trans_a = trans_b
                    trans_b = tmp
                    tmp = a_ten
                    a_ten = b_ten
                    b_ten = tmp
                    a_meta_ten, b_meta_ten = b_meta_ten, a_meta_ten
                m = a_ten.shape[int(trans_a)]
                k = a_ten.shape[int(not trans_a)]
                k2 = b_ten.shape[int(trans_b)]
                assert k2 == k
                n = b_ten.shape[int(not trans_b)]
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
                params = ker.gemm_params.python_ctor(
                    m,
                    n,
                    k,
                    a_ptr,
                    b_ptr,
                    c_ptr,
                    c_ptr,
                    1.0,
                    0.0,
                    split_k_slice=split_k_slice)
                func = partial(ker.gemm_kernel_python, params=params)
                blocks = params.grid_dims
                threads = cudasim.Dim3(ker.num_threads, 1, 1)
                print("Simulation", blocks, threads)
                return asyncio.run(
                    cudasim.kernel_launch(func, blocks, threads,
                                          ker.smem_size)), blocks, threads
        raise NotImplementedError
