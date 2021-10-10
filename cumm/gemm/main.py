import asyncio
from functools import partial
from cumm.gemm.core.metaarray import MetaArray
from cumm import tensorview as tv 
from cumm.core_cc.csrc.arrayref import ArrayPtr
import sys
from pathlib import Path

from cumm.gemm.algospec.core import GemmAlgo, ShuffleStrideType, TensorOpParams
import pccm
import numpy as np
from cumm.constants import CUTLASS_MODE

from typing import Callable, Dict, Tuple, Optional, List
from cumm.common import PyBind11, TensorView, TensorViewKernel, GemmBasic
from cumm import cudasim
from cumm.gemm import codeops, kernel 
from ccimport import compat
from pccm.core import CodeFormatter, FunctionCode
# from myclang import clangformat
from cumm import dtypes
from cumm.gemm.core import metaseq, seq, MetaArray, array_type
import os 

class GemmAlgoParams(object):
    def __init__(self,
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
                 shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle):
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


def gen_gemm_params(ts,
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
    for ta in [True]:
        for tb in [False]:
            for tc in [False]:
                p = GemmAlgoParams(ts, wts, stage, dtypes_string, ta, tb, tc,
                                   algo, tensorop, splitk_serial, splitk_parallel,
                                   shuffle_stride)
                if not p.skipped():
                    res.append(p)

    return res

def gen_shuffle_params(ts,
                    wts,
                    dss: List[str],
                    stage: int,
                    algo: kernel.GemmAlgo,
                    tensorop: Optional[kernel.TensorOpParams]):
    res = []
    for ds in dss:
        for tb in [False, True]:
            p = GemmAlgoParams(ts, wts, stage, ds, False, tb, False,
                                algo, tensorop, False, False,
                                ShuffleStrideType.ShuffleAC)
            if not p.skipped():
                res.append(p)
        p = GemmAlgoParams(ts, wts, stage, ds, True, False, False,
                            algo, tensorop, True, False,
                            ShuffleStrideType.ShuffleAB)
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
    for ta in [False, True]:
        for tb in [False, True]:
            for tc in [False]:
                p = GemmAlgoParams(ts, wts, stage, dtypes_string, ta, tb, tc,
                                   algo, tensorop, splitk_serial, splitk_parallel)
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
                             shuffle_stride=params.shuffle_stride)

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

class GemmAlgoDesp(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
        # why not std::optional?
        # c++17 requires cuda11. to support cuda 10.2, we can't use c++17 for now.
        self.add_member("dtype_a,dtype_b,dtype_c", "int") # -1 means unset
        self.add_member("trans_a_,trans_b_,trans_c_", "int") # -1 means unset
        self.add_pybind_member("tile_shape,warp_tile_shape", "std::array<int, 3>", pyanno="Tuple[int, int, int]")
        self.add_pybind_member("num_stage", "int")
        self.add_pybind_member("dacc,dcomp", "int")
        self.add_pybind_member("algo", "std::string")
        self.add_pybind_member("tensorop", "std::array<int, 3>", "std::array<int, 3>{}")
        self.add_pybind_member("split_k_serial_", "int", "0") # -1 means unset
        self.add_pybind_member("split_k_parallel_", "int", "0") # -1 means unset
        self.add_pybind_member("shuffle_type", "std::string", f"\"{ShuffleStrideType.NoShuffle.value}\"")

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
        code.ctor_init("shuffle_type", f"\"{ShuffleStrideType.NoShuffle.value}\"")
        code.ctor_init("split_k_serial_", "0")
        code.ctor_init("split_k_parallel_", "0")

        return code 


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

class GemmParams(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, GemmAlgoDesp)
        # why not std::optional?
        # c++17 requires cuda11. to support cuda 10.2, we can't use c++17 for now.
        self.add_pybind_member("algo_desp", "GemmAlgoDesp", pyanno="GemmAlgoDesp")
        self.add_member("a,b,c", "tv::Tensor", pyanno="cumm.tensorview.Tensor")
        self.add_pybind_member("split_k_slices", "int", "1")
        self.add_pybind_member("workspace", "tv::Tensor", "tv::Tensor()", pyanno="cumm.tensorview.Tensor = Tensor()")
        # for spatial sparse convolution (split kernel algorithm)
        self.add_pybind_member("a_inds", "tv::Tensor", "tv::Tensor()", pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("b_inds", "tv::Tensor", "tv::Tensor()", pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("c_inds", "tv::Tensor", "tv::Tensor()", pyanno="cumm.tensorview.Tensor = Tensor()")

    @pccm.pybind.mark 
    @pccm.constructor
    def default_ctor(self):
        code = pccm.FunctionCode()
        code.ctor_init("a", "tv::Tensor()")
        code.ctor_init("b", "tv::Tensor()")
        code.ctor_init("c", "tv::Tensor()")
        code.ctor_init("split_k_slices", "1")
        code.ctor_init("workspace", "tv::Tensor()")
        code.ctor_init("a_inds", "tv::Tensor()")
        code.ctor_init("b_inds", "tv::Tensor()")
        code.ctor_init("c_inds", "tv::Tensor()")

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
            TV_ASSERT_RT_ERR(!a_inds.empty() && !c_inds.empty(), "a_inds,c_inds tensor must not empty");
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

class GemmMainUnitTest(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, GemmBasic, GemmParams)
        is_debug = os.getenv("CUMM_DEBUG", None)
        if is_debug is not None and is_debug == "1":
            self.simt_params = [
                # *gen_gemm_params((64, 128, 32), (32, 64, 32), 2, "s8,s8,s32,s32,s32", kernel.GemmAlgo.SimtDP4A, None),
                # *gen_gemm_params((64, 64, 16),
                #                  (32, 32, 16), 2, "f16,f16,f16,f16,f16",
                #                  kernel.GemmAlgo.Simt, None),
                # *gen_gemm_params((64, 64, 32),
                #                  (32, 32, 32), 2, "s8,s8,s32,s32,s32",
                #                  kernel.GemmAlgo.SimtDP4A, None),
                *gen_gemm_params((128, 128, 8),
                                (32, 64, 8), 2, "f32,f32,f32,f32,f32",
                                kernel.GemmAlgo.Simt, None),
                *gen_gemm_params((128, 128, 8),
                                (32, 64, 8), 2, "f32,f32,f32,f32,f32",
                                kernel.GemmAlgo.Simt, None, shuffle_stride=ShuffleStrideType.ShuffleAB, splitk_serial=True),

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
            self.volta_params = [
                # *gen_gemm_params_rowmajor_c((64, 64, 64), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                # *gen_gemm_params((64, 64, 32),
                #                  (32, 32, 32), 2, "f16,f16,f32,f32,f32",
                #                  kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                # *gen_gemm_params_rowmajor_c((128, 128, 32), (64, 64, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
            ]
            self.turing_params = [
                # *gen_gemm_params(
                #     (64, 64, 32),
                #     (64, 64, 16), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing,
                #     TensorOpParams((16, 8, 8))),
                # interleave = 4:
                # *gen_gemm_params((128, 64, 32), (64, 32, 32), 2, "s8,s8,s8,s32,s32", kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 16))),

                # *gen_gemm_params((64, 64, 32), (32, 32, 32), 2, "tf32,tf32,tf32,tf32,tf32", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                # *gen_gemm_params((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 16])),

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
            self.simt_params = [
                *gen_shuffle_params((64, 128, 32), (32, 64, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2, kernel.GemmAlgo.SimtDP4A, None),
                *gen_shuffle_params((128, 64, 32), (64, 32, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2, kernel.GemmAlgo.SimtDP4A, None),
                *gen_shuffle_params((128, 128, 32), (32, 64, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2, kernel.GemmAlgo.SimtDP4A, None),
                *gen_shuffle_params((128, 128, 32), (64, 32, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2, kernel.GemmAlgo.SimtDP4A, None),
                *gen_shuffle_params((64, 64, 32), (32, 32, 32), ["s8,s8,s8,s32,s32", "s8,s8,s32,s32,s32"], 2, kernel.GemmAlgo.SimtDP4A, None),
                # *gen_shuffle_params((128, 128, 8), (64, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
                # *gen_shuffle_params((128, 128, 8), (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
                # *gen_shuffle_params((64, 128, 8), (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
                # *gen_shuffle_params((64, 128, 8), (64, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
                # *gen_shuffle_params((128, 64, 8), (32, 64, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
                # *gen_shuffle_params((128, 64, 8), (64, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
                # *gen_shuffle_params((64, 64, 8), (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
                # *gen_shuffle_params((32, 64, 16), (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
                # *gen_shuffle_params((64, 32, 16), (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),
                # *gen_shuffle_params((32, 32, 32), (32, 32, 8), ["f32,f32,f32,f32,f32"], 2, kernel.GemmAlgo.Simt, None),

                # *gen_gemm_params((64, 128, 32), (32, 64, 32), 2, "s8,s8,s32,s32,s32", kernel.GemmAlgo.SimtDP4A, None),
            ]  # type: List[GemmAlgoParams]
            self.volta_params = [
                # *gen_gemm_params_rowmajor_c((64, 64, 64), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                # *gen_gemm_params((64, 64, 32),
                #                  (32, 32, 32), 2, "f16,f16,f32,f32,f32",
                #                  kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                # *gen_gemm_params_rowmajor_c((128, 128, 32), (64, 64, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
                # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f32,f32", kernel.GemmAlgo.Volta, TensorOpParams((8, 8, 4))),
            ]
            self.turing_params = [
                # *gen_gemm_params(
                #     (64, 64, 32),
                #     (64, 64, 16), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing,
                #     TensorOpParams((16, 8, 8))),
                # interleave = 4:
                # *gen_gemm_params((128, 64, 32), (64, 32, 32), 2, "s8,s8,s8,s32,s32", kernel.GemmAlgo.Turing, TensorOpParams((16, 8, 16))),

                # *gen_gemm_params((64, 64, 32), (32, 32, 32), 2, "tf32,tf32,tf32,tf32,tf32", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                # *gen_gemm_params((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 16])),

                # *gen_gemm_params_rowmajor_c((64, 128, 32), (32, 64, 32), 2, "f16,f16,f32,f32,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                # *gen_gemm_params_rowmajor_c((128, 256, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                # *gen_gemm_params_rowmajor_c((256, 128, 32), (64, 64, 32), 2, "f16,f16,f32,f32,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                # *gen_gemm_params_rowmajor_c((64, 64, 32), (32, 32, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Turing, TensorOpParams([16, 8, 8])),
                # *gen_gemm_params_rowmajor_c((64, 64, 32), (64, 64, 32), 2, "f16,f16,f16,f16,f16", kernel.GemmAlgo.Volta, None),
            ]

        self.all_params = self.simt_params + self.volta_params + self.turing_params
        self.all_kernels = [gen_gemm_kernels(p) for p in self.all_params]
        # self.add_impl_only_dependency(self.shuffle_matmul_ref, SpconvKernel)

    # @pccm.pybind.mark
    # @pccm.cuda.static_function
    # def shuffle_matmul_ref(self):
    #     code = pccm.FunctionCode()
    #     code.add_dependency(SpconvKernel)
    #     code.arg("output", "tv::Tensor")
    #     code.arg("features", "tv::Tensor")
    #     code.arg("filters", "tv::Tensor")
    #     code.arg("indicesIn", "tv::Tensor")
    #     code.arg("indicesOut", "tv::Tensor")
    #     code.arg("nHot", "int")

    #     code.raw(f"""
    #     auto in_nchannel = features.dim(1);
    #     auto out_nchannel = output.dim(1);
    #     int shared_mem_size = -1;
    #     if ((in_nchannel > 16 && out_nchannel > 16 &&
    #         in_nchannel * out_nchannel >= 512) ||
    #         (in_nchannel > 24 && out_nchannel > 24))
    #         shared_mem_size = 32;
    #     else if (in_nchannel % 24 == 0 && out_nchannel % 24 == 0)
    #         shared_mem_size = 24;
    #     else if ((in_nchannel > 8 && out_nchannel > 8) ||
    #             (in_nchannel % 16 == 0 && out_nchannel % 16 == 0))
    #         shared_mem_size = 16;
    #     else
    #         shared_mem_size = 8;
    #     constexpr int MAX_GRID = 65535;
    #     using shmem_sizes_t = tv::mp_list_c<int, 32, 24, 16, 8>;
    #     int num_grid = (nHot + shared_mem_size - 1) / shared_mem_size;
    #     int num_div = (num_grid + MAX_GRID - 1) / MAX_GRID;
    #     int step = (nHot + num_div - 1) / num_div;
    #     dim3 threads(shared_mem_size, shared_mem_size);
    #     auto timer = tv::CudaContextTimer<>();

    #     tv::dispatch<float>(output.dtype(), [&](auto I) {{
    #         using T = TV_DECLTYPE(I);
    #         tv::DispatchInt<shmem_sizes_t>()(shared_mem_size, [&](auto ShSizeValue) {{
    #             constexpr int ShmemSize = TV_DECLTYPE(ShSizeValue)::value;
    #             for (int s = 0; s < num_div; s++) {{
    #                 int remainder = nHot - step * s;
    #                 int curr_num_active = remainder < step ? remainder : step;
    #                 dim3 grid((out_nchannel + threads.x - 1) / threads.x,
    #                         (curr_num_active + threads.y - 1) / threads.y);
    #                 matmulX<T, int32_t, ShmemSize><<<grid, threads>>>(
    #                     features.data_ptr<T>(), in_nchannel, curr_num_active,
    #                     filters.data_ptr<T>(), out_nchannel, in_nchannel,
    #                     output.data_ptr<T>(), indicesIn.data_ptr<int32_t>(),
    #                     indicesOut.data_ptr<int32_t>());
    #             }}
    #         }});
    #     }});
    #     tv::ssprint("fused shuffle conv time", timer.report() / 1000.0);

    #     """)
    #     return code 

    def matmul_select_helper(self, code: FunctionCode):
        """if based algorithm selector
        """
        tabc_to_kers = codeops.group_by(lambda x: (x.trans_a, x.trans_b, x.trans_c), self.all_kernels)
        func: Callable[[kernel.GemmKernel], Tuple[int, int, int]] = lambda x: (x.warp_tile_shape[0], x.warp_tile_shape[1], x.warp_tile_shape[2])
        func2: Callable[[kernel.GemmKernel], Tuple[int, int, int]] = lambda x: (x.num_stage, x.dtype_acc.tv_dtype, x.dtype_comp.tv_dtype)
        func3: Callable[[kernel.GemmKernel], Optional[Tuple[int, int, int]]] = lambda x: (x.tensorop[0], x.tensorop[1], x.tensorop[2]) if x.tensorop is not None else None

        for tabc, tabc_kers in tabc_to_kers.items():
            if_tests = [
                f"algo_desp.trans_a() == {pccm.boolean(tabc[0])}",
                f"algo_desp.trans_b() == {pccm.boolean(tabc[1])}",
                f"algo_desp.trans_c() == {pccm.boolean(tabc[2])}",
            ]
            with code.if_(" && ".join(if_tests)):
                ts_to_kers = codeops.group_by(lambda x: (x.tile_shape[0], x.tile_shape[1], x.tile_shape[2]), tabc_kers)
                for ts, ts_kers in ts_to_kers.items():
                    with code.if_(f"algo_desp.tile_shape == std::array<int, 3>{{{ts[0]}, {ts[1]}, {ts[2]}}}"):
                        wts_to_kers = codeops.group_by(func, ts_kers)
                        for wts, wts_kers in wts_to_kers.items():
                            with code.if_(f"algo_desp.warp_tile_shape == std::array<int, 3>{{{wts[0]}, {wts[1]}, {wts[2]}}}"):
                                saccomp_to_kers = codeops.group_by(func2, wts_kers)
                                for saccomp, saccomp_kers in saccomp_to_kers.items():
                                    saccomp_if_tests = [
                                        f"algo_desp.num_stage == {saccomp[0]}",
                                        f"algo_desp.dacc == {saccomp[1]}",
                                        f"algo_desp.dcomp == {saccomp[2]}",
                                    ]
                                    with code.if_(" && ".join(saccomp_if_tests)):
                                        spks_to_kers = codeops.group_by(lambda k: k.splitk_serial, saccomp_kers)
                                        for spks, spks_kers in spks_to_kers.items():
                                            spks_if_test = f"({pccm.boolean(spks)} && params.split_k_slices > 1) || (!{pccm.boolean(spks)} && params.split_k_slices == 1)"
                                            with code.if_(spks_if_test):
                                                top_to_kers = codeops.group_by(func3, spks_kers)
                                                for top, top_kers in top_to_kers.items():
                                                    algo_to_kers = codeops.group_by(lambda x: (x.algo, x.shuffle_stride), top_kers)

                                                    if top is None:
                                                        for (algo, shuf), algo_kers in algo_to_kers.items():
                                                            assert algo == GemmAlgo.Simt or algo == GemmAlgo.SimtDP4A or algo == GemmAlgo.SimtDP2A
                                                            with code.if_(f"algo_desp.algo == \"{algo.value}\" && algo_desp.shuffle_type == \"{shuf.value}\""):
                                                                dabc_to_kers = codeops.group_by(lambda x: (x.dtype_a.tv_dtype, x.dtype_b.tv_dtype, x.dtype_c.tv_dtype), algo_kers)
                                                                for dabc, dabc_kers in dabc_to_kers.items():
                                                                    assert len(dabc_kers) == 1, "find multiple kernels for one configuration"
                                                                    dtype_if_tests = [ 
                                                                        f"algo_desp.dtype_a == tv::DType({dabc[0]})",
                                                                        f"algo_desp.dtype_b == tv::DType({dabc[1]})",
                                                                        f"algo_desp.dtype_c == tv::DType({dabc[2]})",
                                                                    ]
                                                                    with code.if_(" && ".join(dtype_if_tests)):
                                                                        yield dabc_kers[0]
                                                    else:
                                                        with code.if_(f"algo_desp.tensorop == std::array<int, 3>{{{top[0]}, {top[1]}, {top[2]}}}"):
                                                            for (algo, shuf), algo_kers in algo_to_kers.items():
                                                                assert algo != GemmAlgo.Simt and algo != GemmAlgo.SimtDP4A and algo != GemmAlgo.SimtDP2A
                                                                with code.if_(f"algo_desp.algo == \"{algo.value}\" && algo_desp.shuffle_type == \"{shuf.value}\""):
                                                                    dabc_to_kers = codeops.group_by(lambda x: (x.dtype_a.tv_dtype, x.dtype_b.tv_dtype, x.dtype_c.tv_dtype), algo_kers)
                                                                    for dabc, dabc_kers in dabc_to_kers.items():
                                                                        assert len(dabc_kers) == 1, "find multiple kernels for one configuration"
                                                                        dtype_if_tests = [ 
                                                                            f"algo_desp.dtype_a == tv::DType({dabc[0]})",
                                                                            f"algo_desp.dtype_b == tv::DType({dabc[1]})",
                                                                            f"algo_desp.dtype_c == tv::DType({dabc[2]})",
                                                                        ]
                                                                        with code.if_(" && ".join(dtype_if_tests)):
                                                                            yield dabc_kers[0]


    @pccm.pybind.mark
    @pccm.cuda.static_function
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
            desp.warp_tile_shape = {{{ker.tile_shape[0]}, {ker.tile_shape[1]}, {ker.tile_shape[2]}}};
            """)
            if ker.tensorop is not None:
                code.raw(f"desp.tensorop = {{{ker.tensorop[0]}, {ker.tensorop[1]}, {ker.tensorop[2]}}};")
            else:
                code.raw(f"desp.tensorop = {{-1, -1, -1}};")
            code.raw(f"""
            desp.num_stage = {ker.num_stage};
            desp.algo = "{ker.algo.value}";
            desp.split_k_serial_set({pccm.boolean(ker.splitk_serial)});
            desp.split_k_parallel_set({pccm.boolean(ker.splitk_parallel)});
            desp.shuffle_type = "{ker.shuffle_stride.value}";

            desps.push_back(desp);
            """)
            code.raw("}")
        code.raw(f"""
        return desps;
        """)
        return code.ret("std::vector<GemmAlgoDesp>")

    @pccm.pybind.mark
    @pccm.cuda.static_function(name="matmul2")
    def matmul2(self):
        code = pccm.FunctionCode()
        for p, ker in zip(self.all_params, self.all_kernels):
            code.add_param_class("gp" + ker.get_algo_name(), ker.gemm_params, "GemmParams" + ker.get_algo_name())
            code.add_param_class( ker.get_algo_name(), ker, "Gemm" + ker.get_algo_name())
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

        """)
        for ker in self.matmul_select_helper(code):
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
                auto m = a_inds.dim(0);
                auto k = a_ten.dim(int(!trans_a));
                auto k2 = b_ten.dim(int(trans_b));
                auto n = b_ten.dim(int(!trans_b) );
                """)
            elif ker.shuffle_stride == ShuffleStrideType.ShuffleAB:
                code.raw(f"""
                TV_ASSERT_RT_ERR(trans_a && !trans_b, "shuffle AB must be nt, i.e. backward weight");
                auto m = a_ten.dim(int(trans_a));
                auto k = a_inds.dim(0);
                auto k2 = b_inds.dim(0);
                auto n = b_ten.dim(int(!trans_b) );
                """)
            else:
                code.raw(f"""
                auto m = a_ten.dim(int(trans_a));
                auto k = a_ten.dim(int(!trans_a));
                auto k2 = b_ten.dim(int(trans_b));
                auto n = b_ten.dim(int(!trans_b) );
                """)

            code.raw(f"""
            TV_ASSERT_INVALID_ARG(k == k2, "error");
            TV_ASSERT_INVALID_ARG(a_ten.dim(1) % {ker.input_spec.input_iter_a.sub_tile_shape[1]} == 0, "error");
            TV_ASSERT_INVALID_ARG(b_ten.dim(1) % {ker.input_spec.input_iter_b.sub_tile_shape[1]} == 0, "error");


            int workspace_size = 0;
            auto logical_tile_count = {param_type_str}::get_logical_tile_count(m, n, k, split_k_slices);
            if (split_k_slices > 1){{
                if ({pccm.boolean(ker.splitk_serial)}){{
                    workspace_size = sizeof(int) * logical_tile_count.x * logical_tile_count.y;
                }} else if ({pccm.boolean(ker.splitk_parallel)}){{
                    workspace_size = {ker.dtype_acc.itemsize()} * m * n * logical_tile_count.z;
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
                    {{a_ten.data_ptr<{ker.dtype_a}>(), a_ten.size(1)}},
                    {{b_ten.data_ptr<{ker.dtype_b}>(), b_ten.size(1)}},
                    {{c_ten.data_ptr<{ker.dtype_c}>(), c_ten.size(1)}},
                    {{c_ten.data_ptr<{ker.dtype_c}>(), c_ten.size(1)}},
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
                    TV_ASSERT_RT_ERR(!a_inds.empty() && !c_inds.empty(), "error");
                    {param_type_str} params(
                        m, n, k, a_ten.data_ptr<{ker.dtype_a}>(), b_ten.data_ptr<{ker.dtype_b}>(),
                        c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(), 
                        a_inds.data_ptr<const int>(), c_inds.data_ptr<const int>(),
                        {ker.dtype_a}(1.0), {ker.dtype_b}(1.0), split_k_slices{", workspace.raw_data()" if ker.support_splitk() else ""});
                    """)
                elif ker.shuffle_stride == ShuffleStrideType.ShuffleAB:
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(!a_inds.empty() && !b_inds.empty(), "error");
                    {param_type_str} params(
                        m, n, k, a_ten.data_ptr<{ker.dtype_a}>(), b_ten.data_ptr<{ker.dtype_b}>(),
                        c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(), 
                        a_inds.data_ptr<const int>(), b_inds.data_ptr<const int>(),
                        {ker.dtype_comp}(1.0), {ker.dtype_comp}(1.0), split_k_slices{", workspace.raw_data()" if ker.support_splitk() else ""});
                    """)
                else:
                    code.raw(f"""
                    {param_type_str} params(
                        m, n, k, a_ten.data_ptr<{ker.dtype_a}>(), b_ten.data_ptr<{ker.dtype_b}>(),
                        c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(), 
                        {ker.dtype_comp}(1.0), {ker.dtype_comp}(1.0), split_k_slices{", workspace.raw_data()" if ker.support_splitk() else ""});
                    """)

                code.raw(f"""
                tv::cuda::Launch launcher(params.grid_dims, dim3({ker.num_threads}),
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
                auto timer = tv::CudaContextTimer<>();
                launcher({ker.get_algo_name()}::gemm_kernel, params);
                cudaFuncAttributes attr;
                checkCudaErrors(
                    cudaFuncGetAttributes(&attr, {ker.get_algo_name()}::gemm_kernel));
                tv::ssprint("{ker.get_algo_name()} kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);
                """)
        code.raw("""
        if (!found){
            TV_THROW_INVALID_ARG("Can't Found Algorithm for params:", algo_desp.tile_shape, algo_desp.warp_tile_shape, 
                algo_desp.num_stage, tv::dtype_str(a.dtype()), 
                tv::dtype_str(b.dtype()), tv::dtype_str(c.dtype()), tv::dtype_str(dacc), 
                tv::dtype_str(dcomp), ta, tb, tc, algo_desp.algo, algo_desp.tensorop);
        }
        """)
        return code


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
    def matmul_python(self, a: np.ndarray, b: np.ndarray, c: np.ndarray,
                      a_meta: np.ndarray, b_meta: np.ndarray, ta: bool,
                      tb: bool, tc: bool, ts: np.ndarray, wts: np.ndarray,
                      num_stage: int, dacc: dtypes.DType, dcomp: dtypes.DType,
                      algo: str, tensorop: np.ndarray):
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
                params = ker.gemm_params.python_ctor(m, n, k, a_ptr, b_ptr,
                                                     c_ptr, c_ptr, 1.0, 0.0)
                func = partial(ker.gemm_kernel_python, params=params)
                blocks = params.grid_dims
                threads = cudasim.Dim3(ker.num_threads, 1, 1)
                return asyncio.run(
                    cudasim.kernel_launch(func, blocks, threads,
                                          ker.smem_size)), blocks, threads
        raise NotImplementedError
