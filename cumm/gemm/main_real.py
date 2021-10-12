import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

os.environ["CUMM_DEBUG"] = "0"
# _cudart = ctypes.CDLL('libcudart.so')

import pickle
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pccm
import torch
from pccm.core import CodeFormatter

from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.gemm import kernel
from cumm.gemm.algospec.core import ShuffleStrideType
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from cumm.gemm.gather import GatherKernel
from cumm.gemm.main import GemmMainUnitTest, gen_gemm_kernels

# def cu_prof_start():
#     ret = _cudart.cudaProfilerStart()
#     if ret != 0:
#         raise Exception('cudaProfilerStart() returned %d' % ret)

# def cu_prof_stop():
#     ret = _cudart.cudaProfilerStop()
#     if ret != 0:
#         raise Exception('cudaProfilerStop() returned %d' % ret)

_SPCONV_ROOT = Path(__file__).parent.parent.parent
_GEMM_ROOT = _SPCONV_ROOT / "src/spgemm/gemmdev"
_REFGEMM_ROOT = _SPCONV_ROOT / "src/spgemm/refops"


def build_gemm_lib(cus: List[pccm.Class]):
    lib = pccm.builder.build_pybind(cus,
                                    Path(__file__).parent / "mygemm_test",
                                    build_dir=Path(__file__).parent / "build" /
                                    "build_unittest",
                                    pybind_file_suffix=".cc",
                                    verbose=False)

    return lib


def _asdv_test_simt():
    # /usr/local/cuda/bin/nvprof -o out_lidardet.nvvp --profile-from-start off --analysis-metrics -f --csv python /home/yy/OneDrive/dev/spconv/spconv/gemm/main_real.py
    # sudo PATH=$PATH:/usr/bin:/home/yy/library/anaconda3/bin:/usr/local/cuda/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 /usr/local/cuda/bin/nvprof --events shared_ld_bank_conflict,shared_st_bank_conflict /home/yy/OneDrive/dev/spconv/spconv/gemm/main_real.py
    # sudo PATH=$PATH:/usr/bin:/home/yy/library/anaconda3/bin:/usr/local/cuda/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 /usr/local/cuda/bin/nvprof -o out_lidardet.nvvp --print-gpu-trace  --analysis-metrics -f --csv /home/yy/OneDrive/dev/spconv/spconv/gemm/main_real.py
    np.random.seed(12315)
    main_cu = GemmMainUnitTest()

    lib = build_gemm_lib([main_cu])
    lib_object = lib.cumm.gemm.main.GemmMainUnitTest()
    params_cls = lib.cumm.gemm.main.GemmParams

    for params in main_cu.simt_params:
        if params.shuffle_stride != ShuffleStrideType.NoShuffle:
            continue
        ker = gen_gemm_kernels(params)
        if params.algo != kernel.GemmAlgo.SimtDP4A:
            m = 256 + 32
            n = 256 + 40
            k = 136
            m *= 2
            n *= 2
            k *= 2
            m = 316
            n = 235
            k = 658
            a = np.random.uniform(-1, 1, size=[m, k]).astype(
                dtypes.get_npdtype(params.dtype_a))
            b = np.random.uniform(-1, 1, size=[k, n]).astype(
                dtypes.get_npdtype(params.dtype_b))
        else:
            m = 256 + 32
            n = 256 + 40
            k = 136
            # m *= 2
            # n *= 2
            # k *= 2
            a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
            b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
        # print("DATA GEN FINISH")
        c = (a @ b).astype(dtypes.get_npdtype(params.dtype_c))
        if params.trans_a:
            a = np.ascontiguousarray(a.transpose(1, 0))
        if params.trans_b:
            b = np.ascontiguousarray(b.transpose(1, 0))
        if params.trans_c:
            c = np.ascontiguousarray(c.transpose(1, 0))
        # print("WTF PREPARED")
        # cu_prof_start()
        a_tv = tv.from_numpy(a).cuda()
        b_tv = tv.from_numpy(b).cuda()
        c_tv = tv.zeros(c.shape, params.dtype_c.tv_dtype, 0)
        params_cpp = params_cls()
        params_cpp.tile_shape = params.ts
        params_cpp.warp_tile_shape = params.wts
        params_cpp.num_stage = params.num_stage
        params_cpp.dacc = params.dtype_acc.tv_dtype
        params_cpp.dcomp = params.dtype_comp.tv_dtype
        params_cpp.algo = params.algo.value
        params_cpp.split_k_slices = 1
        params_cpp.a = a_tv
        params_cpp.b = b_tv
        params_cpp.c = c_tv
        params_cpp.trans_a = params.trans_a
        params_cpp.trans_b = params.trans_b
        params_cpp.trans_c = params.trans_c

        # print("CUDA PREPARED")
        lib_object.matmul2(params_cpp)
        c_cpu = c_tv.cpu().numpy()
        # cu_prof_stop()

        print(params.get_algo_name(), np.linalg.norm(c_cpu - c))


def _asdv_test_simt_shuffle():
    # /usr/local/cuda/bin/nvprof -o out_lidardet.nvvp --profile-from-start off --analysis-metrics -f --csv python /home/yy/OneDrive/dev/spconv/spconv/gemm/main_real.py
    # sudo PATH=$PATH:/usr/bin:/home/yy/library/anaconda3/bin:/usr/local/cuda/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 /usr/local/cuda/bin/nvprof --events shared_ld_bank_conflict,shared_st_bank_conflict /home/yy/OneDrive/dev/spconv/spconv/gemm/main_real.py
    # sudo PATH=$PATH:/usr/bin:/home/yy/library/anaconda3/bin:/usr/local/cuda/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 /usr/local/cuda/bin/nvprof -o out_lidardet.nvvp --print-gpu-trace  --analysis-metrics -f --csv /home/yy/OneDrive/dev/spconv/spconv/gemm/main_real.py
    np.random.seed(12315)
    main_cu = GemmMainUnitTest()
    main_cu.namespace = "cumm.gemm.main"
    gather_cu = GatherKernel(dtypes.float32, seq(32, 128), 1, 512)
    gather_cu.namespace = "rtx"
    print("BUILD FINISHTED")
    lib = build_gemm_lib([main_cu])
    lib_object = lib.cumm.gemm.main.GemmMainUnitTest()
    params_cls = lib.cumm.gemm.main.GemmParams
    algo_cls = lib.cumm.gemm.main.GemmAlgoDesp
    # gather_lib = lib_object.rtx.GatherKernel()

    for params, ker in zip(main_cu.all_params, main_cu.all_kernels):
        if params.shuffle_stride == ShuffleStrideType.NoShuffle:
            continue
        print("RTX", params.shuffle_stride, params.trans_a, params.trans_b,
              params.trans_c)
        m = 64000
        n = 128
        k = 128
        a_inds = np.arange(m, dtype=np.int32)
        c_inds = np.arange(m, dtype=np.int32)
        np.random.shuffle(a_inds)
        np.random.shuffle(c_inds)

        if params.dtype_a == dtypes.int8:
            a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
            if params.shuffle_stride != ShuffleStrideType.ShuffleAB:
                b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
                c_tmp = (a[a_inds].astype(np.float32) @ b.astype(
                    np.float32)).astype(dtypes.get_npdtype(params.dtype_c))
                c = np.zeros_like(c_tmp)
                c[c_inds] = c_tmp
            else:
                b = np.random.randint(-2, 2, size=[m, n]).astype(np.int8)
                c = (a[a_inds].T.astype(np.float32) @ b[c_inds].astype(
                    np.float32)).astype(dtypes.get_npdtype(params.dtype_c))

        else:
            a = np.random.uniform(-1, 1, size=[m, k]).astype(
                dtypes.get_npdtype(params.dtype_a))
            if params.shuffle_stride != ShuffleStrideType.ShuffleAB:
                b = np.random.uniform(-1, 1, size=[k, n]).astype(
                    dtypes.get_npdtype(params.dtype_b))
                c_tmp = (a[a_inds].astype(np.float32) @ b.astype(
                    np.float32)).astype(dtypes.get_npdtype(params.dtype_c))
                c = np.zeros_like(c_tmp)
                c[c_inds] = c_tmp
            else:
                b = np.random.uniform(-1, 1, size=[m, n]).astype(
                    dtypes.get_npdtype(params.dtype_b))
                c = (a[a_inds].T.astype(np.float32) @ b[c_inds].astype(
                    np.float32)).astype(dtypes.get_npdtype(params.dtype_c))
        if params.trans_b:
            b = np.ascontiguousarray(b.T)

        a_tv = tv.from_numpy(a).cuda()
        b_tv = tv.from_numpy(b).cuda()
        c_tv = tv.from_numpy(c).cuda()
        a_inds_tv = tv.from_numpy(a_inds).cuda()
        c_inds_tv = tv.from_numpy(c_inds).cuda()
        if params.splitk_serial:
            ksplit = 16
        else:
            ksplit = 1
        # print("CUDA PREPARED")
        algo = algo_cls()
        algo.tile_shape = params.ts
        algo.warp_tile_shape = params.wts
        algo.num_stage = params.num_stage
        algo.dacc = params.dtype_acc.tv_dtype
        algo.dcomp = params.dtype_comp.tv_dtype
        algo.algo = params.algo.value
        algo.shuffle_type = params.shuffle_stride.value
        algo.trans_a = params.trans_a
        algo.trans_b = params.trans_b
        algo.trans_c = params.trans_c
        if params.tensorop is not None:
            algo.tensorop = params.tensorop.shape
        params_cpp = params_cls()
        params_cpp.algo_desp = algo
        params_cpp.split_k_slices = ksplit
        params_cpp.a = a_tv
        params_cpp.b = b_tv
        params_cpp.c = c_tv

        for i in range(3):
            # gather_lib.gather(a_gout_tv, a_tv, a_inds_tv)
            # a_gout_tv_cpu = a_gout_tv.cpu().numpy()
            # print(np.linalg.norm(a_gout_tv_cpu - a[a_inds]))
            # c_tv.zero_()
            # lib_object.shuffle_matmul_ref(c_tv_f32, a_tv_f32, b_tv_transposed, a_inds_tv, c_inds_tv, a_inds_tv.dim(0))
            if params.shuffle_stride == ShuffleStrideType.ShuffleAB:
                params_cpp.a_inds = a_inds_tv
                params_cpp.b_inds = c_inds_tv
                c_tv.zero_()
                lib_object.matmul2(params_cpp)
            else:
                c_tv.zero_()
                params_cpp.a_inds = a_inds_tv
                params_cpp.c_inds = c_inds_tv
                lib_object.matmul2(params_cpp)
        assert params.get_algo_name() == ker.get_algo_name()
        # print(a_tv.shape, b_tv.shape, c_tv.shape)
        c_cpu = c_tv.cpu().numpy()
        # cu_prof_stop()

        print(params.dtype_c, params.get_algo_name(),
              np.linalg.norm(c_cpu - c))


COUNTBIT_LOOKUP_TABLE = np.array([bin(i).count('1')
                                  for i in range(256)]).astype(np.int32)


def count_set_bits(v):
    'returns the count of set bits in each element of v'
    assert v.dtype in (np.uint8, np.uint16, np.uint32,
                       np.uint64), 'must be an unsigned int dtype'
    return COUNTBIT_LOOKUP_TABLE[np.reshape(v.view(np.uint8),
                                            v.shape + (-1, ))].sum(axis=-1)


def _asdv_test_regular_gemm():
    np.random.seed(12315)
    with cudasim.enter_debug_context(True, 3):
        main_cu = GemmMainUnitTest()
        main_cu.namespace = "cumm.gemm.main"
        lib = build_gemm_lib([main_cu])
    lib_object = lib.cumm.gemm.main.GemmMainUnitTest()
    params_cls = lib.cumm.gemm.main.GemmParams
    algo_cls = lib.cumm.gemm.main.GemmAlgoDesp

    for params in main_cu.all_params:
        if params.shuffle_stride != ShuffleStrideType.NoShuffle:
            continue

        m = 256 + 32
        n = 256 + 40
        k = 136
        m *= 2
        n *= 2
        k *= 2
        m = 64
        n = 64
        k = 64
        m = max(params.ts[0], m)
        n = max(params.ts[1], n)
        k = max(params.ts[2], k)

        if params.dtype_a == dtypes.int8:
            a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
            b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
            dtype_c = params.dtype_c.npdtype()
            c = (a.astype(np.float32) @ b.astype(np.float32)).astype(
                dtypes.get_npdtype(params.dtype_c))
        else:
            a = np.random.uniform(-1, 1, size=[m, k]).astype(
                dtypes.get_npdtype(params.dtype_a))
            b = np.random.uniform(-1, 1, size=[k, n]).astype(
                dtypes.get_npdtype(params.dtype_b))
            # a[:, 32:] = 0
            # b[32:] = 0
            c = (a.astype(np.float32) @ b.astype(np.float32)).astype(
                dtypes.get_npdtype(params.dtype_c))
        # print("DATA GEN FINISH")
        if params.trans_a:
            a = np.ascontiguousarray(a.transpose(1, 0))
        if params.trans_b:
            b = np.ascontiguousarray(b.transpose(1, 0))
        if params.trans_c:
            c = np.ascontiguousarray(c.transpose(1, 0))
        # print("WTF PREPARED")
        if params.splitk_serial:
            ksplit = 2
        else:
            ksplit = 1
        # print("CUDA PREPARED")
        algo = algo_cls()
        algo.tile_shape = params.ts
        algo.warp_tile_shape = params.wts
        algo.num_stage = params.num_stage
        algo.dacc = params.dtype_acc.tv_dtype
        algo.dcomp = params.dtype_comp.tv_dtype
        algo.algo = params.algo.value
        algo.trans_a = params.trans_a
        algo.trans_b = params.trans_b
        algo.trans_c = params.trans_c
        if params.tensorop is not None:
            algo.tensorop = params.tensorop.shape
        params_cpp = params_cls()
        params_cpp.algo_desp = algo
        params_cpp.split_k_slices = ksplit
        a_tv = tv.from_numpy(a).cuda()
        b_tv = tv.from_numpy(b).cuda()

        c_tv = tv.zeros(c.shape, params.dtype_c.tv_dtype, 0)

        params_cpp.a = a_tv
        params_cpp.b = b_tv
        params_cpp.c = c_tv

        # print("CUDA PREPARED")
        lib_object.matmul2(params_cpp)
        c_cpu = c_tv.cpu().numpy()
        print(c_cpu.reshape(-1)[-16:])
        print(c.reshape(-1)[-16:])
        print(params.get_algo_name(), a.mean(), b.mean(), c.mean(),
              np.linalg.norm(c_cpu - c))


def _asdv_test_turing():
    np.random.seed(12315)
    main_cu = GemmMainUnitTest()
    lib = build_gemm_lib([main_cu])
    lib_object = lib.cumm.gemm.main.GemmMainUnitTest()
    for params in main_cu.turing_params:
        m = 256 + 32
        n = 256 + 40
        k = 136
        m *= 2
        n *= 2
        k *= 2
        m = 1024
        n = 1024
        k = 1024
        m = max(params.ts[0], m)
        n = max(params.ts[1], n)
        k = max(params.ts[2], k)
        # print("HAHAHA PREPARED")

        if params.dtype_a == dtypes.int8:
            a = np.random.randint(-2, 2, size=[m, k]).astype(np.int8)
            b = np.random.randint(-2, 2, size=[k, n]).astype(np.int8)
            dtype_c = params.dtype_c.npdtype()
            c = (a.astype(dtype_c) @ b.astype(dtype_c)).astype(
                dtypes.get_npdtype(params.dtype_c))
        else:
            a = np.random.uniform(-1, 1, size=[m, k]).astype(
                dtypes.get_npdtype(params.dtype_a))
            b = np.random.uniform(-1, 1, size=[k, n]).astype(
                dtypes.get_npdtype(params.dtype_b))
            c = (a.astype(np.float32) @ b.astype(np.float32)).astype(
                dtypes.get_npdtype(params.dtype_c))
        if params.trans_a:
            a = np.ascontiguousarray(a.transpose(1, 0))
        if params.trans_b:
            b = np.ascontiguousarray(b.transpose(1, 0))
        if params.trans_c:
            c = np.ascontiguousarray(c.transpose(1, 0))
        # print("WTF PREPARED")

        a_tv = tv.from_numpy(a).cuda()
        b_tv = tv.from_numpy(b).cuda()
        c_tv = tv.zeros(c.shape, params.dtype_c.tv_dtype, 0)
        # print("CUDA PREPARED")
        lib_object.matmul(a_tv,
                          b_tv,
                          c_tv,
                          params.trans_a,
                          params.trans_b,
                          params.trans_c,
                          ts=params.ts,
                          wts=params.wts,
                          num_stage=params.num_stage,
                          dacc=params.dtype_acc.tv_dtype,
                          dcomp=params.dtype_comp.tv_dtype,
                          algo=params.algo.value,
                          tensorop=params.tensorop.shape)
        c_cpu = c_tv.cpu().numpy()
        print(params.get_algo_name(), np.linalg.norm(c_cpu - c))


if __name__ == "__main__":
    _asdv_test_simt_shuffle()
    # _asdv_test_regular_gemm()
