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

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import sys
import time
from pathlib import Path

import numpy as np
import pccm
import torch
import torch.nn.functional as F

from cumm import dtypes
from cumm import tensorview as tv
from cumm.constants import PACKAGE_ROOT
from cumm.conv.bases import NCHW, NHWC, ConvIterAlgo, ConvOpType
from cumm.conv.main import ConvMainUnitTest, gen_gemm_kernels
from cumm.conv.params import ConvProblem
from cumm.gemm import kernel
import os 
os.environ["CUMM_DEBUG"] = "1"


def _asdv_test_simt_python():
    np.random.seed(12315)
    main_cu = ConvMainUnitTest()
    lib = pccm.builder.build_pybind([main_cu],
                                    Path(__file__).parent / "imgemm_test",
                                    includes=[
                                        PACKAGE_ROOT / "include",
                                    ],
                                    namespace_root=PACKAGE_ROOT / "cumm",
                                    build_dir=Path(__file__).parent / "build" /
                                    "build_unittest_conv",
                                    pybind_file_suffix=".cc",
                                    verbose=False,
                                    disable_anno=True)

    lib_object = lib.cumm.conv.main.ConvMainUnitTest()
    for params in main_cu.all_params[:6]:
        if params.mask_sparse:
            continue
        # NCHW -> KCRS @ NCRSPQ = NKPQ
        print(params.get_algo_name())
        ndim = params.ndim
        ker = gen_gemm_kernels(params)
        # print("START", params.get_algo_name())
        if ndim == 3:
            HW = [56] * ndim
        else:
            HW = [244] * ndim

        RS = [1] * ndim
        N = 1
        C = 128
        K = 128
        padding = [RS[0] // 2] * ndim
        stride = [1] * ndim
        dilation = [1] * ndim
        out_dims = ConvProblem.calc_output_dims_python(HW, RS, padding, stride,
                                                       dilation)
        PQ = out_dims
        op_type = params.op_type
        iter_algo = ConvIterAlgo.Analytic

        if params.dtype_a == dtypes.int8:
            inp = np.random.randint(-2, 2, size=[N, *HW, C]).astype(np.int8)
            weight = np.random.randint(-2, 2, size=[K, *RS, C]).astype(np.int8)
            output = np.random.randint(-2, 2, size=[N, *PQ, K]).astype(np.int8)
            doutput = np.random.randint(-2, 2, size=[N, *PQ,
                                                     K]).astype(np.int8)

        else:
            inp = np.random.uniform(-1, 1, size=[N, *HW, C]).astype(
                dtypes.get_npdtype(params.dtype_input))
            weight = np.random.uniform(-1, 1, size=[K, *RS, C]).astype(
                dtypes.get_npdtype(params.dtype_weight))
            output = np.random.uniform(-1, 1, size=[N, *PQ, K]).astype(
                dtypes.get_npdtype(params.dtype_output))
            doutput = np.random.uniform(-1, 1, size=[N, *PQ, K]).astype(
                dtypes.get_npdtype(params.dtype_output))
        nhwc_to_nchw_inds = [0, ndim + 1, *range(1, ndim + 1)]
        nchw_to_nhwc_inds = [0, *range(2, ndim + 2), 1]

        inp_th = torch.from_numpy(inp).permute(*nhwc_to_nchw_inds).to(
            torch.float32).contiguous()
        weight_th = torch.from_numpy(weight).permute(*nhwc_to_nchw_inds).to(
            torch.float32).contiguous()
        output_th = torch.from_numpy(output).permute(*nhwc_to_nchw_inds).to(
            torch.float32).contiguous()
        doutput_th = torch.from_numpy(doutput).permute(*nhwc_to_nchw_inds).to(
            torch.float32).contiguous()

        if params.dtype_a.itemsize() != 1:
            inp_th = inp_th.cuda()
            weight_th = weight_th.cuda()
            output_th = output_th.cuda()
            doutput_th = doutput_th.cuda()

            inp_th.requires_grad = True
            weight_th.requires_grad = True
        th_t = time.time()
        if ndim == 1:
            out_ref = F.conv1d(inp_th,
                               weight_th,
                               padding=padding,
                               stride=stride,
                               dilation=dilation)
        elif ndim == 2:
            out_ref = F.conv2d(inp_th,
                               weight_th,
                               padding=padding,
                               stride=stride,
                               dilation=dilation)
        elif ndim == 3:
            out_ref = F.conv3d(inp_th,
                               weight_th,
                               padding=padding,
                               stride=stride,
                               dilation=dilation)
        else:
            raise NotImplementedError
        torch.cuda.synchronize()
        print("TORCH time", time.time() - th_t)
        th_t = time.time()
        if params.dtype_a.itemsize() != 1:
            out_ref.backward(doutput_th)
        torch.cuda.synchronize()
        print("TORCH BW time", time.time() - th_t)

        out_ref_nhwc = out_ref.detach().permute(
            *nchw_to_nhwc_inds).contiguous().cpu().numpy()
        if params.dtype_a.itemsize() != 1:

            din_ref_nhwc = inp_th.grad.detach().permute(
                *nchw_to_nhwc_inds).contiguous().cpu().numpy()
            dw_ref_nhwc = weight_th.grad.detach().permute(
                *nchw_to_nhwc_inds).contiguous().cpu().numpy()
        else:
            din_ref_nhwc = np.zeros_like(inp)
            dw_ref_nhwc = np.zeros_like(weight)

        # print("WTF PREPARED")

        if params.op_type == ConvOpType.kBackwardInput:
            inp_tv = tv.zeros(inp.shape, params.dtype_input.tv_dtype, 0)
        else:
            inp_tv = tv.from_numpy(inp).cuda()
        if params.op_type == ConvOpType.kBackwardWeight:
            weight_tv = tv.zeros(weight.shape, params.dtype_weight.tv_dtype, 0)
        else:
            weight_tv = tv.from_numpy(weight).cuda()
        if params.op_type == ConvOpType.kForward:
            output_tv = tv.zeros(output.shape, params.dtype_output.tv_dtype, 0)
        else:
            output_tv = tv.from_numpy(doutput).cuda()
        torch.cuda.synchronize()

        t = time.time()

        # print("CUDA PREPARED")
        spk = 1
        if op_type == ConvOpType.kBackwardWeight:
            # TODO support splitk parallel
            spk = 16
        for i in range(10):
            lib_object.implicit_gemm(
                inp_tv,
                weight_tv,
                output_tv,
                padding,
                stride,
                dilation,
                ndim=ndim,
                iter_algo_=params.iter_algo.value,
                op_type_=params.op_type.value,
                i_ltype_=params.layout_desp_input.layout_type.value,
                w_ltype_=params.layout_desp_weight.layout_type.value,
                o_ltype_=params.layout_desp_output.layout_type.value,
                ts=params.ts,
                wts=params.wts,
                num_stage=params.num_stage,
                dacc=params.dtype_acc.tv_dtype,
                dcomp=params.dtype_comp.tv_dtype,
                algo=params.algo.value,
                tensorop=[0, 0, 0],
                split_k_slices=spk)  # type: tv.Tensor
            print(time.time() - t)
            if i != 9:
                t = time.time()

        op_duration = time.time() - t
        if params.op_type == ConvOpType.kForward:
            output_cpu = output_tv.cpu().numpy()
            if params.dtype_a.itemsize() == 1:
                output_cpu = output_cpu.astype(np.float32)
            duration = time.time() - t
            print(output_cpu.reshape(-1)[:10], out_ref_nhwc.reshape(-1)[:10])
            print(params.get_algo_name(),
                  np.linalg.norm(out_ref_nhwc - output_cpu), "Time=",
                  op_duration)
        elif params.op_type == ConvOpType.kBackwardInput:
            print(ker.input_spec.tmap_b.iterations)
            din_cpu = inp_tv.cpu().numpy()
            duration = time.time() - t
            print(din_cpu.reshape(-1)[:10], din_ref_nhwc.reshape(-1)[:10])
            print(params.get_algo_name(),
                  np.linalg.norm(din_cpu - din_ref_nhwc), "Time=", op_duration)
        else:
            dw_cpu = weight_tv.cpu().numpy()
            duration = time.time() - t
            print(dw_cpu.reshape(-1)[:10], dw_ref_nhwc.reshape(-1)[:10])
            print(params.get_algo_name(), np.linalg.norm(dw_cpu - dw_ref_nhwc),
                  "Time=", op_duration)

def _asdv_test_simt_python_v2():
    np.random.seed(12315)
    main_cu = ConvMainUnitTest()
    lib = pccm.builder.build_pybind([main_cu],
                                    Path(__file__).parent / "imgemm_test",
                                    includes=[
                                        PACKAGE_ROOT / "include",
                                    ],
                                    namespace_root=PACKAGE_ROOT / "cumm",
                                    build_dir=Path(__file__).parent / "build" /
                                    "build_unittest_conv",
                                    pybind_file_suffix=".cc",
                                    verbose=True,
                                    disable_anno=True)

    lib_object = lib.cumm.conv.main.ConvMainUnitTest()
    algo_cls = lib.cumm.conv.main.ConvAlgoDesp
    params_cls = lib.cumm.conv.main.ConvParams

    for params in main_cu.all_params[:1]:
        if params.mask_sparse:
            continue
        # NCHW -> KCRS @ NCRSPQ = NKPQ
        print(params.get_algo_name())
        ndim = params.ndim
        ker = gen_gemm_kernels(params)
        # print("START", params.get_algo_name())
        if ndim == 3:
            HW = [56] * ndim
        else:
            HW = [244] * ndim

        RS = [1] * ndim
        N = 1
        C = 15
        K = 128
        padding = [RS[0] // 2] * ndim
        stride = [1] * ndim
        dilation = [1] * ndim
        out_dims = ConvProblem.calc_output_dims_python(HW, RS, padding, stride,
                                                       dilation)
        PQ = out_dims
        op_type = params.op_type
        iter_algo = ConvIterAlgo.Analytic

        if params.dtype_a == dtypes.int8:
            inp = np.random.randint(-2, 2, size=[N, *HW, C]).astype(np.int8)
            weight = np.random.randint(-2, 2, size=[K, *RS, C]).astype(np.int8)
            output = np.random.randint(-2, 2, size=[N, *PQ, K]).astype(np.int8)
            doutput = np.random.randint(-2, 2, size=[N, *PQ,
                                                     K]).astype(np.int8)

        else:
            inp = np.random.uniform(-1, 1, size=[N, *HW, C]).astype(
                dtypes.get_npdtype(params.dtype_input))
            weight = np.random.uniform(-1, 1, size=[K, *RS, C]).astype(
                dtypes.get_npdtype(params.dtype_weight))
            output = np.random.uniform(-1, 1, size=[N, *PQ, K]).astype(
                dtypes.get_npdtype(params.dtype_output))
            doutput = np.random.uniform(-1, 1, size=[N, *PQ, K]).astype(
                dtypes.get_npdtype(params.dtype_output))
        nhwc_to_nchw_inds = [0, ndim + 1, *range(1, ndim + 1)]
        nchw_to_nhwc_inds = [0, *range(2, ndim + 2), 1]

        inp_th = torch.from_numpy(inp).permute(*nhwc_to_nchw_inds).to(
            torch.float32).contiguous()
        weight_th = torch.from_numpy(weight).permute(*nhwc_to_nchw_inds).to(
            torch.float32).contiguous()
        output_th = torch.from_numpy(output).permute(*nhwc_to_nchw_inds).to(
            torch.float32).contiguous()
        doutput_th = torch.from_numpy(doutput).permute(*nhwc_to_nchw_inds).to(
            torch.float32).contiguous()

        if params.dtype_a.itemsize() != 1:
            inp_th = inp_th.cuda()
            weight_th = weight_th.cuda()
            output_th = output_th.cuda()
            doutput_th = doutput_th.cuda()

            inp_th.requires_grad = True
            weight_th.requires_grad = True
        th_t = time.time()
        if ndim == 1:
            out_ref = F.conv1d(inp_th,
                               weight_th,
                               padding=padding,
                               stride=stride,
                               dilation=dilation)
        elif ndim == 2:
            out_ref = F.conv2d(inp_th,
                               weight_th,
                               padding=padding,
                               stride=stride,
                               dilation=dilation)
        elif ndim == 3:
            out_ref = F.conv3d(inp_th,
                               weight_th,
                               padding=padding,
                               stride=stride,
                               dilation=dilation)
        else:
            raise NotImplementedError
        torch.cuda.synchronize()
        print("TORCH time", time.time() - th_t)
        th_t = time.time()
        if params.dtype_a.itemsize() != 1:
            out_ref.backward(doutput_th)
        torch.cuda.synchronize()
        print("TORCH BW time", time.time() - th_t)

        out_ref_nhwc = out_ref.detach().permute(
            *nchw_to_nhwc_inds).contiguous().cpu().numpy()
        if params.dtype_a.itemsize() != 1:

            din_ref_nhwc = inp_th.grad.detach().permute(
                *nchw_to_nhwc_inds).contiguous().cpu().numpy()
            dw_ref_nhwc = weight_th.grad.detach().permute(
                *nchw_to_nhwc_inds).contiguous().cpu().numpy()
        else:
            din_ref_nhwc = np.zeros_like(inp)
            dw_ref_nhwc = np.zeros_like(weight)

        # print("WTF PREPARED")

        if params.op_type == ConvOpType.kBackwardInput:
            inp_tv = tv.zeros(inp.shape, params.dtype_input.tv_dtype, 0)
        else:
            inp_tv = tv.from_numpy(inp).cuda()
        if params.op_type == ConvOpType.kBackwardWeight:
            weight_tv = tv.zeros(weight.shape, params.dtype_weight.tv_dtype, 0)
        else:
            weight_tv = tv.from_numpy(weight).cuda()
        if params.op_type == ConvOpType.kForward:
            output_tv = tv.zeros(output.shape, params.dtype_output.tv_dtype, 0)
        else:
            output_tv = tv.from_numpy(doutput).cuda()
        torch.cuda.synchronize()
        spk = 1
        if op_type == ConvOpType.kBackwardWeight:
            # TODO support splitk parallel
            spk = 32

        t = time.time()
        algo = algo_cls(ker.problem.ndim, ker.problem.op_type.value)
        algo.tile_shape = params.ts
        algo.warp_tile_shape = params.wts
        algo.num_stage = params.num_stage
        algo.dacc = params.dtype_acc.tv_dtype
        algo.dcomp = params.dtype_comp.tv_dtype
        algo.algo = params.algo.value
        algo.trans_a = params.trans_a
        algo.trans_b = params.trans_b
        algo.trans_c = params.trans_c
        algo.element_per_access_a = ker.input_spec.input_sub_tile_shape_a[1]
        algo.element_per_access_b = ker.input_spec.input_sub_tile_shape_b[1]
        algo.element_per_access_c = ker.output_spec.out_iter.element_per_acc
        algo.split_k_serial = params.splitk_serial
        algo.dtype_a = params.dtype_a.tv_dtype
        algo.dtype_b = params.dtype_b.tv_dtype
        algo.dtype_c = params.dtype_c.tv_dtype

        if params.tensorop is not None:
            algo.tensorop = params.tensorop.shape
        params_cpp = params_cls(ker.problem.ndim, ker.problem.op_type.value)
        params_cpp.conv_algo_desp = algo
        params_cpp.split_k_slices = spk
        params_cpp.input = inp_tv
        params_cpp.weight = weight_tv
        params_cpp.output = output_tv
        params_cpp.padding = padding
        params_cpp.stride = stride
        params_cpp.dilation = dilation

        params_cpp.beta = 0.0

        # print("CUDA PREPARED")
        for i in range(10):
            lib_object.implicit_gemm2(params_cpp)  # type: tv.Tensor
            print(time.time() - t)
            if i != 9:
                t = time.time()

        op_duration = time.time() - t
        if params.op_type == ConvOpType.kForward:
            output_cpu = output_tv.cpu().numpy()
            if params.dtype_a.itemsize() == 1:
                output_cpu = output_cpu.astype(np.float32)
            duration = time.time() - t
            print(output_cpu.reshape(-1)[:10], out_ref_nhwc.reshape(-1)[:10])
            print(params.get_algo_name(),
                  np.linalg.norm(out_ref_nhwc - output_cpu), "Time=",
                  op_duration)
        elif params.op_type == ConvOpType.kBackwardInput:
            print(ker.input_spec.tmap_b.iterations)
            din_cpu = inp_tv.cpu().numpy()
            duration = time.time() - t
            print(din_cpu.reshape(-1)[:10], din_ref_nhwc.reshape(-1)[:10])
            print(params.get_algo_name(),
                  np.linalg.norm(din_cpu - din_ref_nhwc), "Time=", op_duration)
        else:
            dw_cpu = weight_tv.cpu().numpy()
            duration = time.time() - t
            print(dw_cpu.reshape(-1)[:10], dw_ref_nhwc.reshape(-1)[:10])
            print(params.get_algo_name(), np.linalg.norm(dw_cpu - dw_ref_nhwc),
                  "Time=", op_duration)


if __name__ == "__main__":
    _asdv_test_simt_python_v2()
