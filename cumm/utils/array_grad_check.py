import numpy as np 
from cumm.gemm.layout_tensorop import rowmajor_inverse
from cumm.inliner import NVRTCInlineBuilder
from cumm.common import TensorView, TensorViewCPU, TensorViewNVRTC, TensorViewNVRTCHashKernel, TensorViewArrayLinalg, TensorViewNVRTCDev, EigenLib
from cumm import tensorview as tv

from ccimport import compat 
def check_array_op_grad(inp: np.ndarray, inp_shape: list[int], out_shape: list[int], op: str, grad_op: str, delta: float = 1e-4):
    num_element = np.prod(inp_shape)
    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg], std="c++17")
    dtype = np.float64
    tv_dtype = tv.float64
    if compat.IsAppleSiliconMacOs:
        dtype = np.float32 
        tv_dtype = tv.float32 
    inp = inp.astype(dtype)
    inp_tv = tv.from_numpy(inp).cuda()

    grad_scalar = np.random.uniform(0.5, 1.5, size=out_shape).astype(dtype)
    num_check = inp.shape[0]
    inp_shape_str = ", ".join(map(str, inp_shape))
    out_shape_str = ", ".join(map(str, out_shape))
    my_val_tv = tv.zeros([num_check], tv_dtype, 0)
    ref_val_tv = tv.zeros([num_check], tv_dtype, 0)

    assert inp_shape == list(inp.shape[1:]), f"{inp_shape}, {inp.shape[1:]}"
    inp_shape_np = np.array(inp_shape, np.int32)
    for i in range(num_element):
        inp_delta = np.zeros(inp_shape, dtype)
        index = i
        inp_delta.reshape(-1)[i] = delta 
        slice_indexes = rowmajor_inverse(i, inp_shape_np)
        slice_indexes_str = "".join(map(lambda x: f"[{x}]", slice_indexes))
        inliner.kernel_1d(f"check_grad_op_{op}_{grad_op}_{inp_shape}", num_check, 0, f"""
        namespace op = tv::arrayops;
        auto inp_ptr = op::reinterpret_cast_array_nd<float, {inp_shape_str}>($inp_tv);
        auto inp_arr = inp_ptr[i];
        auto grad_scale = $grad_scalar;
        auto inp_delta_val = $inp_delta;

        auto out_arr = inp_arr.op<op::{op}>() * grad_scale;
        auto out_arr_with_delta = (inp_arr + inp_delta_val).op<op::{op}>() * grad_scale;
        auto out_arr_with_delta_sum = op::reshape<-1>(out_arr_with_delta - out_arr).op<op::sum>(); 
        $my_val_tv[i] = op::reshape<-1>(grad_scale.op<op::{grad_op}>(inp_arr))[$index];
        $ref_val_tv[i] = out_arr_with_delta_sum / op::reshape<-1>(inp_delta_val)[$index];
        """)

        my_val = my_val_tv.cpu().numpy()
        ref_val = ref_val_tv.cpu().numpy()
        print(my_val)
        print(ref_val)