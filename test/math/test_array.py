from cumm.inliner import NVRTCInlineBuilder, torch_tensor_to_tv
from cumm.common import TensorView, TensorViewCPU, TensorViewNVRTC, TensorViewNVRTCHashKernel, TensorViewArrayLinalg, TensorViewNVRTCDev, EigenLib
import numpy as np
from cumm import dtypes
import pccm 
import numba 

from cumm import tensorview as tv


def test_broadcast_op():
    arr1_np = np.random.uniform(-1, 1, size=(1, 4,)).astype(np.float32)
    arr2_np = np.random.uniform(-1, 1, size=(4, 4,)).astype(np.float32)
    arr3_np = np.random.uniform(-1, 1, size=(4, 1,)).astype(np.float32)

    # we still need to use .cuda to make code compatabile with cuda
    arr_res_add = tv.zeros([4, 4], tv.float32, 0)
    arr_res_maximum = tv.zeros([4, 4], tv.float32, 0)
    arr_res_clamp = tv.zeros([4, 4], tv.float32, 0)

    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg, TensorViewNVRTCHashKernel], std="c++17")
    cnt = tv.zeros([1], tv.int32, 0)
    inliner.kernel_1d("test_broadcast_op", 1, 0, f"""
    namespace op = tv::arrayops;
    auto arr1_val = $arr1_np;
    auto arr2_val = $arr2_np;
    auto arr3_val = $arr3_np;
    auto arr_res_add_val = arr1_val + arr3_val;
    op::reinterpret_cast_array_nd<float, 4, 4>($arr_res_add)[0] = arr_res_add_val;
    op::reinterpret_cast_array_nd<float, 4, 4>($arr_res_maximum)[0] = arr1_val.op<op::maximum>(arr2_val);
    op::reinterpret_cast_array_nd<float, 4, 4>($arr_res_clamp)[0] = arr2_val.op<op::clamp>(arr1_val, arr3_val);

    """)
    np.allclose(arr_res_add.cpu().numpy(), arr1_np + arr2_np)
    np.allclose(arr_res_maximum.cpu().numpy(), np.maximum(arr1_np, arr2_np))
    np.allclose(arr_res_clamp.cpu().numpy(), np.clip(arr2_np, arr1_np, arr3_np))
    
if __name__ == "__main__":
    test_broadcast_op()