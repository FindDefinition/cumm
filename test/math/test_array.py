from cumm.inliner import NVRTCInlineBuilder, torch_tensor_to_tv
from cumm.common import TensorView, TensorViewCPU, TensorViewNVRTC, TensorViewNVRTCHashKernel, TensorViewArrayLinalg, TensorViewNVRTCDev, EigenLib
import numpy as np
from cumm import dtypes
import pccm 
import numba 
from cumm.utils.array_grad_check import check_array_op_grad, check_array_binary_op_grad

from cumm import tensorview as tv

def test_normalize_and_grad():
    quat = np.random.uniform(-1, 1, size=[5, 3]).astype(np.float32)
    check_array_op_grad(quat, [3], "normalize", "normalize_grad", delta=1e-4)
    check_array_binary_op_grad([quat], [3], "normalize",[ "normalize_grad"], delta=1e-4)

def test_variance_transform_and_grad():
    A = np.random.uniform(-1, 1, size=[5, 3, 4]).astype(np.float32)
    B = np.random.uniform(-1, 1, size=[5, 2, 3]).astype(np.float32)
    check_array_binary_op_grad([A, B], [2, 4], "mm_nnn", ["mm_nnn_grad_lfs", "mm_nnn_grad_rfs"], delta=1e-4)
    # A = np.random.uniform(-1, 1, size=[5, 4, 3]).astype(np.float32)
    # B = np.random.uniform(-1, 1, size=[5, 3, 2]).astype(np.float32)
    # check_array_binary_op_grad([A, B], [4, 2], "mm_ttt", ["mm_ttt_grad_lfs", "mm_ttt_grad_rfs"], delta=1e-4)

    A = np.random.uniform(-1, 1, size=[5, 4, 4]).astype(np.float32)
    B = np.random.uniform(-1, 1, size=[5, 4, 4]).astype(np.float32)

    # check_array_binary_op_grad([A, B], [4, 4], "variance_transform_ttt", ["variance_transform_ttt_grad_lfs", "variance_transform_ttt_grad_rfs"], delta=1e-4)
    check_array_binary_op_grad([A, B], [4, 4], "variance_transform_nnn", ["variance_transform_nnn_grad_lfs", "variance_transform_nnn_grad_rfs"], delta=1e-4)
    # check_array_binary_op_grad([A], [4, 4], "identity_variance_transform_ttt", ["identity_variance_transform_ttt_grad"], delta=1e-4)
    # check_array_binary_op_grad([A], [4, 4], "symmetric_variance_transform_nnn", ["identity_variance_transform_ttt_grad"], delta=1e-4)

def test_broadcast_op():
    np.random.seed(50052)
    arr1_np = np.random.uniform(-1, 1, size=(1, 4,)).astype(np.float32)
    arr2_np = np.random.uniform(-1, 1, size=(4, 4,)).astype(np.float32)
    arr3_np = np.random.uniform(-1, 1, size=(4, 1,)).astype(np.float32)
    ivec0 = np.random.randint(-15, 100, size=(4,)).astype(np.int32)
    ivecupper = np.random.randint(0, 15, size=(4,)).astype(np.int32)

    # we still need to use .cuda to make code compatabile with cuda
    arr_res_add = tv.zeros([4, 4], tv.float32, 0)
    arr_res_maximum = tv.zeros([4, 4], tv.float32, 0)
    arr_res_clamp = tv.zeros([4, 4], tv.float32, 0)
    arr_res_clamp2 = tv.zeros([4, 4], tv.float32, 0)
    vec_res_i0 = tv.zeros([4,], tv.int32, 0)

    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg, TensorViewNVRTCHashKernel], std="c++17")
    cnt = tv.zeros([1], tv.int32, 0)
    inliner.kernel_1d("test_broadcast_op", 1, 0, f"""
    namespace op = tv::arrayops;
    auto arr1_val = $arr1_np;
    auto arr2_val = $arr2_np;
    auto arr3_val = $arr3_np;
    auto arr_res_add_val = arr1_val + arr3_val;
    auto ivec0_val = $ivec0;
    auto ivecupper_val = $ivecupper;
    op::reinterpret_cast_array_nd<4, 4>($arr_res_add)[0] = arr_res_add_val;
    op::reinterpret_cast_array_nd<4, 4>($arr_res_maximum)[0] = arr1_val.op<op::maximum>(arr2_val);
    op::reinterpret_cast_array_nd<4, 4>($arr_res_clamp)[0] = arr2_val.op<op::clamp>(arr1_val, arr3_val);
    op::reinterpret_cast_array_nd<4, 4>($arr_res_clamp2)[0] = arr2_val.op<op::clamp>(0.0f, arr3_val);
    op::reinterpret_cast_array_nd<4>($vec_res_i0)[0] = ivec0_val.op<op::clamp>(0, ivecupper_val);

    """)
    assert np.allclose(arr_res_add.cpu().numpy(), arr1_np + arr3_np)
    assert np.allclose(arr_res_maximum.cpu().numpy(), np.maximum(arr1_np, arr2_np))
    assert np.allclose(arr_res_clamp.cpu().numpy(), np.clip(arr2_np, arr1_np, arr3_np))
    assert np.allclose(arr_res_clamp2.cpu().numpy(), np.clip(arr2_np, 0, arr3_np))
    assert np.allclose(vec_res_i0.cpu().numpy(), np.clip(ivec0, 0, ivecupper))

def test_matrix_ops():
    np.random.seed(50052)
    A_np = np.random.uniform(-1, 1, size=(4, 4,)).astype(np.float32)
    B_np = np.random.uniform(-1, 1, size=(4, 4,)).astype(np.float32)

    # we still need to use .cuda to make code compatabile with cuda
    arr_res_mm_nnn = tv.zeros([4, 4], tv.float32, 0)
    arr_res_mm_ttt = tv.zeros([4, 4], tv.float32, 0)
    arr_res_mm_tnn = tv.zeros([4, 4], tv.float32, 0)
    arr_res_mm_ntn = tv.zeros([4, 4], tv.float32, 0)
    arr_res_var_tr_ttt = tv.zeros([4, 4], tv.float32, 0)
    arr_res_var_tr_nnn = tv.zeros([4, 4], tv.float32, 0)
    arr_res_ivar_tr_nnn = tv.zeros([4, 4], tv.float32, 0)

    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg, TensorViewNVRTCHashKernel], std="c++17")
    cnt = tv.zeros([1], tv.int32, 0)
    inliner.kernel_1d("test_broadcast_op", 1, 0, f"""
    namespace op = tv::arrayops;
    auto A = $A_np;
    auto B = $B_np;
    op::reinterpret_cast_array_nd<4, 4>($arr_res_mm_nnn)[0] = A.op<op::mm_nnn>(B);
    op::reinterpret_cast_array_nd<4, 4>($arr_res_mm_ttt)[0] = A.op<op::mm_ttt>(B);
    op::reinterpret_cast_array_nd<4, 4>($arr_res_mm_tnn)[0] = A.op<op::mm_tnn>(B);
    op::reinterpret_cast_array_nd<4, 4>($arr_res_mm_ntn)[0] = A.op<op::mm_ntn>(B);
    op::reinterpret_cast_array_nd<4, 4>($arr_res_var_tr_ttt)[0] = A.op<op::variance_transform_ttt>(B);
    op::reinterpret_cast_array_nd<4, 4>($arr_res_var_tr_nnn)[0] = A.op<op::variance_transform_nnn>(B);
    op::reinterpret_cast_array_nd<4, 4>($arr_res_ivar_tr_nnn)[0] = A.op<op::identity_variance_transform_nnn>();

    """)
    assert np.allclose(arr_res_mm_nnn.cpu().numpy(), (A_np.T @ B_np.T).T)
    assert np.allclose(arr_res_mm_ttt.cpu().numpy(), A_np @ B_np)
    assert np.allclose(arr_res_mm_tnn.cpu().numpy(), (A_np @ B_np.T).T)
    assert np.allclose(arr_res_mm_ntn.cpu().numpy(), (A_np.T @ B_np).T)
    assert np.allclose(arr_res_var_tr_ttt.cpu().numpy(), A_np @ B_np @ A_np.T)
    assert np.allclose(arr_res_var_tr_nnn.cpu().numpy(), (A_np.T @ B_np.T @ A_np).T)
    assert np.allclose(arr_res_ivar_tr_nnn.cpu().numpy(), (A_np.T @ A_np).T)

if __name__ == "__main__":
    test_variance_transform_and_grad()
    test_broadcast_op()
    test_normalize_and_grad()
    test_matrix_ops()