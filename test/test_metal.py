import pickle
import time

import torch

from cumm import tensorview as tv
from cumm.constants import PACKAGE_ROOT, TENSORVIEW_INCLUDE_PATH
from cumm.inliner import NVRTCInlineBuilder
from cumm.common import TensorView, TensorViewCPU, TensorViewNVRTCHashKernel, TensorViewArrayLinalg, TensorViewNVRTCDev, EigenLib
import numpy as np

def test_metal_basic():
    # init cuda
    a = tv.zeros([100000, 3], tv.float32, 0)
    cnt = tv.zeros([1], tv.int32, 0)
    data = tv.zeros([32], tv.uint64, 0)
    aa = np.eye(4).astype(np.float32)
    inliner = NVRTCInlineBuilder([], root=PACKAGE_ROOT.parent, std="c++17")
    for i in range(5):
        t = time.time()
        inliner.kernel_1d("nvrtc_std", a.shape[0], 0, 
                        f"""
        namespace op = tv::arrayops;
        auto a_ptr = reinterpret_cast<TV_METAL_DEVICE tv::array<float, 3>*>($a);
        auto a_val = a_ptr[i];
        a_ptr[i] = a_val.op<op::transform_3d>($aa);

        auto c_atomic_ptr = reinterpret_cast<device std::atomic_int*>($cnt);
        int old = tv::parallel::atomicAggInc($cnt);
        if (i <= 32){{
            auto wtf_max = metal::numeric_limits<tv::parallel::vote_t>::max();
            $data[i] = tv::parallel::detail::lanemask_lt();
            // $data[i] = tv::parallel::warp_size() - tv::parallel::lane_index();
        }}
        
        """)
        print(time.time() - t)
    print(a.cpu().numpy())
    print(cnt.cpu().numpy())
    print(data.cpu().numpy())

    # print(inliner.get_nvrtc_kernel_attrs("nvrtc_std"))

    # mod = CummNVRTCModule([TensorViewNVRTCDev()], verbose=True)

def test_metal_hash():
    keys_np = np.random.randint(0, 10, size=(2500)).astype(np.int32)
    # we still need to use .cuda to make code compatabile with cuda
    th_ten = torch.rand(1).to("mps")
    print(th_ten.is_mps)
    keys = tv.from_numpy(keys_np).cuda()
    wtf = tv.zeros([100], tv.float16, 0)
    keys_debug = tv.zeros([keys.shape[0]], tv.uint32, 0)
    hash_length = int(keys.shape[0] * 2)
    hashkeys = tv.empty([hash_length], tv.int32, 0)
    hashvalues = tv.empty([hash_length], tv.int32, 0)
    inliner = NVRTCInlineBuilder([TensorViewArrayLinalg, TensorViewNVRTCHashKernel], root=PACKAGE_ROOT.parent, std="c++17")
    cnt = tv.zeros([1], tv.int32, 0)
    inliner.kernel_1d("clear_hash", hashkeys.dim(0), 0, f"""
    using table_t = tv::hash::LinearHashTableSplit<int32_t, int>;
    $hashkeys[i] = table_t::empty_key;
    """)
    # print(keys.cpu().numpy())
    inliner.kernel_1d("insert_table", keys.dim(0), 0, f"""
    using table_t = tv::hash::LinearHashTableSplit<int, int>;
    table_t table($hashkeys, $hashvalues, $hash_length);
    auto key = $keys[i];
    table.insert(key, 1);
    // $keys_debug[i] = $hash_length;
    """)

    inliner.kernel_1d("collect_unique_res", hashkeys.dim(0), 0, f"""
    using table_t = tv::hash::LinearHashTableSplit<int, int>;
    table_t table($hashkeys, $hashvalues, $hash_length);
    auto wtf_ptr = $th_ten;
    if ($hashkeys[i] != table_t::empty_key){{
        tv::parallel::atomicAdd($cnt, 1);
    }}
    """)

    print(cnt.cpu().numpy())

if __name__ == "__main__":
    test_metal_hash()
