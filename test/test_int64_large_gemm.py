"""
End-to-end validation that cumm's NVRTC GEMM kernel produces correct
output when the *gather-index * stride* byte-offset into `a` exceeds
INT32_MAX.

Strategy:
  * Allocate a = (N_total, K) float32 on GPU, populate a[row, :] = row.
  * Build a shuffle-AC SIMT GEMM kernel via NVRTC.
  * Repeat the same GEMM with increasingly large max(a_inds), sweeping
    across the INT32_MAX / (K * 4) threshold to expose any int32 offset
    arithmetic inside the kernel.
  * Compute reference c_ref = a[a_inds] @ b via PyTorch on GPU and compare.

Pass criterion: all sweep points match reference within fp32 tolerance.
A failure localized to indices where `idx * K * 4 > 2^31` would indicate
that the NVRTC kernel uses int32 internally for pointer arithmetic.
"""
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import sys
import numpy as np
import torch

from cumm import tensorview as tv
from cumm.gemm import kernel as _kernel
from cumm.gemm.algospec.core import ShuffleStrideType
from cumm.gemm.constants import NVRTCConstants, NVRTCMode
from cumm.gemm.main import GemmAlgoParams, gen_gemm_kernels
from cumm.nvrtc import CummNVRTCModule


INT32_MAX = (1 << 31) - 1


# --------------------------------------------------------------------
# kernel setup
# --------------------------------------------------------------------
def build_shuffle_ac_f32_kernel():
    # Small SIMT f32 tile that's in SHUFFLE_SIMT_PARAMS.
    ts = (64, 64, 8)
    wts = (32, 32, 8)
    params = GemmAlgoParams(
        ts, wts, 2, "f32,f32,f32,f32,f32",
        trans_a=False, trans_b=False, trans_c=False,
        algo=_kernel.GemmAlgo.Simt, tensorop=None,
        splitk_serial=False, splitk_parallel=False,
        shuffle_stride=ShuffleStrideType.ShuffleAC,
    )
    nvrtc_mode = NVRTCMode.ConstantMemory
    ker = gen_gemm_kernels(params, nvrtc_mode=nvrtc_mode)
    ker.namespace = "wtf"
    custom_names = [f"&wtf::{NVRTCConstants.CONSTANT_PARAM_KEY}"]
    mod = CummNVRTCModule([ker], verbose=False, custom_names=custom_names)
    mod.load()
    return params, ker, mod, nvrtc_mode


def make_algo_desp(params):
    algo = tv.gemm.GemmAlgoDesp()
    algo.tile_shape = list(params.ts)
    algo.warp_tile_shape = list(params.wts)
    algo.num_stage = params.num_stage
    algo.dtype_a = tv.float32
    algo.dtype_b = tv.float32
    algo.dtype_c = tv.float32
    algo.dacc = params.dtype_acc.tv_dtype
    algo.dcomp = params.dtype_comp.tv_dtype
    algo.algo = params.algo.value
    algo.trans_a = params.trans_a
    algo.trans_b = params.trans_b
    algo.trans_c = params.trans_c
    algo.shuffle_type = tv.gemm.ShuffleStrideType.ShuffleAC
    return algo


def make_nvrtc_params(ker, mod, nvrtc_mode):
    nv = tv.gemm.NVRTCParams()
    nv.cumodule = mod.get_cpp_object()
    nv.mode = nvrtc_mode.value
    nv.num_threads = ker.num_threads
    nv.smem_size = ker.smem_size
    nv.kernel_name = mod.get_lowered_name(f"{ker.namespace}::gemm_kernel")
    nv.init_kernel_name = mod.get_lowered_name(
        f"{ker.namespace}::nvrtc_kernel_cpu_out")
    nv.param_size = mod.const_values[
        f"{ker.namespace}::{NVRTCConstants.SIZEOF_KEY}"]
    nv.constant_name = mod.get_lowered_name(
        f"&{ker.namespace}::{NVRTCConstants.CONSTANT_PARAM_KEY}")
    nv.param_storage = tv.empty([nv.param_size], tv.uint8, 0)
    return nv


# --------------------------------------------------------------------
# data
# --------------------------------------------------------------------
def fill_pattern_chunked(a_th: torch.Tensor, chunk_rows: int = 1 << 18):
    """a_th[row, j] = row (float32). Done in chunks to avoid huge tmp tensors."""
    N, K = a_th.shape
    for start in range(0, N, chunk_rows):
        end = min(N, start + chunk_rows)
        rows = torch.arange(start, end, device=a_th.device, dtype=torch.float32)
        a_th[start:end].copy_(rows.unsqueeze(1).expand(-1, K))


# --------------------------------------------------------------------
# one GEMM run
# --------------------------------------------------------------------
def run_one(a_th, b_th, a_inds_np, params, ker, mod, nvrtc_mode, algo_desp):
    """Run cumm GEMM for a single a-index tensor; return cumm_c, torch_ref."""
    m = int(a_inds_np.shape[0])
    K = int(a_th.shape[1])
    n = int(b_th.shape[1])

    # c_inds writes output to dense c rows [0..m-1]. a_inds gathers from a.
    c_inds_np = np.arange(m, dtype=np.int32)

    a_tv = tv.from_blob(a_th.data_ptr(), list(a_th.shape), tv.float32, 0)
    b_tv = tv.from_blob(b_th.data_ptr(), list(b_th.shape), tv.float32, 0)
    c_th = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    c_tv = tv.from_blob(c_th.data_ptr(), [m, n], tv.float32, 0)

    a_inds_th = torch.from_numpy(a_inds_np).cuda()
    c_inds_th = torch.from_numpy(c_inds_np).cuda()
    a_inds_tv = tv.from_blob(a_inds_th.data_ptr(), [m], tv.int32, 0)
    c_inds_tv = tv.from_blob(c_inds_th.data_ptr(), [m], tv.int32, 0)

    params_cpp = tv.gemm.GemmParams()
    params_cpp.algo_desp = algo_desp
    params_cpp.split_k_slices = 1
    params_cpp.a = a_tv
    params_cpp.b = b_tv
    params_cpp.c = c_tv
    params_cpp.a_inds = a_inds_tv
    params_cpp.c_inds = c_inds_tv
    params_cpp.alpha = 1.0
    params_cpp.beta = 0.0
    params_cpp.act_type = tv.gemm.Activation.None_
    params_cpp.nvrtc_params = make_nvrtc_params(ker, mod, nvrtc_mode)

    tv.gemm.run_nvrtc_gemm_kernel(params_cpp)
    torch.cuda.synchronize()

    ref = a_th.index_select(0, a_inds_th.to(torch.long)).matmul(b_th)
    return c_th, ref


def compare(c, ref, label):
    abs_err = (c - ref).abs()
    max_abs = float(abs_err.max())
    max_ref = float(ref.abs().max())
    rel = max_abs / max(max_ref, 1e-30)
    ok = max_abs == max_abs and rel < 1e-3  # the NaN-check guards overflow → nan
    print(f"  [{label}]  max_abs_err={max_abs:.4e}  max_ref={max_ref:.4e}  "
          f"rel_err={rel:.4e}  -> {'OK' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------
def main():
    torch.cuda.init()
    device_free, _ = torch.cuda.mem_get_info()
    print(f"GPU free memory: {device_free / 1024**3:.1f} GiB")

    K = 256
    bytes_per_row = K * 4
    threshold_row = INT32_MAX // bytes_per_row  # = 2_097_151 for K=256 f32
    print(f"K={K}, bytes/row={bytes_per_row}, "
          f"int32-overflow row threshold = {threshold_row:,} "
          f"(row * K * 4 > INT32_MAX for row > {threshold_row})")

    # Need N_total > threshold_row so we can probe both sides.
    # 2^24 rows * 256 cols * 4B = 16 GiB.
    N_total = 1 << 24
    print(f"Allocating a: ({N_total:,}, {K}) f32 = "
          f"{N_total * K * 4 / 1024**3:.2f} GiB")
    a_th = torch.empty(N_total, K, device="cuda", dtype=torch.float32)
    fill_pattern_chunked(a_th)

    # Small random b and small m.
    m = 128
    n = 64
    b_th = torch.randn(K, n, device="cuda", dtype=torch.float32)

    params, ker, mod, nvrtc_mode = build_shuffle_ac_f32_kernel()
    algo_desp = make_algo_desp(params)

    # sweep: picked row indices straddling the INT32 threshold.
    probes = [
        ("under int32 (small indices)", np.arange(m, dtype=np.int32)),
        ("just below threshold",
         np.linspace(threshold_row - m, threshold_row - 1,
                     m, dtype=np.int32)),
        ("just above threshold",
         np.linspace(threshold_row + 1, threshold_row + m,
                     m, dtype=np.int32)),
        ("far above threshold (2x)",
         np.linspace(2 * threshold_row, 2 * threshold_row + m - 1,
                     m, dtype=np.int32)),
        ("near end of a",
         np.linspace(N_total - m, N_total - 1,
                     m, dtype=np.int32)),
        ("mixed under+over",
         np.concatenate([
             np.arange(m // 2, dtype=np.int32),
             np.linspace(N_total - m // 2, N_total - 1,
                         m // 2, dtype=np.int32),
         ])),
    ]

    results = []
    for label, inds_np in probes:
        max_idx = int(inds_np.max())
        off_bytes = max_idx * K * 4
        over = off_bytes > INT32_MAX
        print(f"\n{label}: max_idx={max_idx:,} -> max offset "
              f"{off_bytes:,} bytes ({'over-int32' if over else 'under-int32'})")
        try:
            c, ref = run_one(a_th, b_th, inds_np,
                             params, ker, mod, nvrtc_mode, algo_desp)
            ok = compare(c, ref, label)
        except Exception as e:
            msg = str(e).splitlines()[0]
            print(f"  [{label}]  CUDA/host error: {msg!r}  -> FAIL")
            ok = False
            # device context is poisoned; re-init is not possible in CUDA
            # so we stop after the first crash.
            results.append((label, ok))
            break
        results.append((label, ok))

    print("\n================= SUMMARY =================")
    all_ok = True
    for label, ok in results:
        print(f"  {label}: {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    if all_ok:
        print("\nPASS: cumm GEMM shuffle-AC gather is correct for all probed "
              "index magnitudes; kernel handles offsets > INT32_MAX.")
        sys.exit(0)
    else:
        print("\nFAIL: at least one probe produced wrong output.")
        sys.exit(1)


if __name__ == "__main__":
    main()
