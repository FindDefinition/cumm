from codeai.astex import lineprof

from cumm import tensorview as tv
from cumm.constants import TENSORVIEW_INCLUDE_PATH


# @lineprof.lineprof_wrapper_cpp
def test_nvrtc():

    # init driver
    a = tv.zeros([1], tv.float32, 0)
    n = tv.zeros([1], tv.uint64, -1)
    n.numpy_view()[0] = a.dim(0)
    prog = tv.NVRTCProgram(
        """
    #include <tensorview/core/all.h>
    #include <tensorview/gemm/debug.h>

    extern \"C\" __global__
    void add(float *x, int64_t n)
    {
        tv::printf2_once(sizeof(long));
        tv::printf2_once(sizeof(int));
        tv::printf2_once(sizeof(long long));

        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            x[tid] += 1;
        }
    }
    """,
        opts=["--std=c++17", "-I",
              str(TENSORVIEW_INCLUDE_PATH)])
    print(prog.ptx())

    mod = tv.NVRTCModule(prog)
    mod.load()
    mod.prepare_launch([1, 1, 1], [128, 1, 1], 0, 0)
    mod.run_kernel("add", a, a.dim(0))

    print(a.cpu().numpy())


if __name__ == "__main__":
    test_nvrtc()
