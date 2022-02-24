from codeai.astex import lineprof

from cumm import tensorview as tv
from cumm.constants import TENSORVIEW_INCLUDE_PATH


# @lineprof.lineprof_wrapper_cpp
def test_nvrtc():

    # init driver
    a = tv.zeros([1], tv.float32, 0)
    prog = tv.NVRTCProgram(
        """
    #include <tensorview/core/all.h>
    #include <tensorview/gemm/debug.h>
    #include <tensorview/core/arrayops/all.h>
    #include <cuda/std/cfloat>

    extern \"C\" __global__
    void add(float *x, int64_t n)
    {
        namespace op = tv::arrayops;
        tv::array<int, 2> lfs{};
        tv::array<int, 4> rfs{1, 1, 2, 3};
        // constexpr tv::array<tv::array<float, 3>, 3> a{tv::array<float, 3>{1, 2, 3}, tv::array<float, 3>{4, 5, 6}, tv::array<float, 3>{7, 8, 9}};
        // constexpr auto inv_a = a.op<row>(0);
constexpr tv::array<tv::array<float, 3>, 3> a{tv::array<float, 3>{1, 2, 3}, tv::array<float, 3>{4, 5, 6}, tv::array<float, 3>{7, 8, 3}};
 constexpr tv::array<tv::array<float, 3>, 3> b{tv::array<float, 3>{9, 7, 8}, tv::array<float, 3>{6, 5, 4}, tv::array<float, 3>{3, 2, 1}};
constexpr auto c = a.op<op::mv_colmajor>(b[0]);
constexpr auto c2 = a.op<op::inverse>();

        tv::printf2_once(sizeof(long));
        tv::printf2_once(sizeof(int));
        tv::printf2_once(sizeof(long long));
        auto v3 = tv::arrayops::apply(tv::detail::array_sum<int>, lfs, 2);
        tv::printf2_once(v3[0]);
        auto aa = a.op<op::col>(0);
        tv::printf2_once(aa[0], aa[1], aa[2]);
        aa -= 3;
        auto aa2 = -aa;
        tv::printf2_once(aa2[0], aa2[1], aa2[2]);
        constexpr tv::array<float, 3> a1{1, 2, 3};
        constexpr auto b1 = a1.op<tv::arrayops::max>(2);

        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            x[tid] += 1;
        }
        float fval = -4.6;
        tv::printf2_once(int(fval), static_cast<int>(fval), floorf(fval));
    }
    """,
        opts=["--std=c++17", "-I",
              str(TENSORVIEW_INCLUDE_PATH), 
              "-I", "/usr/local/cuda/include"])
    print(prog.ptx())

    mod = tv.NVRTCModule(prog)
    mod.load()
    launch = mod.get_launch_param([1, 1, 1], [128, 1, 1], 0, 0)
    mod.run_kernel("add", launch, a, a.dim(0))

    print(a.cpu().numpy())


if __name__ == "__main__":
    test_nvrtc()
