#pragma once
#include <tensorview/core/all.h>
namespace tv {

namespace gemm {

namespace GemmAlgo {
enum _GemmAlgo {
    kSimt = 0,
    kSimtDP4A = 1,
    kSimtDP2A = 2,
    kVolta = 3,
    kTuringTensorOp = 4,
    kEnd = 999999
};

using all_gemm_algo_t = tv::mp_list_int<kSimt, kSimtDP4A, kSimtDP2A, kVolta>;
constexpr auto all_gemm_algo_v = tv::mp_list_c_to_array<all_gemm_algo_t>;

}


} // namespace gemm
} // namespace tv