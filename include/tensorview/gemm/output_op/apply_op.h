#pragma once
#include <array>
#include <iterator>
#include <tensorview/core/all.h>
#include <tensorview/gemm/core/all.h>
#include <tensorview/gemm/math/functional.h>
#include <tensorview/gemm/math/numeric_convert.h>
#include <type_traits>

namespace tv {

namespace gemm {

template <int ElementPerAccess, class OutputOp, class OutFragment,
          class InputFragment>
TV_HOST_DEVICE_INLINE void apply_output_operator(OutFragment &output_fragment,
                           OutputOp const &output_op, ///< Output operator
                           InputFragment const &aligned_accum_fragment,
                           OutFragment const &source_fragment) {
  constexpr int kOutFragCount = tv::array_size_v<OutFragment>;
  constexpr int kInputFragCount = tv::array_size_v<InputFragment>;
  using OutAccessType =
      tv::array<typename OutFragment::value_type, ElementPerAccess>;
  using InputAccessType =
      tv::array<typename InputFragment::value_type, ElementPerAccess>;
  OutAccessType *output_frag_ptr =
      reinterpret_cast<OutAccessType *>(&output_fragment);
  InputAccessType const *compute_frag_ptr =
      reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
  OutAccessType const *source_frag_ptr =
      reinterpret_cast<OutAccessType const *>(&source_fragment);
  constexpr int kOutOpIterations = kOutFragCount / ElementPerAccess;
  TV_PRAGMA_UNROLL
  for (int i = 0; i < kOutOpIterations; ++i) {
    output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
  }
};

template <int ElementPerAccess, class OutputOp, class OutFragment,
          class InputFragment>
TV_HOST_DEVICE_INLINE void apply_output_operator_no_source(
    OutFragment &output_fragment,
    OutputOp const &output_op, ///< Output operator
    InputFragment const &aligned_accum_fragment) {
  constexpr int kOutFragCount = tv::array_size_v<OutFragment>;
  constexpr int kInputFragCount = tv::array_size_v<InputFragment>;

  using OutAccessType =
      tv::array<typename OutFragment::value_type, ElementPerAccess>;
  using InputAccessType =
      tv::array<typename InputFragment::value_type, ElementPerAccess>;
  OutAccessType *output_frag_ptr =
      reinterpret_cast<OutAccessType *>(&output_fragment);
  InputAccessType const *compute_frag_ptr =
      reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
  constexpr int kOutOpIterations = kOutFragCount / ElementPerAccess;
  TV_PRAGMA_UNROLL
  for (int i = 0; i < kOutOpIterations; ++i) {
    output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
  }
};

} // namespace gemm
} // namespace tv