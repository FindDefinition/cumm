/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once
#include "inst_mma.h"
#include <tensorview/core/array.h>
#include <tensorview/core/defs.h>
#include <tensorview/gemm/core/all.h>
#include <tensorview/gemm/dtypes/all.h>

namespace tv {

namespace gemm {
namespace thread {

template <typename Shape, typename TA, typename TB, typename TC, typename LA,
          typename LB, typename LC, typename Enable = void>
struct Mma {
  static constexpr tv::array<int, 3> kShape = tv::mp_list_c_to_array<Shape>;
  using ElementA = TA;
  using ElementB = TB;
  using ElementC = TC;
  using FragmentA = tv::array<ElementA, kShape[0] * kShape[2]>;
  using FragmentB = tv::array<ElementB, kShape[2] * kShape[1]>;
  using FragmentC = tv::array<ElementC, kShape[0] * kShape[1]>;
  using InstMma = tv::gemm::inst::Mma<tv::mp_list_int<1, 1, 1>, 1, ElementA,
                                      ElementB, ElementC, LA, LB, LC>;

  TV_HOST_DEVICE_INLINE
  void operator()(FragmentC &D, FragmentA const &A, FragmentB const &B,
                  FragmentC const &C) {
    constexpr LA layoutA = LA::from_shape({kShape[0], kShape[2]});
    constexpr LB layoutB = LB::from_shape({kShape[2], kShape[1]});
    constexpr LC layoutC = LC::from_shape({kShape[0], kShape[1]});
    InstMma imma;
    D = C;
    TV_PRAGMA_UNROLL
    for (int k = 0; k < kShape[2]; ++k) {
      TV_PRAGMA_UNROLL
      for (int n = 0; n < kShape[1]; ++n) {
        TV_PRAGMA_UNROLL
        for (int m = 0; m < kShape[0]; ++m) {
          // what's this????
          // Column-major serpentine sequence to maximize reuse of A operand.
          // "mma_tensor_op_sm70.h:243"
          int m_serpentine = (n % 2) ? (kShape[0] - 1 - m) : m;
          typename InstMma::FragmentC d;
          typename InstMma::FragmentA a;
          typename InstMma::FragmentB b;
          d[0] = D[layoutC(m_serpentine, n)];
          a[0] = A[layoutA(m_serpentine, k)];
          b[0] = B[layoutB(k, n)];
          imma(d, a, b, d);
          D[layoutC(m_serpentine, n)] = d[0];
        }
      }
    }
  }
};

namespace detail {
template <typename Shape>
struct MmaDP4AEnable {
  static constexpr tv::array<int, 3> kShape = tv::mp_list_c_to_array<Shape>;
  static constexpr bool value = kShape[2] % 4 == 0;

};
}


template <typename Shape, typename LC>
struct Mma<Shape, int8_t, int8_t, int32_t, layout::RowMajor,
           layout::ColumnMajor, LC, typename std::enable_if_t<detail::MmaDP4AEnable<Shape>::value>> {
  static constexpr tv::array<int, 3> kShape = tv::mp_list_c_to_array<Shape>;
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::ColumnMajor;
  using LayoutC = LC;

  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = int32_t;
  using FragmentA = tv::array<ElementA, kShape[0] * kShape[2]>;
  using FragmentB = tv::array<ElementB, kShape[2] * kShape[1]>;
  using FragmentC = tv::array<ElementC, kShape[0] * kShape[1]>;
  using InstShape = tv::mp_list_int<1, 1, 4>;
  static constexpr tv::array<int, 3> kInstShape =
      tv::mp_list_c_to_array<InstShape>;
  using InstMma = tv::gemm::inst::Mma<InstShape, 1, ElementA, ElementB,
                                      ElementC, LayoutA, LayoutB, LayoutC>;

  TV_HOST_DEVICE_INLINE
  void operator()(FragmentC &D, FragmentA const &A, FragmentB const &B,
                  FragmentC const &C) {
    constexpr LC layoutC = LC::from_shape({kShape[0], kShape[1]});
    InstMma imma;
    D = C;
    tv::array<int8_t, 4> const *ptr_A =
        reinterpret_cast<tv::array<int8_t, 4> const *>(&A);
    tv::array<int8_t, 4> const *ptr_B =
        reinterpret_cast<tv::array<int8_t, 4> const *>(&B);

    TV_PRAGMA_UNROLL
    for (int k = 0; k < kShape[2] / kInstShape[2]; ++k) {

      TV_PRAGMA_UNROLL
      for (int n = 0; n < kShape[1]; ++n) {

        TV_PRAGMA_UNROLL
        for (int m = 0; m < kShape[0]; ++m) {

          tv::array<int32_t, 1> tmp =
              reinterpret_cast<tv::array<int32_t, 1> &>(D[layoutC(m, n)]);

          imma(tmp, ptr_A[m * kShape[2] / kInstShape[2] + k],
               ptr_B[n * kShape[2] / kInstShape[2] + k], tmp);

          D[layoutC(m, n)] = reinterpret_cast<int32_t &>(tmp);
        }
      }
    }
  }
};

template <typename Shape, typename LC>
struct Mma<Shape, int8_t, int8_t, int32_t, layout::ColumnMajor,
           layout::RowMajor, LC, typename std::enable_if_t<detail::MmaDP4AEnable<Shape>::value>> {
  static constexpr tv::array<int, 3> kShape = tv::mp_list_c_to_array<Shape>;
  using LayoutA = layout::ColumnMajor;
  using LayoutB = layout::RowMajor;
  using LayoutC = LC;

  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = int32_t;
  using FragmentA = tv::array<ElementA, kShape[0] * kShape[2]>;
  using FragmentB = tv::array<ElementB, kShape[2] * kShape[1]>;
  using FragmentC = tv::array<ElementC, kShape[0] * kShape[1]>;
  using InstShape = tv::mp_list_int<1, 1, 4>;
  static constexpr tv::array<int, 3> kInstShape =
      tv::mp_list_c_to_array<InstShape>;
  using InstMma = tv::gemm::inst::Mma<InstShape, 1, ElementA, ElementB,
                                      ElementC, LayoutA, LayoutB, LayoutC>;

  TV_HOST_DEVICE_INLINE
  void operator()(FragmentC &D, FragmentA const &A, FragmentB const &B,
                  FragmentC const &C) {
    constexpr LC layoutC = LC::from_shape({kShape[0], kShape[1]});
    InstMma imma;
    D = C;
    tv::array<int8_t, 4> const *ptr_A =
        reinterpret_cast<tv::array<int8_t, 4> const *>(&A);
    tv::array<int8_t, 4> const *ptr_B =
        reinterpret_cast<tv::array<int8_t, 4> const *>(&B);
    TV_PRAGMA_UNROLL
    for (int k = 0; k < kShape[2] / kInstShape[2]; ++k) {

      TV_PRAGMA_UNROLL
      for (int n = 0; n < kShape[1]; ++n) {

        TV_PRAGMA_UNROLL
        for (int m = 0; m < kShape[0]; ++m) {
          tv::array<int32_t, 1> tmp =
              reinterpret_cast<tv::array<int32_t, 1> &>(D[layoutC(m, n)]);
          imma(tmp, ptr_A[m + k * kShape[0]], ptr_B[n + k * kShape[0]], tmp);
          D[layoutC(m, n)] = reinterpret_cast<int32_t &>(tmp);
        }
      }
    }
  }
};


namespace detail {
template <typename Shape, typename LA, typename LB, typename LC>
struct MmaHfma2Enable {
  static constexpr tv::array<int, 3> kShape = tv::mp_list_c_to_array<Shape>;

  static constexpr bool kRowMajorA = std::is_same<LA, layout::RowMajor>::value;
  static constexpr bool kRowMajorB = std::is_same<LB, layout::RowMajor>::value;
  static constexpr bool kRowMajorC = std::is_same<LC, layout::RowMajor>::value;
  static constexpr bool value = (kRowMajorC && kShape[1] % 2 == 0) || (!kRowMajorC && kShape[0] % 2 == 0);

};
}

// TODO inner product hfma
template <typename Shape, typename LA, typename LB, typename LC>
struct Mma<Shape, half_t, half_t, half_t, LA, LB, LC,
           typename std::enable_if<detail::MmaHfma2Enable<Shape, LA, LB, LC>::value>::type> {
  static constexpr tv::array<int, 3> kShape = tv::mp_list_c_to_array<Shape>;
  using ElementA = half_t;
  using ElementB = half_t;
  using ElementC = half_t;
  using FragmentA = tv::array<ElementA, kShape[2] * kShape[0]>;
  using FragmentB = tv::array<ElementB, kShape[2] * kShape[1]>;
  using FragmentC = tv::array<ElementC, kShape[0] * kShape[1]>;

  static constexpr bool kRowMajorA = std::is_same<LA, layout::RowMajor>::value;
  static constexpr bool kRowMajorB = std::is_same<LB, layout::RowMajor>::value;
  static constexpr bool kRowMajorC = std::is_same<LC, layout::RowMajor>::value;
  using InstMmaShape = std::conditional_t<!kRowMajorC, tv::mp_list_int<2, 1, 1>,
                                          tv::mp_list_int<1, 2, 1>>;
  static constexpr auto kInstMmaShape = tv::mp_list_c_to_array<InstMmaShape>;
  static constexpr int kCCount = kInstMmaShape[0] * kInstMmaShape[1];

  TV_DEVICE_INLINE
  void operator()(FragmentC &D, FragmentA const &A, FragmentB const &B,
                  FragmentC const &C) {
    using InstMma = tv::gemm::inst::Mma<InstMmaShape, 1, ElementA, ElementB,
                                        ElementC, LA, LB, LC>;
    array<half_t, kCCount> *ptr_D =
        reinterpret_cast<array<half_t, kCCount> *>(&D);
    array<half_t, kInstMmaShape[0]> const *ptr_A =
        reinterpret_cast<array<half_t, kInstMmaShape[0]> const *>(&A);
    array<half_t, kInstMmaShape[1]> const *ptr_B =
        reinterpret_cast<array<half_t, kInstMmaShape[1]> const *>(&B);

    InstMma mma;
    if_constexpr_cuda<!kRowMajorC>([&](auto _){
      // 2, 1, 1 mma.
      TV_PRAGMA_UNROLL
      for (int k = 0; k < kShape[2] / InstMma::kShape[2]; k++) {
        TV_PRAGMA_UNROLL
        for (int m = 0; m < kShape[0] / InstMma::kShape[0]; m++) {
          TV_PRAGMA_UNROLL
          for (int n = 0; n < kShape[1] / InstMma::kShape[1]; n++) {
            array<half_t, 2> tmp;
            array<half_t, 2> *ptr_tmp = &tmp;
            if_constexpr_cuda<kRowMajorA>([&](auto _){
              ptr_tmp[0] = ptr_D[n * kShape[0] / 2 + m];
              array<half_t, 2> tmp_A;
              // row major A, read 2 elements in 2 row
              tmp_A[0] = (*ptr_A)[2 * m * kShape[2] + k];
              tmp_A[1] = (*ptr_A)[(2 * m + 1) * kShape[2] + k];
              if_constexpr_cuda<kRowMajorB>([&](auto _){
                mma(_(tmp), _(tmp_A), _(ptr_B)[k * kShape[1] + n], _(tmp));
              }, [&](auto _){
                mma(_(tmp), _(tmp_A), _(ptr_B)[n * kShape[2] + k], _(tmp));
              });
            }, [&](auto _){
              // col major A, just read two contiguous element.
              ptr_tmp[0] = ptr_D[n * kShape[0] / 2 + m];
              if_constexpr_cuda<kRowMajorB>([&](auto _){
                mma(tmp, ptr_A[k * kShape[0] / 2 + m], ptr_B[k * kShape[1] + n],
                    tmp);
              }, [&](auto _){
                mma(tmp, ptr_A[k * kShape[0] / 2 + m], ptr_B[n * kShape[2] + k],
                    tmp);
              });
            });
            ptr_D[m + n * kShape[0] / 2] = ptr_tmp[0];
          }
        }
      }
    }, [&](auto _){
      // 1, 2, 1 mma.
      TV_PRAGMA_UNROLL
      for (int k = 0; k < kShape[2] / InstMma::kShape[2]; k++) {
        TV_PRAGMA_UNROLL
        for (int n = 0; n < kShape[1] / InstMma::kShape[1]; n++) {
          TV_PRAGMA_UNROLL
          for (int m = 0; m < kShape[0] / InstMma::kShape[0]; m++) {
            array<half_t, 2> tmp;
            array<half_t, 2> *ptr_tmp = &tmp;
            if_constexpr_cuda<kRowMajorB>([&](auto _){
              // row major B, just read two contiguous element.
              ptr_tmp[0] = ptr_D[m * kShape[1] / 2 + n];
              if_constexpr_cuda<kRowMajorA>([&](auto _){
                mma(tmp, ptr_A[m * kShape[2] + k], ptr_B[k * kShape[1] / 2 + n],
                    tmp);
              }, [&](auto _){
                mma(tmp, ptr_A[k * kShape[0] + m], ptr_B[k * kShape[1] / 2 + n],
                    tmp);
              });
            }, [&](auto _){
              ptr_tmp[0] = ptr_D[m * kShape[1] / 2 + n];
              // col major B, read 2 elements in 2 row
              array<half_t, 2> tmp_B;
              tmp_B[0] = (*ptr_B)[2 * n * kShape[2] + k];
              tmp_B[1] = (*ptr_B)[(2 * n + 1) * kShape[2] + k];
              if_constexpr_cuda<kRowMajorA>([&](auto _){
                mma(tmp, ptr_A[m * kShape[2] + k], tmp_B, tmp);
              }, [&](auto _){
                mma(tmp, ptr_A[k * kShape[0] + m], tmp_B, tmp);
              });
            });

            ptr_D[m * kShape[1] / 2 + n] = ptr_tmp[0];
          }
        }
      }
    });

    // if TV_IF_CONSTEXPR (!kRowMajorC) {
    // } else {
    // }
  }
};

// template <typename Shape, typename LA, typename LB, typename LC>
// struct Mma<Shape, half_t, half_t, half_t, LA, LB, LC,
//            typename std::enable_if<detail::MmaHfma2Enable<Shape, LA, LB, LC>::value>::type> {
//   static constexpr tv::array<int, 3> kShape = tv::mp_list_c_to_array<Shape>;
//   using ElementA = half_t;
//   using ElementB = half_t;
//   using ElementC = half_t;
//   using FragmentA = tv::array<ElementA, kShape[2] * kShape[0]>;
//   using FragmentB = tv::array<ElementB, kShape[2] * kShape[1]>;
//   using FragmentC = tv::array<ElementC, kShape[0] * kShape[1]>;

//   static constexpr bool kRowMajorA = std::is_same<LA, layout::RowMajor>::value;
//   static constexpr bool kRowMajorB = std::is_same<LB, layout::RowMajor>::value;
//   static constexpr bool kRowMajorC = std::is_same<LC, layout::RowMajor>::value;
//   using InstMmaShape = std::conditional_t<!kRowMajorC, tv::mp_list_int<2, 1, 1>,
//                                           tv::mp_list_int<1, 2, 1>>;
//   static constexpr auto kInstMmaShape = tv::mp_list_c_to_array<InstMmaShape>;
//   static constexpr int kCCount = kInstMmaShape[0] * kInstMmaShape[1];

//   TV_DEVICE_INLINE
//   void operator()(FragmentC &D, FragmentA const &A, FragmentB const &B,
//                   FragmentC const &C) {
//     using InstMma = tv::gemm::inst::Mma<InstMmaShape, 1, ElementA, ElementB,
//                                         ElementC, LA, LB, LC>;
//     array<half_t, kCCount> *ptr_D =
//         reinterpret_cast<array<half_t, kCCount> *>(&D);
//     array<half_t, kInstMmaShape[0]> const *ptr_A =
//         reinterpret_cast<array<half_t, kInstMmaShape[0]> const *>(&A);
//     array<half_t, kInstMmaShape[1]> const *ptr_B =
//         reinterpret_cast<array<half_t, kInstMmaShape[1]> const *>(&B);

//     InstMma mma;

//     if TV_IF_CONSTEXPR (!kRowMajorC) {
//       // 2, 1, 1 mma.
//       TV_PRAGMA_UNROLL
//       for (int k = 0; k < kShape[2] / InstMma::kShape[2]; k++) {
//         TV_PRAGMA_UNROLL
//         for (int m = 0; m < kShape[0] / InstMma::kShape[0]; m++) {
//           TV_PRAGMA_UNROLL
//           for (int n = 0; n < kShape[1] / InstMma::kShape[1]; n++) {
//             array<half_t, 2> tmp;
//             array<half_t, 2> *ptr_tmp = &tmp;
//             // if_constexpr<kRowMajorA>([&](auto _){})
//             if TV_IF_CONSTEXPR (kRowMajorA) {
//               ptr_tmp[0] = ptr_D[n * kShape[0] / 2 + m];
//               array<half_t, 2> tmp_A;
//               // row major A, read 2 elements in 2 row
//               tmp_A[0] = (*ptr_A)[2 * m * kShape[2] + k];
//               tmp_A[1] = (*ptr_A)[(2 * m + 1) * kShape[2] + k];
//               if TV_IF_CONSTEXPR (kRowMajorB) {
//                 mma(tmp, tmp_A, ptr_B[k * kShape[1] + n], tmp);
//               } else {
//                 mma(tmp, tmp_A, ptr_B[n * kShape[2] + k], tmp);
//               }
//             } else {
//               // col major A, just read two contiguous element.
//               ptr_tmp[0] = ptr_D[n * kShape[0] / 2 + m];
//               if TV_IF_CONSTEXPR (kRowMajorB) {
//                 mma(tmp, ptr_A[k * kShape[0] / 2 + m], ptr_B[k * kShape[1] + n],
//                     tmp);
//               } else {
//                 mma(tmp, ptr_A[k * kShape[0] / 2 + m], ptr_B[n * kShape[2] + k],
//                     tmp);
//               }
//             }
//             ptr_D[m + n * kShape[0] / 2] = ptr_tmp[0];
//           }
//         }
//       }
//     } else {
//       // 1, 2, 1 mma.
//       TV_PRAGMA_UNROLL
//       for (int k = 0; k < kShape[2] / InstMma::kShape[2]; k++) {
//         TV_PRAGMA_UNROLL
//         for (int n = 0; n < kShape[1] / InstMma::kShape[1]; n++) {
//           TV_PRAGMA_UNROLL
//           for (int m = 0; m < kShape[0] / InstMma::kShape[0]; m++) {
//             array<half_t, 2> tmp;
//             array<half_t, 2> *ptr_tmp = &tmp;
//             if TV_IF_CONSTEXPR (kRowMajorB) {
//               // row major B, just read two contiguous element.
//               ptr_tmp[0] = ptr_D[m * kShape[1] / 2 + n];
//               if TV_IF_CONSTEXPR (kRowMajorA) {
//                 mma(tmp, ptr_A[m * kShape[2] + k], ptr_B[k * kShape[1] / 2 + n],
//                     tmp);
//               } else {
//                 mma(tmp, ptr_A[k * kShape[0] + m], ptr_B[k * kShape[1] / 2 + n],
//                     tmp);
//               }
//             } else {
//               ptr_tmp[0] = ptr_D[m * kShape[1] / 2 + n];
//               // col major B, read 2 elements in 2 row
//               array<half_t, 2> tmp_B;
//               tmp_B[0] = (*ptr_B)[2 * n * kShape[2] + k];
//               tmp_B[1] = (*ptr_B)[(2 * n + 1) * kShape[2] + k];
//               if TV_IF_CONSTEXPR (kRowMajorA) {
//                 mma(tmp, ptr_A[m * kShape[2] + k], tmp_B, tmp);
//               } else {
//                 mma(tmp, ptr_A[k * kShape[0] + m], tmp_B, tmp);
//               }
//             }
//             ptr_D[m * kShape[1] / 2 + n] = ptr_tmp[0];
//           }
//         }
//       }
//     }
//   }
// };


} // namespace thread

} // namespace gemm
} // namespace tv