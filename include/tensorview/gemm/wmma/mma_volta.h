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
#include <tensorview/core/array.h>
#include <tensorview/core/defs.h>
#include <tensorview/gemm/core/all.h>

#include "inst_mma.h"

namespace tv {

namespace gemm {
namespace mma {

template <typename Shape, typename TA, typename TB, typename TC, typename LA,
          typename LB, typename LC>
struct MmaVolta {
  static constexpr tv::array<int, 3> kShape = tv::mp_list_c_to_array<Shape>;
  static constexpr tv::array<int, 3> kInterleavedWmmaShape{32, 32, 4};
  // a mma.sync handle 16x16x4 in a warp.
  static constexpr tv::array<int, 3> kInstShape{16, 16, 4};
  static constexpr tv::array<int, 2> kMmaIterations =
      arrayops::slice<0, 2>(kInterleavedWmmaShape / kInstShape);
  static constexpr tv::array<int, 2> kMmaTileIterations =
      arrayops::slice<0, 2>(kShape / kInterleavedWmmaShape);

  using ElementA = TA;
  using ElementB = TB;
  using ElementC = TC;
  using InstMma = tv::gemm::inst::Mma<tv::mp_list_int<8, 8, 4>, 8, ElementA,
                                      ElementB, ElementC, LA, LB, LC>;
  using FragmentA = tv::array<ElementA, kInstShape[2] * kMmaIterations[0] *
                                            kMmaTileIterations[0]>;
  using FragmentB = tv::array<ElementB, kInstShape[2] * kMmaIterations[1] *
                                            kMmaTileIterations[1]>;
  using FragmentC =
      tv::array<ElementC, arrayops::prod(kMmaIterations) *
                              arrayops::prod(kMmaTileIterations) * 8>;

  TV_HOST_DEVICE_INLINE
  void operator()(FragmentC &D, FragmentA const &A, FragmentB const &B,
                  FragmentC const &C) {
    InstMma mma;

    using MmaOperandA = typename InstMma::FragmentA; // 4
    using MmaOperandB = typename InstMma::FragmentB; // 4
    using MmaOperandC = typename InstMma::FragmentC; // 8

    D = C;
    MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
    MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);
    // tv::printf2_once("=================");
    TV_PRAGMA_UNROLL
    for (int outer_col = 0; outer_col < kMmaTileIterations[1]; ++outer_col) {
      TV_PRAGMA_UNROLL
      for (int inner_col = 0; inner_col < kMmaIterations[1]; ++inner_col) {
        TV_PRAGMA_UNROLL
        for (int outer_row = 0; outer_row < kMmaTileIterations[0];
             ++outer_row) {
          TV_PRAGMA_UNROLL
          for (int inner_row = 0; inner_row < kMmaIterations[0]; ++inner_row) {
            int op_col = inner_col + kMmaIterations[1] * outer_col;
            // Column-major serpentine sequence to maximize reuse of A operand.
            int inner_row_serp = inner_row;
            int outer_row_serp = outer_row;
            if (op_col & 1) {
              inner_row_serp = kMmaIterations[0] - inner_row - 1;
              outer_row_serp = kMmaTileIterations[0] - outer_row - 1;
            }
            int op_row = inner_row_serp + kMmaIterations[0] * outer_row_serp;
            // op_idx: [kMmaTileIterations[1], kMmaTileIterations[0],
            // kMmaIterations[1], kMmaIterations[0]]

            int op_idx =
                inner_row_serp +
                kMmaIterations[0] *
                    (inner_col +
                     kMmaIterations[1] *
                         (outer_row_serp + kMmaTileIterations[0] * outer_col));
            // auto p = ptr_A[op_row];
            // auto p2 = ptr_B[op_col];
            // auto p3 = ptr_D[op_idx];

            // constexpr int Astride = 32;
            // constexpr int Bstride = 128;


            mma(ptr_D[op_idx], ptr_A[op_row], ptr_B[op_col], ptr_D[op_idx]);
            // // tv::printf2_once(op_idx, float(p[0]), float(p[1]), float(p[2]), float(p[3]));
            // if (threadIdx.x >= 64 && threadIdx.x < 68 && blockIdx.x == 0 && blockIdx.y == 0){
            //   tv::printf2(threadIdx.x, outer_col, inner_col, outer_row, inner_row, 
            //     int(p[0]), int(p[1]), int(p[2]), int(p[3]), 
            //     int(p2[0]), int(p2[1]), int(p2[2]), int(p2[3]));
            //   // tv::printf2(threadIdx.x, outer_col, inner_col, outer_row, inner_row, 
            //   //   float(p3[0]), float(p3[1]), float(p3[2]), float(p3[3]), 
            //   //   float(p3[4]), float(p3[5]), float(p3[6]), float(p3[7]));

            // }

          }
        }
      }
    }
  }
};

} // namespace mma

} // namespace gemm
} // namespace tv
