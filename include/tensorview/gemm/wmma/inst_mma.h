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
#include <cuda_fp16.h>
#include <tensorview/core/all.h>

#include <tensorview/gemm/core/layout.h>
#include <tensorview/gemm/dtypes/all.h>

namespace tv {
namespace gemm {
namespace inst {

template <typename Shape, int NumThreads, typename TA, typename TB, typename TC,
          typename LA, typename LB, typename LC>
struct Mma;

template <typename TA, typename TB, typename TC, typename LA, typename LB,
          typename LC>
struct Mma<tv::mp_list_c<int, 1, 1, 1>, 1, TA, TB, TC, LA, LB, LC> {
  static constexpr tv::array<int, 3> kShape{1, 1, 1};
  using ElementA = TA;
  using ElementB = TB;
  using ElementC = TC;
  using FragmentA = tv::array<ElementA, kShape[0]>;
  using FragmentB = tv::array<ElementB, kShape[1]>;
  using FragmentC = tv::array<ElementC, kShape[2]>;

  TV_HOST_DEVICE_INLINE void operator()(FragmentC &d, FragmentA const &a,
                                        FragmentB const &b,
                                        FragmentC const &c) {
    d[0] = a[0] * b[0] + c[0];
  }
};

template <typename LA, typename LB, typename LC>
struct Mma<tv::mp_list_c<int, 1, 1, 1>, 1, half_t, half_t, float, LA, LB, LC> {
  using ElementA = half_t;
  using ElementB = half_t;
  using ElementC = float;
  static constexpr tv::array<int, 3> kShape{1, 1, 1};
  using FragmentA = tv::array<ElementA, 1>;
  using FragmentB = tv::array<ElementB, 1>;
  using FragmentC = tv::array<ElementC, 1>;
  // TODO stmts below produce strange error.
  // using FragmentA = tv::array<ElementA, int(kShape[0])>;
  // using FragmentB = tv::array<ElementB, int(kShape[1])>;
  // using FragmentC = tv::array<ElementC, int(kShape[2])>;

  TV_HOST_DEVICE_INLINE void operator()(FragmentC &d, FragmentA const &a,
                                        FragmentB const &b,
                                        FragmentC const &c) {
    d[0] = float(a[0]) * float(b[0]) + c[0];
  }
};

template <typename LA, typename LB, typename LC>
struct Mma<tv::mp_list_c<int, 1, 1, 4>, 1, int8_t, int8_t, int32_t, LA, LB,
           LC> {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = int32_t;
  static constexpr tv::array<int, 3> kShape{1, 1, 4};
  using FragmentA = tv::array<ElementA, kShape[2]>;
  using FragmentB = tv::array<ElementB, kShape[2]>;
  using FragmentC = tv::array<ElementC, 1>;
  TV_HOST_DEVICE_INLINE void operator()(FragmentC &d, FragmentA const &a,
                                        FragmentB const &b,
                                        FragmentC const &c) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610))
    unsigned const &A = reinterpret_cast<unsigned const &>(a);
    unsigned const &B = reinterpret_cast<unsigned const &>(b);
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                 : "=r"(d[0])
                 : "r"(A), "r"(B), "r"(c[0]));
#else
    d[0] = c[0];
    TV_PRAGMA_UNROLL
    for (int k = 0; k < kShape[2]; ++k) {
      d[0] += a[k] * b[k];
    }

#endif
  }
};

template <typename LA, typename LB, typename LC>
struct Mma<tv::mp_list_c<int, 1, 1, 2>, 1, int16_t, int16_t, int32_t, LA, LB,
           LC> {
  using ElementA = int16_t;
  using ElementB = int16_t;
  using ElementC = int32_t;
  static constexpr tv::array<int, 3> kShape{1, 1, 2};
  using FragmentA = tv::array<ElementA, kShape[2]>;
  using FragmentB = tv::array<ElementB, kShape[2]>;
  using FragmentC = tv::array<ElementC, 1>;

  TV_HOST_DEVICE_INLINE void operator()(FragmentC &d, FragmentA const &a,
                                        FragmentB const &b,
                                        FragmentC const &c) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610))
    unsigned const &A = reinterpret_cast<unsigned const &>(a);
    unsigned const &B = reinterpret_cast<unsigned const &>(b);

    asm volatile("dp2a.s32.s32 %0, %1, %2, %3;"
                 : "=r"(d[0])
                 : "r"(A), "r"(B), "r"(c[0]));
#else
    d[0] = c[0];
    TV_PRAGMA_UNROLL
    for (int k = 0; k < kShape[2]; ++k) {
      d[0] += a[k] * b[k];
    }
#endif
  }
};

template <typename LA, typename LB, typename LC>
struct Mma<tv::mp_list_c<int, 2, 1, 1>, 1, half_t, half_t, half_t, LA, LB,
           LC> {
  using ElementA = half_t;
  using ElementB = half_t;
  using ElementC = half_t;
  static constexpr tv::array<int, 3> kShape{2, 1, 1};
  using FragmentA = tv::array<ElementA, kShape[0]>;
  using FragmentB = tv::array<ElementB, kShape[1]>;
  using FragmentC = tv::array<ElementC, kShape[1] * kShape[0]>;

  TV_HOST_DEVICE_INLINE void operator()(FragmentC &d, FragmentA const &a,
                                        FragmentB const &b,
                                        FragmentC const &c) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600))
    __half2 const & A = reinterpret_cast<__half2 const &>(a);
    __half2 B = __half2half2(reinterpret_cast<__half const &>(b));
    __half2 const & C = reinterpret_cast<__half2 const &>(c);

    __half2 D = __hfma2(A, B, C);

    d = reinterpret_cast<array<half_t, 2> &>(D);
#else
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      d[i] = a[i] * b[0] + c[i];
    }
#endif
  }
};

template <typename LA, typename LB>
struct Mma<tv::mp_list_c<int, 1, 2, 1>, 1, half_t, half_t, half_t, LA, LB,
           layout::RowMajor> {
  using ElementA = half_t;
  using ElementB = half_t;
  using ElementC = half_t;
  static constexpr tv::array<int, 3> kShape{1, 2, 1};
  using FragmentA = tv::array<ElementA, kShape[0]>;
  using FragmentB = tv::array<ElementB, kShape[1]>;
  using FragmentC = tv::array<ElementC, kShape[1] * kShape[0]>;

  TV_HOST_DEVICE_INLINE void operator()(FragmentC &d, FragmentA const &a,
                                        FragmentB const &b,
                                        FragmentC const &c) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600))
    __half2 const & A = __half2half2(reinterpret_cast<__half const &>(a));
    __half2 B = reinterpret_cast<__half2 const &>(b);
    __half2 const & C = reinterpret_cast<__half2 const &>(c);

    __half2 D = __hfma2(A, B, C);

    d = reinterpret_cast<array<half_t, 2> &>(D);

#else
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      d[i] = a[0] * b[i] + c[i];
    }
#endif
  }
};

template <typename LC>
struct Mma<tv::mp_list_c<int, 2, 2, 1>, 1, half_t, half_t, half_t, layout::ColumnMajor, layout::RowMajor,
           LC> {
  using ElementA = half_t;
  using ElementB = half_t;
  using ElementC = half_t;
  static constexpr tv::array<int, 3> kShape{2, 2, 1};
  using FragmentA = tv::array<ElementA, kShape[0]>;
  using FragmentB = tv::array<ElementB, kShape[1]>;
  using FragmentC = tv::array<ElementC, kShape[1] * kShape[0]>;
  static constexpr bool kRowMajorC = std::is_same<LC, layout::RowMajor>::value;

  TV_HOST_DEVICE_INLINE void operator()(FragmentC &d, FragmentA const &a,
                                        FragmentB const &b,
                                        FragmentC const &c) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600))
  if TV_IF_CONSTEXPR (!kRowMajorC){
    __half2 const & A = reinterpret_cast<__half2 const &>(a);
    __half2 Blo = __low2half2(reinterpret_cast<__half2 const &>(b));
    __half2 Bhi = __high2half2(reinterpret_cast<__half2 const &>(b));

    __half2 const *C = reinterpret_cast<__half2 const *>(&c);

    __half2 Dlo = __hfma2(A, Blo, C[0]);
    __half2 Dhi = __hfma2(A, Bhi, C[1]);

    array<half_t, 2> * D = reinterpret_cast<array<half_t, 2> *>(&d);

    D[0] = reinterpret_cast<array<half_t, 2> const &>(Dlo);
    D[1] = reinterpret_cast<array<half_t, 2> const &>(Dhi);
  }else{
    __half2 Alo = __low2half2(reinterpret_cast<__half2 const &>(a));
    __half2 Ahi = __high2half2(reinterpret_cast<__half2 const &>(a));
    __half2 const & B = reinterpret_cast<__half2 const &>(b);
    
    __half2 const *C = reinterpret_cast<__half2 const *>(&c);

    __half2 Dlo = __hfma2(Alo, B, C[0]);
    __half2 Dhi = __hfma2(Ahi, B, C[1]);
    
    array<half_t, 2> * D = reinterpret_cast<array<half_t, 2> *>(&d);

    D[0] = reinterpret_cast<array<half_t, 2> &>(Dlo);
    D[1] = reinterpret_cast<array<half_t, 2> &>(Dhi);
  }

#else
  if TV_IF_CONSTEXPR (!kRowMajorC){
    TV_PRAGMA_UNROLL
    for (int j = 0; j < 2; ++j) {
      TV_PRAGMA_UNROLL
      for (int i = 0; i < 2; ++i) {
        d[i + 2 * j] = a[i] * b[j] + c[i + 2 * j];
      }
    }

  }else{
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      TV_PRAGMA_UNROLL
      for (int j = 0; j < 2; ++j) {
        d[i * 2 + j] = a[i] * b[j] + c[i * 2 + j];
      }
    }
  }
#endif
  }
};



#define TV_MMA_SYNC_ASM_HALF(LA, LB, TD, TA, TB, TC)                           \
  asm volatile("mma.sync.aligned.m8n8k4." #LA "." #LB "." #TD "." #TA "." #TB  \
               "." #TC " {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};\n"  \
               : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])                \
               : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "r"(C[0]),        \
                 "r"(C[1]), "r"(C[2]), "r"(C[3]));

#define TV_MMA_SYNC_ASM_FLOAT(LA, LB, TD, TA, TB, TC)                          \
  asm volatile("mma.sync.aligned.m8n8k4." #LA "." #LB "." #TD "." #TA "." #TB  \
               "." #TC "  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11},"      \
               "{%12,%13,%14,%15,%16,%17,%18,%19};\n"                          \
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]),   \
                 "=f"(D[5]), "=f"(D[6]), "=f"(D[7])                            \
               : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "f"(C[0]),        \
                 "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]),        \
                 "f"(C[6]), "f"(C[7]));

template <typename TA, typename TB, typename TC, typename LA, typename LB,
          typename LC>
struct Mma<tv::mp_list_c<int, 8, 8, 4>, 8, TA, TB, TC, LA, LB, LC> {
  static constexpr tv::array<int, 3> kShape{8, 8, 4};
  using ElementA = TA;
  using ElementB = TB;
  using ElementC = TC;
  using FragmentA = tv::array<ElementA, 4>;
  using FragmentB = tv::array<ElementB, 4>;
  using FragmentC = tv::array<ElementC, 8>;
  static constexpr bool kRowMajorA = std::is_same<LA, layout::RowMajor>::value;
  static constexpr bool kRowMajorB = std::is_same<LB, layout::RowMajor>::value;
  static constexpr bool kRowMajorC = std::is_same<LC, layout::RowMajor>::value;
  static constexpr bool kCisFloat = std::is_same<ElementC, float>::value;

  TV_DEVICE_INLINE void operator()(FragmentC &d, FragmentA const &a,
                                   FragmentB const &b, FragmentC const &c) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const *B = reinterpret_cast<unsigned const *>(&b);
    if TV_IF_CONSTEXPR (kCisFloat) {
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);
      if TV_IF_CONSTEXPR (kRowMajorA && kRowMajorB) {
        TV_MMA_SYNC_ASM_FLOAT(row, row, f32, f16, f16, f32);
      } else if (!kRowMajorA && kRowMajorB) {
        TV_MMA_SYNC_ASM_FLOAT(col, row, f32, f16, f16, f32);
      } else if (kRowMajorA && !kRowMajorB) {
        TV_MMA_SYNC_ASM_FLOAT(row, col, f32, f16, f16, f32);
      } else {
        TV_MMA_SYNC_ASM_FLOAT(col, col, f32, f16, f16, f32);
      }
    } else {
      unsigned const *C = reinterpret_cast<unsigned const *>(&c);
      unsigned *D = reinterpret_cast<unsigned *>(&d);
      if TV_IF_CONSTEXPR (kRowMajorA && kRowMajorB) {
        TV_MMA_SYNC_ASM_HALF(row, row, f16, f16, f16, f16);
      } else if (!kRowMajorA && kRowMajorB) {
        TV_MMA_SYNC_ASM_HALF(col, row, f16, f16, f16, f16);
      } else if (kRowMajorA && !kRowMajorB) {
        TV_MMA_SYNC_ASM_HALF(row, col, f16, f16, f16, f16);
      } else {
        TV_MMA_SYNC_ASM_HALF(col, col, f16, f16, f16, f16);
      }
    }

#endif
  }
};
#undef TV_MMA_SYNC_ASM_HALF
#undef TV_MMA_SYNC_ASM_FLOAT

} // namespace inst

} // namespace gemm
} // namespace tv