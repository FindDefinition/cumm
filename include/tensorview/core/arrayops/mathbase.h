// Copyright 2022 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <tensorview/core/defs.h>
#ifndef __CUDACC_RTC__
#include <cmath>
#endif
#ifdef __CUDACC__
#include <cuda_fp16.h>
#if (__CUDACC_VER_MAJOR__ >= 11)
#include <cuda_bf16.h>
#endif
#endif

namespace tv {

namespace arrayops {

template <typename T> struct MathScalarOp {
#ifndef __CUDACC__
  static T copysign(T x, T y) { return std::copysign(x, y); }

  static T atan2(T y, T x) { return std::atan2(y, x); }

  static T scalbn(T x, int n) { return std::scalbn(x, n); }

  static T pow(T x, T n) { return std::pow(x, n); }

  static T fmod(T x, T n) { return std::fmod(x, n); }

  static T neg(T x) { return -x; }

  static T sqrt(T x) { return std::sqrt(x); }

  static T ceil(T x) { return std::ceil(x); }

  static T cos(T x) { return std::cos(x); }

  static T exp(T x) { return std::exp(x); }

  static T exp2(T x) { return std::exp2(x); }

  static T floor(T x) { return std::floor(x); }

  static T log(T x) { return std::log(x); }

  static T log10(T x) { return std::log10(x); }

  static T log2(T x) { return std::log2(x); }

  static T rint(T x) { return std::rint(x); }

  static T sin(T x) { return std::sin(x); }

  static T trunc(T x) { return std::trunc(x); }

  static T fabs(T x) { return std::fabs(x); }

  static T tan(T x) { return std::tan(x); }

  static T asin(T x) { return std::asin(x); }

  static T acos(T x) { return std::acos(x); }

  static T atan(T x) { return std::atan(x); }

  static T round(T x) { return std::round(x); }

  static T sinh(T x) { return std::sinh(x); }

  static T cosh(T x) { return std::cosh(x); }

  static T tanh(T x) { return std::tanh(x); }

  static T asinh(T x) { return std::asinh(x); }

  static T acosh(T x) { return std::acosh(x); }

  static T atanh(T x) { return std::atanh(x); }

  static T rsqrt(T x) { return T(1) / sqrt(x); }
#else

  TV_HOST_DEVICE_INLINE static T copysign(T x, T y) {
    return T(copysignf(float(x), float(y)));
  }

  TV_HOST_DEVICE_INLINE static T atan2(T y, T x) {
    return T(atan2f(float(y), float(x)));
  }

  TV_HOST_DEVICE_INLINE static T scalbn(T x, int n) {
    return T(scalbnf(float(x), n));
  }

  TV_HOST_DEVICE_INLINE static T pow(T x, T n) {
    return T(pow(float(x), float(n)));
  }

  TV_HOST_DEVICE_INLINE static T fmod(T x, T n) {
    return T(fmodf(float(x), float(n)));
  }

  TV_HOST_DEVICE_INLINE static T sqrt(T x) { return T(sqrtf(float(x))); }

  TV_HOST_DEVICE_INLINE static T ceil(T x) { return T(ceilf(float(x))); }

  TV_HOST_DEVICE_INLINE static T cos(T x) { return T(cosf(float(x))); }

  TV_HOST_DEVICE_INLINE static T exp(T x) { return T(expf(float(x))); }

  TV_HOST_DEVICE_INLINE static T exp2(T x) { return T(exp2f(float(x))); }

  TV_HOST_DEVICE_INLINE static T floor(T x) { return T(floorf(float(x))); }

  TV_HOST_DEVICE_INLINE static T log(T x) { return T(logf(float(x))); }

  TV_HOST_DEVICE_INLINE static T log10(T x) { return T(log10f(float(x))); }

  TV_HOST_DEVICE_INLINE static T log2(T x) { return T(log2f(float(x))); }

  TV_HOST_DEVICE_INLINE static T rint(T x) { return T(rintf(float(x))); }

  TV_HOST_DEVICE_INLINE static T sin(T x) { return T(sinf(float(x))); }

  TV_HOST_DEVICE_INLINE static T trunc(T x) { return T(truncf(float(x))); }

  TV_HOST_DEVICE_INLINE static T fabs(T x) { return T(fabsf(float(x))); }

  TV_HOST_DEVICE_INLINE static T tan(T x) { return T(tanf(float(x))); }

  TV_HOST_DEVICE_INLINE static T asin(T x) { return T(asinf(float(x))); }

  TV_HOST_DEVICE_INLINE static T acos(T x) { return T(acosf(float(x))); }

  TV_HOST_DEVICE_INLINE static T atan(T x) { return T(atanf(float(x))); }

  TV_HOST_DEVICE_INLINE static T round(T x) { return T(roundf(float(x))); }

  TV_HOST_DEVICE_INLINE static T sinh(T x) { return T(sinhf(float(x))); }

  TV_HOST_DEVICE_INLINE static T cosh(T x) { return T(coshf(float(x))); }

  TV_HOST_DEVICE_INLINE static T tanh(T x) { return T(tanhf(float(x))); }

  TV_HOST_DEVICE_INLINE static T asinh(T x) { return T(asinhf(float(x))); }

  TV_HOST_DEVICE_INLINE static T acosh(T x) { return T(acoshf(float(x))); }

  TV_HOST_DEVICE_INLINE static T atanh(T x) { return T(atanhf(float(x))); }

  TV_HOST_DEVICE_INLINE static T neg(T x) { return -x; }

  TV_HOST_DEVICE_INLINE static T rsqrt(T x) { return T(1) / sqrt(x); }

#endif
};

#ifdef TV_CUDA_CC
template <> struct MathScalarOp<float> {

  TV_HOST_DEVICE_INLINE static float copysign(float x, float y) {
    return copysignf(x, y);
  }

  TV_HOST_DEVICE_INLINE static float atan2(float y, float x) {
    return atan2f(y, x);
  }

  TV_HOST_DEVICE_INLINE static float scalbn(float x, int n) {
    return scalbnf(x, n);
  }

  TV_HOST_DEVICE_INLINE static float pow(float x, float n) {
    return powf(x, n);
  }

  TV_HOST_DEVICE_INLINE static float fmod(float x, float n) {
    return fmodf(x, n);
  }
  TV_HOST_DEVICE_INLINE static float neg(float x) { return -x; }

  TV_HOST_DEVICE_INLINE static float sqrt(float x) { return sqrtf(x); }

  TV_HOST_DEVICE_INLINE static float rsqrt(float x) { return rsqrtf(x); }

  TV_HOST_DEVICE_INLINE static float ceil(float x) { return ceilf(x); }

  TV_HOST_DEVICE_INLINE static float cos(float x) { return cosf(x); }

  TV_HOST_DEVICE_INLINE static float exp(float x) { return expf(x); }

  TV_HOST_DEVICE_INLINE static float exp10(float x) { return exp10f(x); }

  TV_HOST_DEVICE_INLINE static float exp2(float x) { return exp2f(x); }

  TV_HOST_DEVICE_INLINE static float floor(float x) { return floorf(x); }

  TV_HOST_DEVICE_INLINE static float log(float x) { return logf(x); }

  TV_HOST_DEVICE_INLINE static float log10(float x) { return log10f(x); }

  TV_HOST_DEVICE_INLINE static float log2(float x) { return log2f(x); }

  TV_HOST_DEVICE_INLINE static float rint(float x) { return rintf(x); }

  TV_HOST_DEVICE_INLINE static float sin(float x) { return sinf(x); }

  TV_HOST_DEVICE_INLINE static float trunc(float x) { return truncf(x); }

  TV_HOST_DEVICE_INLINE static float fabs(float x) { return fabsf(x); }

  TV_HOST_DEVICE_INLINE static float tan(float x) { return tanf(x); }

  TV_HOST_DEVICE_INLINE static float asin(float x) { return asinf(x); }

  TV_HOST_DEVICE_INLINE static float acos(float x) { return acosf(x); }

  TV_HOST_DEVICE_INLINE static float atan(float x) { return atanf(x); }

  TV_HOST_DEVICE_INLINE static float round(float x) { return roundf(x); }

  TV_HOST_DEVICE_INLINE static float sinh(float x) { return sinhf(x); }

  TV_HOST_DEVICE_INLINE static float cosh(float x) { return coshf(x); }

  TV_HOST_DEVICE_INLINE static float tanh(float x) { return tanhf(x); }

  TV_HOST_DEVICE_INLINE static float asinh(float x) { return asinhf(x); }

  TV_HOST_DEVICE_INLINE static float acosh(float x) { return acoshf(x); }

  TV_HOST_DEVICE_INLINE static float atanh(float x) { return atanhf(x); }
};
#ifdef __CUDACC__
template <> struct MathScalarOp<__half> {

  TV_DEVICE_INLINE static __half sqrt(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hsqrt(x);
#else
    return __half(sqrtf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half rsqrt(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hrsqrt(x);
#else
    return __half(rsqrtf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half ceil(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hceil(x);
#else
    return __half(ceilf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half cos(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hcos(x);
#else
    return __half(cosf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half exp(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hexp(x);
#else
    return __half(expf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half exp10(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hexp10(x);
#else
    return __half(exp10f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half exp2(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hexp2(x);
#else
    return __half(exp2f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half floor(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hfloor(x);
#else
    return __half(floorf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half log(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hlog(x);
#else
    return __half(logf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half log10(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hlog10(x);
#else
    return __half(log10f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half log2(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hlog2(x);
#else
    return __half(log2f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half rint(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hrint(x);
#else
    return __half(rintf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half sin(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return hsin(x);
#else
    return __half(sinf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half trunc(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return htrunc(x);
#else
    return __half(truncf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half fabs(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __habs(x);
#else
    return __half(fabsf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __half2 v2sqrt(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2sqrt(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = sqrtf(x0);
    x1 = sqrtf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2rsqrt(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2rsqrt(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = rsqrtf(x0);
    x1 = rsqrtf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2ceil(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2ceil(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = ceilf(x0);
    x1 = ceilf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2cos(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2cos(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = cosf(x0);
    x1 = cosf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2exp(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2exp(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = expf(x0);
    x1 = expf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2exp10(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2exp10(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = exp10f(x0);
    x1 = exp10f(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2exp2(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2exp2(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = exp2f(x0);
    x1 = exp2f(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2floor(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2floor(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = floorf(x0);
    x1 = floorf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2log(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2log(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = logf(x0);
    x1 = logf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2log10(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2log10(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = log10f(x0);
    x1 = log10f(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2log2(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2log2(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = log2f(x0);
    x1 = log2f(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2rint(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2rint(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = rintf(x0);
    x1 = rintf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2sin(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2sin(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = sinf(x0);
    x1 = sinf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2trunc(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return h2trunc(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = truncf(x0);
    x1 = truncf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __half2 v2fabs(__half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __habs2(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = fabsf(x0);
    x1 = fabsf(x1);
    return __floats2half2_rn(x0, x1);
#endif
  }

  TV_HOST_DEVICE_INLINE static __half neg(__half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __hneg(x);
#else
    return __half(-(float(x)));
#endif
  }
};
#endif
#if (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11))
template <> struct MathScalarOp<__nv_bfloat16> {

  TV_DEVICE_INLINE static __nv_bfloat16 sqrt(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hsqrt(x);
#else
    return __nv_bfloat16(sqrtf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 rsqrt(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hrsqrt(x);
#else
    return __nv_bfloat16(rsqrtf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 ceil(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hceil(x);
#else
    return __nv_bfloat16(ceilf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 cos(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hcos(x);
#else
    return __nv_bfloat16(cosf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 exp(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hexp(x);
#else
    return __nv_bfloat16(expf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 exp10(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hexp10(x);
#else
    return __nv_bfloat16(exp10f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 exp2(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hexp2(x);
#else
    return __nv_bfloat16(exp2f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 floor(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hfloor(x);
#else
    return __nv_bfloat16(floorf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 log(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hlog(x);
#else
    return __nv_bfloat16(logf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 log10(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hlog10(x);
#else
    return __nv_bfloat16(log10f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 log2(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hlog2(x);
#else
    return __nv_bfloat16(log2f(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 rint(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hrint(x);
#else
    return __nv_bfloat16(rintf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 sin(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return hsin(x);
#else
    return __nv_bfloat16(sinf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 trunc(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return htrunc(x);
#else
    return __nv_bfloat16(truncf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat16 fabs(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __habs(x);
#else
    return __nv_bfloat16(fabsf(float(x)));
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2sqrt(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2sqrt(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = sqrtf(x0);
    x1 = sqrtf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2rsqrt(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2rsqrt(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = rsqrtf(x0);
    x1 = rsqrtf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2ceil(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2ceil(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = ceilf(x0);
    x1 = ceilf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2cos(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2cos(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = cosf(x0);
    x1 = cosf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2exp(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2exp(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = expf(x0);
    x1 = expf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2exp10(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2exp10(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = exp10f(x0);
    x1 = exp10f(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2exp2(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2exp2(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = exp2f(x0);
    x1 = exp2f(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2floor(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2floor(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = floorf(x0);
    x1 = floorf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2log(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2log(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = logf(x0);
    x1 = logf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2log10(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2log10(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = log10f(x0);
    x1 = log10f(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2log2(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2log2(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = log2f(x0);
    x1 = log2f(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2rint(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2rint(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = rintf(x0);
    x1 = rintf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2sin(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2sin(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = sinf(x0);
    x1 = sinf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2trunc(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return h2trunc(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = truncf(x0);
    x1 = truncf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2fabs(__nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __habs2(x);
#else
    auto x0 = __low2float(x);
    auto x1 = __high2float(x);
    x0 = fabsf(x0);
    x1 = fabsf(x1);
    return __floats2bfloat162_rn(x0, x1);
#endif
  }

  TV_HOST_DEVICE_INLINE static __nv_bfloat16 neg(__nv_bfloat16 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __hneg(x);
#else
    return __nv_bfloat16(-(float(x)));
#endif
  }
};
#endif

#endif

} // namespace arrayops
} // namespace tv