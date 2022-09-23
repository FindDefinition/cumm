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
#if (!defined(__CUDACC_RTC__) &&  !defined(__CUDACC__) && !defined(__CUDA_ARCH__))
  static T copysign(T x, T y) { return std::copysign(x, y); }

  static T atan2(T y, T x) { return std::atan2(y, x); }

  static T scalbn(T x, int n) { return std::scalbn(x, n); }

  static T pow(T x, T n) { return std::pow(x, n); }

  static T fmod(T x, T n) { return std::fmod(x, n); }

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

#endif
};

#ifdef __CUDACC__
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

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template <> struct MathScalarOp<__half> {

  TV_DEVICE_INLINE static __half sqrt(__half x) { return hsqrt(x); }

  TV_DEVICE_INLINE static __half rsqrt(__half x) { return hrsqrt(x); }

  TV_DEVICE_INLINE static __half ceil(__half x) { return hceil(x); }

  TV_DEVICE_INLINE static __half cos(__half x) { return hcos(x); }

  TV_DEVICE_INLINE static __half exp(__half x) { return hexp(x); }

  TV_DEVICE_INLINE static __half exp10(__half x) { return hexp10(x); }

  TV_DEVICE_INLINE static __half exp2(__half x) { return hexp2(x); }

  TV_DEVICE_INLINE static __half floor(__half x) { return hfloor(x); }

  TV_DEVICE_INLINE static __half log(__half x) { return hlog(x); }

  TV_DEVICE_INLINE static __half log10(__half x) { return hlog10(x); }

  TV_DEVICE_INLINE static __half log2(__half x) { return hlog2(x); }

  TV_DEVICE_INLINE static __half rint(__half x) { return hrint(x); }

  TV_DEVICE_INLINE static __half sin(__half x) { return hsin(x); }

  TV_DEVICE_INLINE static __half trunc(__half x) { return htrunc(x); }

  TV_DEVICE_INLINE static __half fabs(__half x) { return __habs(x); }

  TV_DEVICE_INLINE static __half2 v2sqrt(__half2 x) { return h2sqrt(x); }

  TV_DEVICE_INLINE static __half2 v2rsqrt(__half2 x) { return h2rsqrt(x); }

  TV_DEVICE_INLINE static __half2 v2ceil(__half2 x) { return h2ceil(x); }

  TV_DEVICE_INLINE static __half2 v2cos(__half2 x) { return h2cos(x); }

  TV_DEVICE_INLINE static __half2 v2exp(__half2 x) { return h2exp(x); }

  TV_DEVICE_INLINE static __half2 v2exp10(__half2 x) { return h2exp10(x); }

  TV_DEVICE_INLINE static __half2 v2exp2(__half2 x) { return h2exp2(x); }

  TV_DEVICE_INLINE static __half2 v2floor(__half2 x) { return h2floor(x); }

  TV_DEVICE_INLINE static __half2 v2log(__half2 x) { return h2log(x); }

  TV_DEVICE_INLINE static __half2 v2log10(__half2 x) { return h2log10(x); }

  TV_DEVICE_INLINE static __half2 v2log2(__half2 x) { return h2log2(x); }

  TV_DEVICE_INLINE static __half2 v2rint(__half2 x) { return h2rint(x); }

  TV_DEVICE_INLINE static __half2 v2sin(__half2 x) { return h2sin(x); }

  TV_DEVICE_INLINE static __half2 v2trunc(__half2 x) { return h2trunc(x); }

  TV_DEVICE_INLINE static __half2 v2fabs(__half2 x) { return __habs2(x); }
};
#endif

#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <> struct MathScalarOp<__nv_bfloat16> {

  TV_DEVICE_INLINE static __nv_bfloat16 sqrt(__nv_bfloat16 x) {
    return hsqrt(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat16 rsqrt(__nv_bfloat16 x) {
    return hrsqrt(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat16 ceil(__nv_bfloat16 x) {
    return hceil(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat16 cos(__nv_bfloat16 x) { return hcos(x); }

  TV_DEVICE_INLINE static __nv_bfloat16 exp(__nv_bfloat16 x) { return hexp(x); }

  TV_DEVICE_INLINE static __nv_bfloat16 exp10(__nv_bfloat16 x) {
    return hexp10(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat16 exp2(__nv_bfloat16 x) {
    return hexp2(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat16 floor(__nv_bfloat16 x) {
    return hfloor(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat16 log(__nv_bfloat16 x) { return hlog(x); }

  TV_DEVICE_INLINE static __nv_bfloat16 log10(__nv_bfloat16 x) {
    return hlog10(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat16 log2(__nv_bfloat16 x) {
    return hlog2(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat16 rint(__nv_bfloat16 x) {
    return hrint(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat16 sin(__nv_bfloat16 x) { return hsin(x); }

  TV_DEVICE_INLINE static __nv_bfloat16 trunc(__nv_bfloat16 x) {
    return htrunc(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat16 fabs(__nv_bfloat16 x) {
    return __habs(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2sqrt(__nv_bfloat162 x) {
    return h2sqrt(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2rsqrt(__nv_bfloat162 x) {
    return h2rsqrt(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2ceil(__nv_bfloat162 x) {
    return h2ceil(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2cos(__nv_bfloat162 x) {
    return h2cos(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2exp(__nv_bfloat162 x) {
    return h2exp(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2exp10(__nv_bfloat162 x) {
    return h2exp10(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2exp2(__nv_bfloat162 x) {
    return h2exp2(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2floor(__nv_bfloat162 x) {
    return h2floor(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2log(__nv_bfloat162 x) {
    return h2log(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2log10(__nv_bfloat162 x) {
    return h2log10(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2log2(__nv_bfloat162 x) {
    return h2log2(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2rint(__nv_bfloat162 x) {
    return h2rint(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2sin(__nv_bfloat162 x) {
    return h2sin(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2trunc(__nv_bfloat162 x) {
    return h2trunc(x);
  }

  TV_DEVICE_INLINE static __nv_bfloat162 v2fabs(__nv_bfloat162 x) {
    return __habs2(x);
  }
};
#endif

#endif
} // namespace arrayops
} // namespace tv