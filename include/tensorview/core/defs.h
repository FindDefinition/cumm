// Copyright 2021 Yan Yan
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
// #include <cstdint>
// #include <cstddef>
#if (defined(__clang__) && defined(__CUDA__)) || defined(__NVCC__)
#define TV_CUDA_CC
#define TV_HOST_DEVICE_INLINE __forceinline__ __device__ __host__
#define TV_DEVICE_INLINE __forceinline__ __device__
#define TV_DEVICE __device__
#define TV_HOST_DEVICE __device__ __host__
#define TV_GPU_LAMBDA __device__ __host__
#define TV_GPU_LAMBDA_DEVICE __device__
#define TV_ASSERT(expr) assert(expr)
#define TV_IS_EXTEND_LAMBDA(x) __nv_is_extended_host_device_lambda_closure_type(x)
#define TV_IS_EXTEND_DEVICE_LAMBDA(x) __nv_is_extended_device_lambda_closure_type(x)

#elif defined(__CUDACC_RTC__)
#define TV_CUDA_CC
#define TV_ASSERT(expr) assert(expr)
#define TV_HOST_DEVICE_INLINE __forceinline__ __device__
#define TV_DEVICE_INLINE __forceinline__ __device__
#define TV_HOST_DEVICE __device__ __host__
#define TV_DEVICE __device__
#define TV_GPU_LAMBDA __device__ __host__
#define TV_GPU_LAMBDA_DEVICE __device__
#define TV_IS_EXTEND_LAMBDA(x) __nv_is_extended_host_device_lambda_closure_type(x)
#define TV_IS_EXTEND_DEVICE_LAMBDA(x) __nv_is_extended_device_lambda_closure_type(x)
#else

#define TV_ASSERT(x) assert(x)
#define TV_HOST_DEVICE_INLINE inline
#define TV_HOST_DEVICE
#define TV_DEVICE
#define TV_DEVICE_INLINE inline
#define TV_GPU_LAMBDA
#define TV_GPU_LAMBDA_DEVICE
#define TV_IS_EXTEND_LAMBDA(x) true
#define TV_IS_EXTEND_DEVICE_LAMBDA(x) true
#endif

#ifndef TV_MAX_DIM
#define TV_MAX_DIM 10
#endif
#ifndef TV_GLOBAL_INDEX
#define TV_GLOBAL_INDEX int64_t
#endif

#if defined(__CUDA_ARCH__)
#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
#define TV_PRAGMA_UNROLL _Pragma("unroll")
#define TV_PRAGMA_NO_UNROLL _Pragma("unroll 1")
#else
#define TV_PRAGMA_UNROLL #pragma unroll
#define TV_PRAGMA_NO_UNROLL #pragma unroll 1
#endif
#else
#define TV_PRAGMA_UNROLL
#define TV_PRAGMA_NO_UNROLL
#endif

#if __cplusplus >= 201703L
#define TV_IF_CONSTEXPR constexpr 
#else 
#define TV_IF_CONSTEXPR 
#endif