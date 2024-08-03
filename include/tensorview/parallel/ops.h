// Copyright 2024 Yan Yan
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

/*
TODO add some ops (e.g. atomic) that can be used in both cuda and openmp.
TODO add CAS
*/
#pragma once
#include <tensorview/core/all.h>
#if defined(TV_METAL_RTC)
#include <metal_stdlib>
#endif
#if defined(TV_HARDWARE_ACC_CUDA) && defined(__CUDACC_RTC__)
#include <tensorview/core/nvrtc_std.h>
#include <cooperative_groups.h>
#endif

namespace tv {
namespace parallel {
template <typename T1, typename T2> struct DataPair {
  T1 first;
  T2 second;
  static_assert(sizeof(T1) == sizeof(T2), "error");
  static_assert((sizeof(T1) == 4 || sizeof(T1) == 2 || sizeof(T1) == 1),
                "error");
};
namespace detail {

template <int Size> struct AtomicDataType;

template <> struct AtomicDataType<2> {
  using type = uint32_t;
};

template <> struct AtomicDataType<4> {
  using type = uint64_t;
};

template <typename T1, typename T2> union DataPairUnion {
  DataPair<T1, T2> data;
  typename AtomicDataType<sizeof(T1)>::type val;
};

} // namespace detail

#if defined(TV_METAL_RTC) && defined(__METAL_VERSION__) &&                     \
    __METAL_VERSION__ >= 310
namespace internal {
uint __apple_metal_warp_size [[threads_per_simdgroup]];
uint __apple_metal_warp_index [[simdgroup_index_in_threadgroup]];
uint __apple_metal_lane_index [[thread_index_in_simdgroup]];
}

using vote_t = metal::simd_vote::vote_t;

TV_DEVICE_INLINE uint warp_size() { return internal::__apple_metal_warp_size; }

TV_DEVICE_INLINE uint warp_index() { return internal::__apple_metal_warp_index; }

TV_DEVICE_INLINE uint lane_index() { return internal::__apple_metal_lane_index; }

namespace detail {
TV_DEVICE_INLINE vote_t lanemask_lt() {
  return ~(metal::numeric_limits<vote_t>::max() << lane_index());
}
} // namespace detail

#endif

#if defined(TV_CUDA_CC)

using vote_t = uint32_t;
TV_DEVICE_INLINE uint32_t warp_size() { return 32; }

TV_DEVICE_INLINE uint32_t warp_index() { return threadIdx.x / 32; }

TV_DEVICE_INLINE uint32_t lane_index() { return threadIdx.x % 32; }

namespace detail {
TV_DEVICE_INLINE vote_t lanemask_lt() {
  vote_t lanemask32_lt;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask32_lt));
  return (lanemask32_lt);
}

} // namespace detail

#endif
namespace detail {
template <typename T> TV_DEVICE_INLINE T atomicAggIncCuda(TV_METAL_DEVICE T *ctr) {
#ifdef TV_CUDA_CC
  namespace cg = cooperative_groups;
  auto g = cg::coalesced_threads();
  T warp_res;
  if (g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, T(g.size()));
  return g.shfl(warp_res, 0) + g.thread_rank();

#endif
  return T(0);
}

}
template <typename T> TV_DEVICE_INLINE T atomicAggInc(TV_METAL_DEVICE T *ctr) {
#ifdef TV_METAL_RTC
  T warp_res = 0;
  auto mask = vote_t(metal::simd_active_threads_mask());
  auto thread_rank = metal::popcount(mask & detail::lanemask_lt());
  if (thread_rank == 0) {
    warp_res = metal::atomic_fetch_add_explicit(
        reinterpret_cast<device metal::atomic<T>*>(ctr), 
        T(metal::popcount(metal::simd_vote::vote_t(mask))), metal::memory_order_relaxed);
  }
  return metal::simd_shuffle(warp_res, metal::ctz(mask)) + thread_rank;
#else

  return detail::atomicAggIncCuda(ctr);

#endif
}

template <typename T> TV_DEVICE_INLINE T atomicAdd(TV_METAL_DEVICE T *ctr, T val) {
#ifdef TV_METAL_RTC
  return metal::atomic_fetch_add_explicit(reinterpret_cast<device metal::atomic<T>*>(ctr), val, metal::memory_order_relaxed);
#else
  return ::atomicAdd(ctr, val);
#endif
}

template <typename T> TV_DEVICE_INLINE T atomicMax(TV_METAL_DEVICE T *ctr, T val) {
#ifdef TV_METAL_RTC
  return metal::atomic_fetch_max_explicit(reinterpret_cast<device metal::atomic<T>*>(ctr), val, metal::memory_order_relaxed);
#else
  return ::atomicMax(ctr, val);
#endif
}

template <typename T> TV_DEVICE_INLINE T atomicMin(TV_METAL_DEVICE T *ctr, T val) {
#ifdef TV_METAL_RTC
  return metal::atomic_fetch_min_explicit(reinterpret_cast<device metal::atomic<T>*>(ctr), val, metal::memory_order_relaxed);
#else
  return ::atomicMin(ctr, val);
#endif
}

#ifdef TV_CUDA_RTC
TV_DEVICE_INLINE float atomicMax (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

TV_DEVICE_INLINE float atomicMin (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
          __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

#endif

#ifdef TV_METAL_RTC

TV_DEVICE_INLINE float __int_as_float(int x){
  return *reinterpret_cast<thread float*>(&x);
}

TV_DEVICE_INLINE int __float_as_int(float x){
  return *reinterpret_cast<thread int*>(&x);
}

TV_DEVICE_INLINE float __uint_as_float(unsigned int x){
  return *reinterpret_cast<thread float*>(&x);
}

TV_DEVICE_INLINE unsigned int __float_as_uint(float x){
  return *reinterpret_cast<thread unsigned int*>(&x);
}

TV_DEVICE_INLINE float atomicMax (device float * addr, float value) {
    float old;
    // *reinterpret_cast<float*>( &val )
    old = (value >= 0) ? __int_as_float(atomicMax((device int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((device unsigned int *)addr, __float_as_uint(value)));

    return old;
}

TV_DEVICE_INLINE float atomicMin (device float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((device int *)addr, __float_as_int(value))) :
          __uint_as_float(atomicMax((device unsigned int *)addr, __float_as_uint(value)));

    return old;
}

#endif


#ifdef TV_METAL_RTC

template <typename T> TV_DEVICE_INLINE T atomicAggInc(TV_METAL_THREADGROUP T *ctr) {
  T warp_res;
  auto mask = vote_t(metal::simd_active_threads_mask());
  auto thread_rank = metal::popcount(mask & detail::lanemask_lt());
  if (thread_rank == 0) {
    warp_res = metal::atomic_fetch_add_explicit(
        reinterpret_cast<device metal::atomic<T>*>(ctr),
        T(metal::popcount(metal::simd_vote::vote_t(mask))), metal::memory_order_relaxed);
  }
  return metal::simd_shuffle(warp_res, metal::ctz(mask)) + thread_rank;
}
template <typename T> TV_DEVICE_INLINE T atomicAdd(device metal::atomic<T> *ctr, T val) {
  return metal::atomic_fetch_add_explicit(ctr, val, metal::memory_order_relaxed);
}
template <typename T> TV_DEVICE_INLINE T atomicAdd(threadgroup T *ctr, T val) {
  return metal::atomic_fetch_add_explicit(reinterpret_cast<device metal::atomic<T>*>(ctr), val, metal::memory_order_relaxed);
}
template <typename T> TV_DEVICE_INLINE T atomicAdd(threadgroup metal::atomic<T> *ctr, T val) {
  return metal::atomic_fetch_add_explicit(ctr, val, metal::memory_order_relaxed);
}
template <typename T> TV_DEVICE_INLINE T atomicMax(device metal::atomic<T> *ctr, T val) {
  return metal::atomic_fetch_max_explicit(ctr, val, metal::memory_order_relaxed);
}
template <typename T> TV_DEVICE_INLINE T atomicMax(threadgroup T *ctr, T val) {
  return metal::atomic_fetch_max_explicit(reinterpret_cast<device metal::atomic<T>*>(ctr), val, metal::memory_order_relaxed);
}
template <typename T> TV_DEVICE_INLINE T atomicMax(threadgroup metal::atomic<T> *ctr, T val) {
  return metal::atomic_fetch_max_explicit(ctr, val, metal::memory_order_relaxed);
}

#endif

template <typename T1, typename T2>
TV_DEVICE_INLINE DataPair<T1, T2> atomicArgMax(TV_METAL_DEVICE DataPair<T1, T2> *addr, T1 first,
                                         T2 second) {
  // for apple, this function only support 32bit kv.
  using atomic_ptr_t = typename detail::AtomicDataType<sizeof(T1)>::type;
  detail::DataPairUnion<T1, T2> ret =
      *(reinterpret_cast<TV_METAL_DEVICE detail::DataPairUnion<T1, T2> *>(addr));
  detail::DataPairUnion<T1, T2> expected;
  expected.data.first = first;
  expected.data.second = second;
  while (first > ret.data.first) {
    atomic_ptr_t old = ret.val;
#if defined(TV_METAL_RTC)
    atomic_ptr_t old_for_apple = ret.val;
    atomic_ptr_t cur;
    bool success;
    do {
      success = metal::atomic_compare_exchange_weak_explicit(
          (device metal::atomic<atomic_ptr_t> *)addr, &old_for_apple, expected.val, metal::memory_order_relaxed,
          metal::memory_order_relaxed);
      cur = old_for_apple;
      old_for_apple = old;
    } while (!success && old_for_apple == cur);
    ret.val = cur;
#else
    ret.val = atomicCAS((atomic_ptr_t *)addr, old, expected.val);
#endif
    if (ret.val == old)
      // insert success
      break;
  }
  return ret.data;
}

template <typename T1, typename T2>
TV_DEVICE_INLINE DataPair<T1, T2> atomicArgMin(TV_METAL_DEVICE DataPair<T1, T2> *addr, T1 first,
                                         T2 second) {
  using atomic_ptr_t = typename detail::AtomicDataType<sizeof(T1)>::type;
  detail::DataPairUnion<T1, T2> ret =
      *(reinterpret_cast<TV_METAL_DEVICE detail::DataPairUnion<T1, T2> *>(addr));
  detail::DataPairUnion<T1, T2> expected;
  expected.data.first = first;
  expected.data.second = second;
  while (first < ret.data.first) {
    atomic_ptr_t old = ret.val;
#if defined(TV_METAL_RTC)
    atomic_ptr_t old_for_apple = ret.val;
    atomic_ptr_t cur;
    bool success;
    do {
      success = metal::atomic_compare_exchange_weak_explicit(
          (device metal::atomic<atomic_ptr_t> *)addr, &old_for_apple, expected.val, metal::memory_order_relaxed,
          metal::memory_order_relaxed);
      cur = old_for_apple;
      old_for_apple = old;
    } while (!success && old_for_apple == cur);
#else
    ret.val = atomicCAS((atomic_ptr_t *)addr, old, expected.val);
#endif
    if (ret.val == old)
      break;
  }
  return ret.data;
}


} // namespace parallel

} // namespace tv