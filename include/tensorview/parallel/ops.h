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

/*
TODO add some ops (e.g. atomic) that can be used in both cuda and openmp.
TODO add CAS
*/
#pragma once
#include <tensorview/core/all.h>
namespace tv {

template <typename T> struct AtomicFetchAdd {
  TV_HOST_DEVICE_INLINE T operator()(T *addr, T val) {
#ifdef __CUDA_ARCH__
    return atomicAdd(addr, val);
#else
    T t;
#ifdef _MSC_VER
#pragma omp critical
#else
#pragma omp atomic capture
#endif
    {
      t = *addr;
      *addr += val;
    }
    return t;
#endif
  }
};

template <typename T> struct AtomicAdd {
  TV_HOST_DEVICE_INLINE void operator()(T *addr, T val) {
#ifdef __CUDA_ARCH__
    atomicAdd(addr, val);
#else
#ifdef _MSC_VER
#pragma omp critical
    { 
      addr[0] += val; 
    }

#else
#pragma omp atomic
    addr[0] += val; 
#endif
#endif
  }
};

} // namespace tv