#pragma once 
#include <cooperative_groups.h>

namespace tv {

template <typename T1, typename T2>
struct DataPair{
  T1 first;
  T2 second;
  static_assert(sizeof(T1) == sizeof(T2), "error");
  static_assert((sizeof(T1) == 4 || sizeof(T1) == 2 || sizeof(T1) == 1), "error");
};


namespace cuda {

namespace detail {


template <int Size>
struct AtomicDataType;

template <>
struct AtomicDataType<2> {
    using type = unsigned long;
};

template <>
struct AtomicDataType<4> {
    using type = unsigned long long;
};

template <typename T1, typename T2>
union DataPairUnion{
  DataPair<T1, T2> data;
  typename AtomicDataType<sizeof(T1)>::type val;

};

}

template <typename T1, typename T2>
__device__ DataPair<T1, T2> atomicArgMax(DataPair<T1, T2> *addr, T1 first, T2 second) {
  using atomic_ptr_t = typename detail::AtomicDataType<sizeof(T1)>::type;
  detail::DataPairUnion<T1, T2> ret = *(reinterpret_cast<detail::DataPairUnion<T1, T2>*>(addr));
  detail::DataPairUnion<T1, T2> expected;
  expected.data.first = first;
  expected.data.second = second;
  while (first > ret.data.first) {
    atomic_ptr_t old = ret.val;
    ret.val = atomicCAS((atomic_ptr_t *)addr, old,
                         expected.val);
    if (ret.val == old)
      break;
  }
  return ret.data;
}

template <typename T>
__device__ T atomicAggInc(T *ctr) {
  namespace cg = cooperative_groups;
  auto g = cg::coalesced_threads();
  T warp_res;
  if(g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + g.thread_rank();
}

}
}