#pragma once 
#include <tensorview/dtypes.h>
#include <cutlass/half.h>
#include <tensorview/core/all.h>
namespace tv {

namespace detail {

template <> struct TypeToDtype<cutlass::half_t> {
  static constexpr DType dtype = float16;
};

template <typename T, int N, bool R> struct get_array_extent<cutlass::Array<T, N, R>> {
  static constexpr int value = N;
};

}

}