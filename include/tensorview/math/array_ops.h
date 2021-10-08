#pragma once
#include <tensorview/core/all.h>

namespace tv {

namespace math {

template <typename T, size_t N> struct UnaryOp;

template <typename T, size_t N> struct UnaryIdentity {
  using argument_t = tv::array<T, N>;
  using result_t = tv::array<T, N>;

  TV_HOST_DEVICE_INLINE constexpr result_t operator()(const argument_t &src) {
      return src;
  }
};

} // namespace math
} // namespace tv
