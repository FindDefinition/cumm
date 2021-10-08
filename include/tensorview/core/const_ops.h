#pragma once
#include "defs.h"
namespace tv {
template <typename T> TV_HOST_DEVICE_INLINE constexpr T div_up(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T> TV_HOST_DEVICE_INLINE constexpr T align_up(T a, T b) {
  return div_up(a, b) * b;
}

} // namespace tv