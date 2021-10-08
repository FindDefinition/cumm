#pragma once

namespace tv {

namespace math {

template <typename T> constexpr T div_up(T a, T b) { return (a + b - 1) / b; }

template <typename T> constexpr T align_up(T a, T b) {
  return div_up(a, b) * b;
}

} // namespace math
} // namespace tv