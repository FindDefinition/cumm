#pragma once

#include "functional.h"
#include <limits>
namespace tv {

namespace math {

template <typename T, typename Tout, size_t N> struct Clamp {
  using argument_t = tv::array<T, N>;
  using result_t = tv::array<T, N>;
//   static constexpr T kClamp = T(std::numeric_limits<Tout>::max());
  TV_HOST_DEVICE_INLINE constexpr result_t operator()(const argument_t &src) {
    constexpr T kClamp = T((1U << (sizeof(Tout) * 8 - 1)) - 1);
    minimum<argument_t> min_op;
    maximum<argument_t> max_op;
    argument_t intermediate = max_op(src, -kClamp - T(1));
    intermediate = min_op(intermediate, kClamp);
    return intermediate;
  }
};

} // namespace math
} // namespace tv