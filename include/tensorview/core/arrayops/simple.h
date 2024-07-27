#pragma once

#include "mathbase.h"
#include <tensorview/core/array.h>

namespace tv {
namespace arrayops {

template <size_t N, typename T>
TV_HOST_DEVICE_INLINE tv::array<T, N> read_ptr(TV_METAL_CONSTANT const T *ptr) {
  return reinterpret_cast<const TV_METAL_CONSTANT tv::array<T, N> *>(ptr)[0];
}
template <typename T, size_t... Ns>
TV_HOST_DEVICE_INLINE TV_METAL_CONSTANT tv::array_nd<T, Ns...>* reinterpret_array_nd_cast(TV_METAL_CONSTANT T *ptr) {
  return reinterpret_cast<TV_METAL_CONSTANT tv::array_nd<T, Ns...> *>(ptr);
}

#ifdef TV_METAL_RTC
template <size_t N, typename T>
TV_HOST_DEVICE_INLINE tv::array<T, N> read_ptr(const thread T *ptr) {
  return reinterpret_cast<const thread tv::array<T, N> *>(ptr)[0];
}
template <size_t N, typename T>
TV_HOST_DEVICE_INLINE tv::array<T, N> read_ptr(const device T *ptr) {
  return reinterpret_cast<const device tv::array<T, N> *>(ptr)[0];
}

template <size_t N, typename T>
TV_HOST_DEVICE_INLINE tv::array<T, N> read_ptr(const threadgroup T *ptr) {
  return reinterpret_cast<const threadgroup tv::array<T, N> *>(ptr)[0];
}

template <typename T, size_t... Ns>
TV_HOST_DEVICE_INLINE thread tv::array_nd<T, Ns...>* reinterpret_array_nd_cast(thread T *ptr) {
  return reinterpret_cast<thread tv::array_nd<T, Ns...> *>(ptr);
}

template <typename T, size_t... Ns>
TV_HOST_DEVICE_INLINE device tv::array_nd<T, Ns...>* reinterpret_array_nd_cast(device T *ptr) {
  return reinterpret_cast<device tv::array_nd<T, Ns...> *>(ptr);
}

template <typename T, size_t... Ns>
TV_HOST_DEVICE_INLINE threadgroup tv::array_nd<T, Ns...>* reinterpret_array_nd_cast(threadgroup T *ptr) {
  return reinterpret_cast<threadgroup tv::array_nd<T, Ns...> *>(ptr);
}

#endif

template <class... Ts>
TV_HOST_DEVICE_INLINE constexpr auto create_array(Ts... vals) {
  return array<mp_nth_t<0, Ts...>, sizeof...(Ts)>{vals...};
}

template <typename T, size_t N, size_t Align> struct max {
  // TODO why we can't use std::max?
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self, TOther other) {
    return apply(max_impl, self, other);
  }

  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return reduce(max_impl, self);
  }

private:
  TV_HOST_DEVICE_INLINE static constexpr const TV_METAL_THREAD T &max_impl(const TV_METAL_THREAD T &a,
                                                           const TV_METAL_THREAD T &b) {
    return (a < b) ? b : a;
  }
};

template <typename T, size_t N, size_t Align> struct min {
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self, TOther other) {
    return apply(min_impl, self, other);
  }

  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return reduce(min_impl, self);
  }

private:
  TV_HOST_DEVICE_INLINE static constexpr const TV_METAL_THREAD T &min_impl(const TV_METAL_THREAD T &a,
                                                           const TV_METAL_THREAD T &b) {
    return (b < a) ? b : a;
  }
};

template <typename T, size_t N, size_t Align> struct abs {
  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return apply(abs_impl, self);
  }

private:
  TV_HOST_DEVICE_INLINE static constexpr T abs_impl(const TV_METAL_THREAD T &a) {
    return a >= T(0) ? a : -a;
  }
};

template <typename T, size_t N, size_t Align> struct sum {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return reduce(detail::array_sum<T>, self);
  }
};

template <typename T, size_t N, size_t Align> struct mean {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return reduce(detail::array_sum<T>, self) / T(N);
  }
};

template <typename T, size_t N, size_t Align> struct prod {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return reduce(detail::array_mul<T>, self);
  }
};

template <typename T, size_t N, size_t Align> struct dot {
  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self, TOther other) {
    return (self * other).template op<sum>();
  }
};

template <typename T, size_t N, size_t Align> struct l2norm {
  TV_HOST_DEVICE_INLINE auto operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return MathScalarOp<T>::sqrt(self.template op<dot>(self));
  }
};

template <typename T, size_t N, size_t Align> struct length {
  TV_HOST_DEVICE_INLINE constexpr auto operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return MathScalarOp<T>::sqrt(self.template op<dot>(self));
  }
};

template <typename T, size_t N, size_t Align> struct length2 {
  TV_HOST_DEVICE_INLINE auto operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return self.template op<dot>(self);
  }
};

template <typename T, size_t N, size_t Align> struct normalize {
  TV_HOST_DEVICE_INLINE auto operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return self / self.template op<l2norm>();
  }
};

template <typename T, size_t N, size_t Align> struct cross {
  TV_HOST_DEVICE_INLINE constexpr auto operator()(const TV_METAL_THREAD array<T, 2, Align> &a,
                                                  const TV_METAL_THREAD array<T, 2, Align> &b) {
    return a[0] * b[1] - a[1] * b[0];
  }
  TV_HOST_DEVICE_INLINE constexpr array<T, 2, Align>
  operator()(const TV_METAL_THREAD array<T, 2, Align> &a, const TV_METAL_THREAD T &b) {
    return {a[1] * b, -a[0] * b};
  }
  TV_HOST_DEVICE_INLINE constexpr array<T, 2, Align>
  operator()(const TV_METAL_THREAD T &a, const TV_METAL_THREAD array<T, 2, Align> &b) {
    return {-a * b[1], a * b[0]};
  }
  TV_HOST_DEVICE_INLINE constexpr array<T, 3, Align>
  operator()(const TV_METAL_THREAD array<T, 3, Align> &a, const TV_METAL_THREAD array<T, 3, Align> &b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
  }
};

#ifndef TV_METAL_RTC
template <typename T, size_t N, size_t Align> struct zeros {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return array<T, N, Align>{};
  }
};

template <typename T, size_t N, size_t Align> struct ones {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self) {
    return constant<T, N, Align>(T(1));
  }
};

template <typename T, size_t N, size_t Align> struct full {
  TV_HOST_DEVICE_INLINE constexpr auto
  operator()(const TV_METAL_THREAD array<T, N, Align> &self, T val) {
    return constant<T, N, Align>(val);
  }
};
#endif

// constexpr tv::array<float, 3> a{1, 2, 3};
// constexpr tv::array<float, 3> aa{4, 5, 6};
// template<class T>
// TV_HOST_DEVICE_INLINE constexpr const T& maxX(const T& a, const T& b)
// {
//     return (a < b) ? b : a;
// }

// constexpr auto b = (a * aa).op<arrayops::min>(2);
// constexpr auto c = (a * aa).cast<int>().op<max>(0).op<min>(1).cast<float>() +
// 0.5f; constexpr auto d = c / aa; using WTF =
// std::decay_t<return_type_t<decltype(maxX<float>)>>;
} // namespace arrayops
} // namespace tv