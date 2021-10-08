#pragma once
#include "array.h"
#include "defs.h"
#ifndef __CUDACC_RTC__
#include <cstdio>
#endif
#include "const_string.h"
namespace tv {

namespace detail {

template <typename T> struct type_to_format;

template <> struct type_to_format<int> {
  static constexpr auto value = make_const_string("%d");
};

template <> struct type_to_format<bool> {
  static constexpr auto value = make_const_string("%d");
};

template <> struct type_to_format<unsigned int> {
  static constexpr auto value = make_const_string("%u");
};

template <> struct type_to_format<long> {
  static constexpr auto value = make_const_string("%ld");
};

template <> struct type_to_format<unsigned long> {
  static constexpr auto value = make_const_string("%lu");
};

template <> struct type_to_format<long long> {
  static constexpr auto value = make_const_string("%lld");
};

template <> struct type_to_format<unsigned long long> {
  static constexpr auto value = make_const_string("%llu");
};

template <> struct type_to_format<float> {
  static constexpr auto value = make_const_string("%f");
};

template <> struct type_to_format<double> {
  static constexpr auto value = make_const_string("%lf");
};

template <> struct type_to_format<char *> {
  static constexpr auto value = make_const_string("%s");
};

template <> struct type_to_format<const char *> {
  static constexpr auto value = make_const_string("%s");
};

template <> struct type_to_format<char> {
  static constexpr auto value = make_const_string("%c");
};

template <> struct type_to_format<int8_t> {
  static constexpr auto value = make_const_string("%hd");
};

template <> struct type_to_format<int16_t> {
  static constexpr auto value = make_const_string("%hd");
};

template <> struct type_to_format<uint8_t> {
  static constexpr auto value = make_const_string("%hu");
};

template <> struct type_to_format<uint16_t> {
  static constexpr auto value = make_const_string("%hu");
};


template <class T, size_t N>
TV_HOST_DEVICE_INLINE constexpr auto generate_array_format(){
  // [%d, %d, %d, %d]
  auto elem_fmt = type_to_format<T>::value;
  int elem_fmt_size = elem_fmt.size();
  tv::const_string<2 + (type_to_format<T>::value).size() * N + 2 * (N - 1)> res{};
  res[0] = '[';
  res[res.size() - 1] = ']';
  for (int i = 0; i < N; ++i){
    for (int j = 1 + i * (elem_fmt_size + 2); j < 1 + i * (elem_fmt_size + 2) + elem_fmt_size; ++j){
      res[j] = elem_fmt[j - (1 + i * (elem_fmt_size + 2))];
    }
    if (i != N - 1){
      res[1 + i * (elem_fmt_size + 2) + elem_fmt_size] = ',';
      res[1 + i * (elem_fmt_size + 2) + elem_fmt_size + 1] = ' ';
    }
  }
  return res;
}

template <class T, size_t N> 
struct type_to_format<array<T, N>> {
  static constexpr auto value = generate_array_format<T, N>();
};


template <class T, size_t N> 
struct type_to_format<T[N]> {
  static constexpr auto value = generate_array_format<T, N>();
};


template <char Sep, class... Ts> struct types_to_format;

template <char Sep, class T> struct types_to_format<Sep, T> {
  static constexpr auto value = type_to_format<std::decay_t<T>>::value + "\n";
};

template <char Sep, class T, class... Ts>
struct types_to_format<Sep, T, Ts...> {
  static constexpr auto value = type_to_format<std::decay_t<T>>::value + const_string<1>(Sep) + types_to_format<Sep, Ts...>::value;
};


} // namespace detail

template <char Sep = ' ', class... Ts>
TV_HOST_DEVICE_INLINE void printf2(Ts... args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  static constexpr auto fmt = detail::types_to_format<Sep, Ts...>::value;
  printf(fmt.c_str(), args...);
}

template <char Sep = ' ', class... Ts>
TV_HOST_DEVICE_INLINE void printf2_once(Ts... args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  static constexpr auto fmt = detail::types_to_format<Sep, Ts...>::value;
#if defined(__CUDA_ARCH__)
  if ((threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0))
    printf(fmt.c_str(), args...);
#else 
  printf(fmt.c_str(), args...);
#endif
}

template <char Sep = ' ', class... Ts>
TV_HOST_DEVICE_INLINE void printf2_block_once(Ts... args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  static constexpr auto fmt = detail::types_to_format<Sep, Ts...>::value;
#if defined(__CUDA_ARCH__)
  if ((blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0))
    printf(fmt.c_str(), args...);
#else 
  printf(fmt.c_str(), args...);
#endif
}

template <char Sep = ' ', class... Ts>
TV_HOST_DEVICE_INLINE void printf2_thread_once(Ts... args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  static constexpr auto fmt = detail::types_to_format<Sep, Ts...>::value;
#if defined(__CUDA_ARCH__)
  if ((threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0))
    printf(fmt.c_str(), args...);
#else 
  printf(fmt.c_str(), args...);
#endif
}


namespace detail {

template <char Sep = ' ', typename T, size_t N, int... Indexes, class... Ts>
TV_HOST_DEVICE_INLINE void printf2_array_impl(array<T, N> arg, mp_list_int<Indexes...>, Ts&&... args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  static constexpr auto fmt = detail::types_to_format<Sep, array<T, N>, Ts...>::value;
  printf(fmt.c_str(), arg[Indexes]..., args...);
}

template <char Sep = ' ', typename T, size_t N, int... Indexes, class... Ts>
TV_HOST_DEVICE_INLINE void printf2_array_impl(T const (&arg)[ N ], mp_list_int<Indexes...>, Ts&&... args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  static constexpr auto fmt = detail::types_to_format<Sep, array<T, N>, Ts...>::value;
  printf(fmt.c_str(), arg[Indexes]..., args...);
}

}


template <char Sep = ' ', typename T, size_t N, class... Ts>
TV_HOST_DEVICE_INLINE void printf2_array(array<T, N> arg, Ts&&... args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  return detail::printf2_array_impl<Sep>(arg, mp_make_list_c_sequence<int, N>{}, std::forward<Ts>(args)...);

}

template <char Sep = ' ', typename T, size_t N, class... Ts>
TV_HOST_DEVICE_INLINE void printf2_array(T const (&arg)[ N ], Ts&&... args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
  return detail::printf2_array_impl<Sep>(arg, mp_make_list_c_sequence<int, N>{}, std::forward<Ts>(args)...);

}


template <char Sep = ' ', class... Ts>
TV_HOST_DEVICE_INLINE void printf2_array_once(Ts&&... args) {
#if defined(__CUDA_ARCH__)
  if ((threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0))
    printf2_array(args...);
#else 
  printf2_array(std::forward<Ts>(args)...);
#endif
}

template <char Sep = ' ', class... Ts>
TV_HOST_DEVICE_INLINE void printf2_array_block_once(Ts&&... args) {
  // this function should only be used for cuda code. host code
  // should use tv::ssprint.
#if defined(__CUDA_ARCH__)
  if ((blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0))
    printf2_array(args...);
#else 
  printf2_array(std::forward<Ts>(args)...);
#endif
}

} // namespace tv
