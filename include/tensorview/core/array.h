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

#pragma once
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wattributes"
#endif
#include <tensorview/core/defs.h>
#include <tensorview/core/mp_helper.h>
#ifdef TV_DEBUG
#include <tensorview/common.h>
#endif
#include "const_ops.h"

#ifdef __CUDACC_RTC__
#include "nvrtc_std.h"
#else
#include <array>
#include <cassert>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>
#endif

namespace tv {

template <class T> struct sizeof_bits {
  static constexpr auto value = sizeof(T) * 8;
};
template <> struct sizeof_bits<bool> { static int const value = 1; };

template <class T>
constexpr size_t sizeof_bits_v = sizeof_bits<std::decay_t<T>>::value;




namespace detail {

template <class T>
struct equivalent_data_type {
  using type = T;
};

template <typename T> struct sizeof_subbyte_impl {
  static_assert(sizeof_bits_v<T> % 8 == 0, "error");
  static constexpr int value = sizeof_bits_v<T> / 8;
};

} // namespace detail

template <class T>
using equivalent_data_type_t = typename detail::equivalent_data_type<T>::type;

template <typename T>
constexpr int sizeof_v = detail::sizeof_subbyte_impl<T>::value;

namespace detail {
template <typename K, typename V> struct InitPair {
  K first;
  V second;
};

template <typename K, typename V>
constexpr TV_HOST_DEVICE_INLINE InitPair<K, V> make_init_pair(K k, V v) {
  return InitPair<K, V>{k, v};
}

template <typename T> TV_HOST_DEVICE_INLINE constexpr T array_sum(T l, T r) {
  return l + r;
}
template <typename T> TV_HOST_DEVICE_INLINE constexpr T array_sub(T l, T r) {
  return l - r;
}
template <typename T> TV_HOST_DEVICE_INLINE constexpr T array_mul(T l, T r) {
  return l * r;
}
template <typename T> TV_HOST_DEVICE_INLINE constexpr T array_div(T l, T r) {
  return l / r;
}

template <typename T> TV_HOST_DEVICE_INLINE constexpr T &array_isum(T &l, T r) {
  return l += r;
}
template <typename T> TV_HOST_DEVICE_INLINE constexpr T &array_isub(T &l, T r) {
  return l -= r;
}
template <typename T> TV_HOST_DEVICE_INLINE constexpr T &array_imul(T &l, T r) {
  return l *= r;
}
template <typename T> TV_HOST_DEVICE_INLINE constexpr T &array_idiv(T &l, T r) {
  return l /= r;
}

template <typename T> TV_HOST_DEVICE_INLINE constexpr T array_minus(T x) {
  return -x;
}

template <typename T, typename TCast>
TV_HOST_DEVICE_INLINE constexpr TCast array_cast(T x) {
  return static_cast<TCast>(x);
}

} // namespace detail

namespace arrayops {

template <class F, class... Args>
TV_HOST_DEVICE_INLINE constexpr auto apply(F &&f, Args &&...args);
} // namespace arrayops

template <typename T, size_t N, size_t Align = 0> struct array {
  // TODO constexpr slice
  typedef T value_type;
  typedef value_type *pointer;
  typedef const value_type *const_pointer;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  typedef value_type *iterator;
  typedef const value_type *const_iterator;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
#ifndef __CUDACC_RTC__
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
#endif
public:
#ifdef TV_DEBUG
  TV_HOST_DEVICE_INLINE T &operator[](int idx) {
    TV_ASSERT(idx >= 0 && idx < N);
    return array_[idx];
  }
  TV_HOST_DEVICE_INLINE const T &operator[](int idx) const {
    TV_ASSERT(idx >= 0 && idx < N);
    return array_[idx];
  }
#else
  TV_HOST_DEVICE_INLINE constexpr reference operator[](int idx) {
    return array_[idx];
  }
  TV_HOST_DEVICE_INLINE constexpr const_reference operator[](int idx) const {
    return array_[idx];
  }
#endif
  // constexpr array(T(&a)[N]) : array_(a) {}
  TV_HOST_DEVICE_INLINE constexpr size_t size() const { return N; }

  TV_HOST_DEVICE_INLINE constexpr const T *data() const { return array_; }
  TV_HOST_DEVICE_INLINE constexpr T *data() { return array_; }

  // TV_HOST_DEVICE_INLINE constexpr const T[N] data_array() const { return
  // array_; } TV_HOST_DEVICE_INLINE constexpr T[N] data_array() { return
  // array_; }

  TV_HOST_DEVICE_INLINE constexpr size_t empty() const { return N == 0; }

  TV_HOST_DEVICE_INLINE constexpr iterator begin() { return iterator(array_); }
  TV_HOST_DEVICE_INLINE constexpr iterator end() {
    return iterator(array_ + N);
  }
  TV_HOST_DEVICE_INLINE constexpr const_iterator begin() const {
    return const_iterator(array_);
  }

  TV_HOST_DEVICE_INLINE constexpr const_iterator end() const {
    return const_iterator(array_ + N);
  }
  TV_HOST_DEVICE_INLINE constexpr const_iterator cbegin() const {
    return const_iterator(array_);
  }

  TV_HOST_DEVICE_INLINE constexpr const_iterator cend() const {
    return const_iterator(array_ + N);
  }
#ifndef __CUDACC_RTC__
  constexpr const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(end());
  }

  constexpr const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(begin());
  }

  constexpr reverse_iterator rbegin() noexcept {
    return reverse_iterator(end());
  }

  constexpr const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }

  constexpr reverse_iterator rend() noexcept {
    return reverse_iterator(begin());
  }

  constexpr const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }
#endif
  TV_HOST_DEVICE_INLINE constexpr reference front() noexcept {
    return *begin();
  }

  TV_HOST_DEVICE_INLINE constexpr const_reference front() const noexcept {
    return array_[0];
  }

  TV_HOST_DEVICE_INLINE constexpr reference back() noexcept {
    return N ? *(end() - 1) : *end();
  }

  TV_HOST_DEVICE_INLINE constexpr const_reference back() const noexcept {
    return N ? array_[N - 1] : array_[0];
  }

  TV_HOST_DEVICE_INLINE constexpr void clear() noexcept {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      array_[i] = T{};
    }
  }

  TV_HOST_DEVICE_INLINE constexpr void fill(const value_type &__u) {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      array_[i] = __u;
    }
  }

  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> &
  operator+=(TOther const &other) {
    arrayops::apply(detail::array_isum<T>, *this, other);
    return *this;
  }

  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> &
  operator-=(TOther const &other) {
    arrayops::apply(detail::array_isub<T>, *this, other);
    return *this;
  }

  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> &
  operator*=(TOther const &other) {
    arrayops::apply(detail::array_imul<T>, *this, other);
    return *this;
  }

  template <typename TOther>
  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> &
  operator/=(TOther const &other) {
    arrayops::apply(detail::array_idiv<T>, *this, other);
    return *this;
  }

  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> operator-() const {
    return arrayops::apply(detail::array_minus<T>, *this);
  }

  template <typename TCast>
  TV_HOST_DEVICE_INLINE constexpr array<TCast, N, Align> cast() const {
    return arrayops::apply(detail::array_cast<T, TCast>, *this);
  }

  template <template <class, size_t, size_t> class Op, class... Args>
  TV_HOST_DEVICE_INLINE constexpr auto op(Args &&...args) {
    return Op<T, N, Align>()(*this, std::forward<Args>(args)...);
  }

  template <template <class, size_t, size_t> class Op, class... Args>
  TV_HOST_DEVICE_INLINE constexpr auto op(Args &&...args) const {
    return Op<T, N, Align>()(*this, std::forward<Args>(args)...);
  }

  T array_[N];
};

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr bool operator==(const array<T, N, Align> &lfs,
                                                const array<T, N, Align> &rfs) {

  for (size_t i = 0; i < N; ++i) {
    if (lfs[i] != rfs[i])
      return false;
  }
  return true;
}

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr bool operator!=(const array<T, N, Align> &lfs,
                                                const array<T, N, Align> &rfs) {

  return !(lfs == rfs);
}

namespace detail {
// mp_list_c to array
template <class... Is> struct mp_list_c_to_array_impl {
  static constexpr array<typename mp_nth_t<0, Is...>::value_type, sizeof...(Is)>
      value{{Is::value...}};
};

// https://stackoverflow.com/questions/19936841/initialize-a-constexpr-array-as-sum-of-other-two-constexpr-arrays
template <int... Is> struct seq {};
template <int I, int... Is> struct gen_seq : gen_seq<I - 1, I - 1, Is...> {};
template <int... Is> struct gen_seq<0, Is...> : seq<Is...> {};

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr const T &
array_or_scalar(const array<T, N, Align> &arr, int i) {
  return arr[i];
}
#ifndef __CUDACC_RTC__
template <typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr const T &
array_or_scalar(const std::array<T, N> &arr, int i) {
  return arr[i];
}
#endif

template <typename T>
TV_HOST_DEVICE_INLINE constexpr const T &array_or_scalar(const T &arr, int i) {
  return arr;
}

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr T &array_or_scalar(array<T, N, Align> &arr,
                                                   int i) {
  return arr[i];
}

template <typename T>
TV_HOST_DEVICE_INLINE constexpr T &array_or_scalar(T &arr, int i) {
  return arr;
}

template <typename T> struct is_tv_array {
  static constexpr bool value = false;
};

template <typename T, size_t N, size_t Align>
struct is_tv_array<tv::array<T, N, Align>> {
  static constexpr bool value = true;
};

template <typename T> struct get_nested_element_type_impl {
  using type = std::decay_t<T>;
};

template <typename T, size_t N, size_t Align>
struct get_nested_element_type_impl<tv::array<T, N, Align>> {
  using type = typename get_nested_element_type_impl<std::decay_t<T>>::type;
};

template <typename T>
using get_nested_element_t = typename get_nested_element_type_impl<T>::type;

template <typename TCast, typename T> struct cast_nested_element_type_impl {
  using type = TCast;
};

template <typename TCast, typename T, size_t N, size_t Align>
struct cast_nested_element_type_impl<TCast, array<T, N, Align>> {
  using type = array<
      typename cast_nested_element_type_impl<TCast, std::decay_t<T>>::type, N>;
};

template <typename T, typename TCast>
using cast_nested_element_t =
    typename cast_nested_element_type_impl<TCast, T>::type;

template <class T>
struct get_tv_array_rank : public std::integral_constant<std::size_t, 0> {};

template <class T, size_t N, size_t Align>
struct get_tv_array_rank<array<T, N, Align>>
    : public std::integral_constant<std::size_t,
                                    get_tv_array_rank<T>::value + 1> {};

template <typename T> struct get_tv_array_shape { using type = mp_list<>; };

template <typename T, size_t N, size_t Align>
struct get_tv_array_shape<tv::array<T, N, Align>> {
  using type = mp_insert_front<typename get_tv_array_shape<T>::type,
                               std::integral_constant<std::size_t, N>>;
};

template <typename T> struct _get_tv_value_type { using type = void; };

template <typename T, size_t N, size_t Align>
struct _get_tv_value_type<tv::array<T, N, Align>> {
  using type = T;
};

template <typename TA, typename TB> struct determine_array_type {
  using __ta = typename std::decay<TA>::type;
  using __tb = typename std::decay<TB>::type;

  using type = typename std::conditional<
      is_tv_array<__ta>::value, __ta,
      typename std::conditional<is_tv_array<__tb>::value, __tb,
                                __tb>::type>::type;
};

template <bool Enable> struct invoke_or_recursive;

template <> struct invoke_or_recursive<true> {
  template <class F, class... Args>
  TV_HOST_DEVICE_INLINE static constexpr auto run(F &&f, int i,
                                                  Args &&...args) {
    return arrayops::apply(std::forward<F>(f),
                           array_or_scalar(std::forward<Args>(args), i)...);
  }
};

template <> struct invoke_or_recursive<false> {
  template <class F, class... Args>
  TV_HOST_DEVICE_INLINE static constexpr auto run(F &&f, int i,
                                                  Args &&...args) {
    return std::forward<F>(f)(array_or_scalar(std::forward<Args>(args), i)...);
  }
};

template <class F, int... Is, class... Args>
TV_HOST_DEVICE_INLINE constexpr auto index_transform_impl(F &&f, seq<Is...>,
                                                          Args &&...args)
    -> cast_nested_element_t<
        array<typename mp_reduce<determine_array_type, mp_list<Args...>,
                                 void>::value_type,
              sizeof...(Is)>,
        std::decay_t<return_type_t<F>>> {
  using arr_type_t = cast_nested_element_t<
      mp_reduce<determine_array_type, mp_list<Args...>, void>,
      std::decay_t<return_type_t<F>>>;
  static_assert(!std::is_same<arr_type_t, void>::value, "wtf");
  return {{invoke_or_recursive<(get_tv_array_rank<arr_type_t>::value > 1)>::run(
      std::forward<F>(f), Is, std::forward<Args>(args)...)...}};
}

// we can't use std::extent here because the default value of extent is ZERO...
template <typename T> struct get_array_extent {
  static constexpr int value = -1;
};

#ifndef __CUDACC_RTC__
// std::array don't support nvrtc.
template <typename T, size_t N> struct get_array_extent<std::array<T, N>> {
  static constexpr int value = N;
};
#endif

template <typename T, size_t N, size_t Align>
struct get_array_extent<array<T, N, Align>> {
  static constexpr int value = N;
};

template <class T> struct get_extent_helper_impl {
  using type =
      std::integral_constant<int, get_array_extent<std::decay_t<T>>::value>;
};
template <class T>
using get_extent_helper = typename get_extent_helper_impl<T>::type;

template <class... Ts>
constexpr int get_max_extent_v =
    mp_reduce_max<mp_transform<get_extent_helper, mp_list<Ts...>>,
                  std::integral_constant<int, -1>>::value;

template <size_t N> struct array_reduce_impl {
  static_assert(N != 0, "N can't equal to zero");
  template <typename F, typename T, size_t N1, size_t Align>
  TV_HOST_DEVICE_INLINE static constexpr T run(F &&f,
                                               const array<T, N1, Align> &arr) {
    return std::forward<F>(f)(
        array_reduce_impl<N - 1>::run(std::forward<F>(f), arr), arr[N - 1]);
  }
};

template <> struct array_reduce_impl<1> {
  template <typename F, typename T, size_t N1, size_t Align>
  TV_HOST_DEVICE_INLINE static constexpr T run(F &&f,
                                               const array<T, N1, Align> &arr) {
    return arr[0];
  }
};

} // namespace detail

template <typename T>
constexpr bool is_tv_array_v = detail::is_tv_array<std::decay_t<T>>::value;

namespace arrayops {

template <class F, class... Args>
TV_HOST_DEVICE_INLINE constexpr auto apply(F &&f, Args &&...args) {
  constexpr int N = detail::get_max_extent_v<Args...>;
  static_assert(N > 0, "error");
  return detail::index_transform_impl(std::forward<F>(f), detail::gen_seq<N>{},
                                      std::forward<Args>(args)...);
}

} // namespace arrayops
namespace detail {

template <typename T, size_t N, size_t... Ns> struct nested_array_type {
  using type = array<typename nested_array_type<T, Ns...>::type, N>;
};

template <typename T, size_t N> struct nested_array_type<T, N> {
  using type = array<T, N>;
};
} // namespace detail

template <typename T, size_t... Ns>
using array_nd = typename detail::nested_array_type<T, Ns...>::type;

#ifndef __CUDACC_RTC__
// std::array don't support nvrtc.
namespace detail {

template <typename T> struct nested_conversion {
  using type = std::decay_t<T>;
};

template <typename T, size_t N> struct nested_conversion<std::array<T, N>> {
  using type = tv::array<typename nested_conversion<std::decay_t<T>>::type, N>;
};

template <typename T, size_t N> struct nested_conversion<tv::array<T, N>> {
  using type = std::array<typename nested_conversion<std::decay_t<T>>::type, N>;
};

template <typename T>
using nested_conversion_t = typename nested_conversion<std::decay_t<T>>::type;

template <typename T, size_t N>
constexpr nested_conversion_t<std::array<T, N>>
from_std_array_impl_expand(const std::array<T, N> &arr);
template <typename T> constexpr T from_std_array_impl_expand(const T &arr) {
  return arr;
}

template <typename T, size_t N>
constexpr nested_conversion_t<tv::array<T, N>>
to_std_array_impl_expand(const tv::array<T, N> &arr);
template <typename T> constexpr T to_std_array_impl_expand(const T &arr) {
  return arr;
}

template <typename T, size_t N, int... Inds>
constexpr auto from_std_array_impl(const std::array<T, N> &arr,
                                   mp_list_int<Inds...>) {
  return nested_conversion_t<std::array<T, N>>{
      from_std_array_impl_expand(arr[Inds])...};
}

template <typename T, size_t N>
constexpr nested_conversion_t<std::array<T, N>>
from_std_array_impl_expand(const std::array<T, N> &arr) {
  return from_std_array_impl(arr, mp_make_list_c_sequence<int, N>{});
}

template <typename T, size_t N, int... Inds>
constexpr auto to_std_array_impl(const tv::array<T, N> &arr,
                                 mp_list_int<Inds...>) {
  return nested_conversion_t<tv::array<T, N>>{
      to_std_array_impl_expand(arr[Inds])...};
}

template <typename T, size_t N>
constexpr nested_conversion_t<tv::array<T, N>>
to_std_array_impl_expand(const tv::array<T, N> &arr) {
  return to_std_array_impl(arr, mp_make_list_c_sequence<int, N>{});
}

} // namespace detail

namespace arrayops {
template <typename T, size_t N>
constexpr detail::nested_conversion_t<std::array<T, N>>
from_std_array(const std::array<T, N> &arr) {
  return detail::from_std_array_impl_expand(arr);
}
template <typename T, size_t N>
constexpr detail::nested_conversion_t<tv::array<T, N>>
to_std_array(const tv::array<T, N> &arr) {
  return detail::to_std_array_impl_expand(arr);
}
} // namespace arrayops
#endif

template <class L>
constexpr auto mp_list_c_to_array =
    mp_rename_v<L, detail::mp_list_c_to_array_impl>;
template <class Array>
constexpr auto array_size_v = detail::get_array_extent<Array>::value;

template <int... Is>
constexpr auto mp_array_int_v = tv::mp_list_c_to_array<tv::mp_list_int<Is...>>;

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator+(const array<T, N, Align> &lfs, const array<T, N, Align> &rfs) {
  return arrayops::apply(detail::array_sum<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator+(const array<T, N, Align> &lfs, const T2 &rfs) {
  return arrayops::apply(detail::array_sum<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator+(const T2 &lfs, const array<T, N, Align> &rfs) {
  return arrayops::apply(detail::array_sum<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator-(const array<T, N, Align> &lfs, const array<T, N, Align> &rfs) {
  return arrayops::apply(detail::array_sub<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator-(const array<T, N, Align> &lfs, const T2 &rfs) {
  return arrayops::apply(detail::array_sub<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator-(const T2 &lfs, const array<T, N, Align> &rfs) {
  return arrayops::apply(detail::array_sub<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator*(const array<T, N, Align> &lfs, const array<T, N, Align> &rfs) {
  return arrayops::apply(detail::array_mul<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator*(const array<T, N, Align> &lfs, const T2 &rfs) {
  return arrayops::apply(detail::array_mul<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator*(const T2 &lfs, const array<T, N, Align> &rfs) {
  return arrayops::apply(detail::array_mul<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator/(const array<T, N, Align> &lfs, const array<T, N, Align> &rfs) {
  return arrayops::apply(detail::array_div<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator/(const array<T, N, Align> &lfs, const T2 &rfs) {
  return arrayops::apply(detail::array_div<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

template <typename T, size_t N, size_t Align, typename T2,
          typename = std::enable_if_t<!is_tv_array_v<T2>>>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
operator/(const T2 &lfs, const array<T, N, Align> &rfs) {
  return arrayops::apply(detail::array_div<detail::get_nested_element_t<T>>,
                         lfs, rfs);
}

// TODO sub-byte type
template <typename T, size_t N, size_t Align = sizeof_v<T> *N>
struct alignas(Align) alignedarray : public array<T, N> {};

namespace detail {
template <typename T, size_t N1, size_t N2, int... IndsL, int... IndsR>
TV_HOST_DEVICE_INLINE constexpr array<T, N1 + N2>
concat_impl(const array<T, N1> &a, const array<T, N2> &b, mp_list_int<IndsL...>,
            mp_list_int<IndsR...>) noexcept {
  return array<T, N1 + N2>{a[IndsL]..., b[IndsR]...};
}

template <typename T, size_t N, int... Inds>
TV_HOST_DEVICE_INLINE constexpr array<T, N>
reverse_impl(const array<T, N> &a, mp_list_int<Inds...>) noexcept {
  return array<T, N>{a[Inds]...};
}

template <typename T, size_t N, size_t Align, int... Inds>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align>
constant_impl(T val, mp_list_int<Inds...>) noexcept {
  return array<T, N, Align>{(Inds, val)...};
}

template <typename T, size_t N, int... Inds>
TV_HOST_DEVICE_INLINE constexpr array<T, sizeof...(Inds)>
slice_impl(const array<T, N> &arr, mp_list_int<Inds...>) noexcept {
  return array<T, sizeof...(Inds)>{arr[Inds]...};
}

template <int Start, int End, typename T, size_t N1, size_t N2, int... Inds>
TV_HOST_DEVICE_INLINE constexpr array<array<T, End - Start>, sizeof...(Inds)>
slice_2d_impl(const array<array<T, N2>, N1> &arr, mp_list_int<Inds...>) noexcept {
  return array<array<T, End - Start>, sizeof...(Inds)>{slice_impl(arr[Inds], mp_list_int_range<Start, End>{})...};
}


template <typename T, size_t N1, size_t N2>
TV_HOST_DEVICE_INLINE constexpr array<T, N1 + N2>
concat2_base_impl(const array<T, N1> &a, const array<T, N2> &b) noexcept {
  return concat_impl(a, b, mp_make_list_c_sequence<int, N1>{},
                     mp_make_list_c_sequence<int, N2>{});
}

} // namespace detail

namespace arrayops {

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr array<T, N>
concat(const array<T, N> &arr) noexcept {
  return arr;
}

template <typename T, size_t N, size_t... Ns>
TV_HOST_DEVICE_INLINE constexpr array<T, mp_reduce_sum_v<size_t, N, Ns...>>
concat(const array<T, N> &arr, const array<T, Ns> &...arrs) noexcept {
  return detail::concat2_base_impl(arr, concat(arrs...));
}

template <typename F, typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr T reduce(F &&f,
                                         const array<T, N, Align> &a) noexcept {
  return detail::array_reduce_impl<N>::run(std::forward<F>(f), a);
}

template <typename T, size_t N, size_t Align = 0>
TV_HOST_DEVICE_INLINE constexpr auto constant(T val) noexcept {
  return detail::constant_impl<T, N, Align>(
      val, mp_make_list_c_sequence_reverse<int, N>{});
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr array<T, N>
reverse(const array<T, N> &a) noexcept {
  return detail::reverse_impl(a, mp_make_list_c_sequence_reverse<int, N>{});
}
template <int Start, int End, typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr array<
    T, (End > 0 ? End : End + int64_t(N)) - Start>
slice(const array<T, N> &arr) {
  // TODO this function have problem in intellisense engine
  return detail::slice_impl(
      arr, mp_list_int_range<Start, (End > 0 ? End : End + int64_t(N))>{});
}

template <int Start1, int Start2, int End1, int End2, typename T, size_t N1, size_t N2>
TV_HOST_DEVICE_INLINE constexpr array<
    array<T, (End2 > 0 ? End2 : End2 + int64_t(N2)) - Start2>, (End1 > 0 ? End1 : End1 + int64_t(N1)) - Start1>
slice_2d(const array<array<T, N2>, N1> &arr) {
  return detail::slice_2d_impl<Start2, (End2 > 0 ? End2 : End2 + int64_t(N2))>(
      arr, mp_list_int_range<Start1, (End1 > 0 ? End1 : End1 + int64_t(N1))>{});
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr bool contains(T val,
                                              const array<T, N> &a) noexcept {
  bool res = false;
  TV_PRAGMA_UNROLL
  for (int i = 0; i < N; ++i) {
    if (a[i] == val) {
      res = true;
    }
  }
  return res;
}

} // namespace arrayops

template <typename T, size_t N1, size_t N2>
TV_HOST_DEVICE_INLINE constexpr array<T, N1 + N2>
operator|(const array<T, N1> &lfs, const array<T, N2> &rfs) {
  return arrayops::concat(lfs, rfs);
}

} // namespace tv
#pragma GCC diagnostic pop