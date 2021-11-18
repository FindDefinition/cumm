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
#include <cassert>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>
#endif

namespace tv {

template <typename T> constexpr int sizeof_bits_v = sizeof(T) * 8;

namespace detail {

template <typename T> struct sizeof_subbyte_impl {
  static_assert(sizeof_bits_v<T> % 8 == 0, "error");
  static constexpr int value = sizeof_bits_v<T> / 8;
};

} // namespace detail

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

} // namespace detail

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
  TV_HOST_DEVICE_INLINE constexpr reference front() noexcept { return *begin(); }

  TV_HOST_DEVICE_INLINE constexpr const_reference front() const noexcept { return array_[0]; }

  TV_HOST_DEVICE_INLINE constexpr reference back() noexcept { return N ? *(end() - 1) : *end(); }

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

  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> &
  operator+=(array<T, N, Align> const &other) {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      array_[i] += other[i];
    }
    return *this;
  }

  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> &
  operator-=(array<T, N, Align> const &other) {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      array_[i] -= other[i];
    }
    return *this;
  }

  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> &
  operator*=(array<T, N, Align> const &other) {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      array_[i] -= other[i];
    }
    return *this;
  }

  TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> &
  operator/=(array<T, N, Align> const &other) {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      array_[i] -= other[i];
    }
    return *this;
  }

  T array_[N];
};

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr bool operator==(const array<T, N, Align> &lfs, const array<T, N, Align> &rfs) {

  for (size_t i = 0; i < N; ++i) {
    if (lfs[i] != rfs[i])
      return false;
  }
  return true;
}

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr bool operator!=(const array<T, N, Align> &lfs, const array<T, N, Align> &rfs) {

  return !(lfs == rfs);
}


// TODO sub-byte type
template <typename T, size_t N, size_t Align = sizeof_v<T> * N>
struct alignas(Align) alignedarray : public array<T, N> {};

// template <typename T, size_t N, size_t Align = sizeof_v<T> * N>
// using alignedarray = array<T, N, Align>;


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

template <class F, class... Args>
TV_HOST_DEVICE_INLINE constexpr auto index_invoke(F f, int i, Args &&...args)
    -> decltype(f(args[i]...)) {
  return f(args[i]...);
}

template <class F, int... Is, class... Args>
TV_HOST_DEVICE_INLINE constexpr auto index_transform_impl(F f, seq<Is...>,
                                                          Args &&...args)
    -> array<decltype(f(args[0]...)), sizeof...(Is)> {
  return {{index_invoke(f, Is, std::forward<Args>(args)...)...}};
}

// we can't use std::extent here because the default value of extent is ZERO...
template <typename T> struct get_array_extent;

#ifndef __CUDACC_RTC__
// std::array don't support nvrtc.
template <typename T, size_t N> struct get_array_extent<std::array<T, N>> {
  static constexpr int value = N;
};
#endif

template <typename T, size_t N, size_t Align> struct get_array_extent<array<T, N, Align>> {
  static constexpr int value = N;
};

template <class T, class...>
struct get_extent_helper
    : std::integral_constant<int, get_array_extent<std::decay_t<T>>::value> {};

template <class F, class... Args>
TV_HOST_DEVICE_INLINE constexpr auto index_transform(F f, Args &&...args)
    -> decltype(index_transform_impl(f, gen_seq<get_extent_helper<Args...>{}>{},
                                     std::forward<Args>(args)...)) {
  using N = get_extent_helper<Args...>;
  return index_transform_impl(f, gen_seq<N{}>{}, std::forward<Args>(args)...);
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

} // namespace detail

template <class L>
constexpr auto mp_list_c_to_array =
    mp_rename_v<L, detail::mp_list_c_to_array_impl>;
template <class Array>
constexpr auto array_size_v = detail::get_array_extent<Array>::value;

template <int... Is>
constexpr auto mp_array_int_v = tv::mp_list_c_to_array<tv::mp_list_int<Is...>>;

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> operator+(const array<T, N, Align> &lfs,
                                                      const array<T, N, Align> &rfs) {
  return detail::index_transform(detail::array_sum<T>, lfs, rfs);
}

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> operator-(const array<T, N, Align> &lfs,
                                                      const array<T, N, Align> &rfs) {
  return detail::index_transform(detail::array_sub<T>, lfs, rfs);
}

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> operator*(const array<T, N, Align> &lfs,
                                                      const array<T, N, Align> &rfs) {
  return detail::index_transform(detail::array_mul<T>, lfs, rfs);
}

template <typename T, size_t N, size_t Align>
TV_HOST_DEVICE_INLINE constexpr array<T, N, Align> operator/(const array<T, N, Align> &lfs,
                                                      const array<T, N, Align> &rfs) {
  return detail::index_transform(detail::array_div<T>, lfs, rfs);
}

namespace arrayops {
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

template <typename T, size_t N, int... Inds>
TV_HOST_DEVICE_INLINE constexpr tv::array<
    T, sizeof...(Inds)>
slice_impl(const tv::array<T, N> &arr, mp_list_int<Inds...>) noexcept {
  return array<T, sizeof...(Inds)>{
      arr[Inds]...};
}

} // namespace detail
template <typename T, size_t N1, size_t N2>
TV_HOST_DEVICE_INLINE constexpr array<T, N1 + N2>
concat(const array<T, N1> &a, const array<T, N2> &b) noexcept {
  return detail::concat_impl(a, b, mp_make_list_c_sequence<int, N1>{},
                             mp_make_list_c_sequence<int, N2>{});
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr T prod(const array<T, N> &a) noexcept {
  // TODO use metaprogram instead of constexpr stmt
  T res = T(1);
  TV_PRAGMA_UNROLL
  for (int i = 0; i < N; ++i) {
    res *= a[i];
  }
  return res;
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr tv::array<T, N>
reverse(const array<T, N> &a) noexcept {
  return detail::reverse_impl(a, mp_make_list_c_sequence_reverse<int, N>{});
}

template <int Start, int End, typename T, size_t N>
TV_HOST_DEVICE_INLINE constexpr tv::array<
    T, (End > 0 ? End : End + int64_t(N)) - Start>
slice(const tv::array<T, N> &arr) {
  // TODO this function have problem in intellisense engine
  return detail::slice_impl(
      arr, mp_list_int_range<Start, (End > 0 ? End : End + int64_t(N))>{});
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

namespace detail {
#ifndef __CUDACC_RTC__
template <typename _InIter>
using _RequireInputIter = typename std::enable_if<std::is_convertible<
    typename std::iterator_traits<_InIter>::iterator_category,
    std::input_iterator_tag>::value>::type;
#endif
}
template <typename T, size_t N>
struct vecarray : public array<T, N> {
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
  TV_HOST_DEVICE_INLINE vecarray(){};
  TV_HOST_DEVICE_INLINE constexpr vecarray(size_t count, T init = T())
      : array<T, N>(), size_(count) {
    for (size_t i = 0; i < count; ++i) {
      array_[i] = init;
    }
  };
  constexpr TV_HOST_DEVICE vecarray(std::initializer_list<T> arr)
      : array<T, N>(), size_(std::min(N, arr.size())) {
    for (auto p = detail::make_init_pair(0, arr.begin());
         p.first < N && p.second != arr.end(); ++p.first, ++p.second) {
      array_[p.first] = *(p.second);
    }
  }
#ifndef __CUDACC_RTC__
  template <typename Iterator, typename = detail::_RequireInputIter<Iterator>>
  vecarray(Iterator first, Iterator last) {
    size_ = 0;
    for (; first != last; ++first) {
      if (size_ >= N) {
        continue;
      }
      array_[size_++] = *first;
    }
  };

  vecarray(const std::vector<T> &arr) {
    TV_ASSERT(arr.size() <= N);
    for (size_t i = 0; i < arr.size(); ++i) {
      array_[i] = arr[i];
    }
    size_ = arr.size();
  }
#endif
#ifdef TV_DEBUG
  TV_HOST_DEVICE_INLINE T &operator[](int idx) {
    TV_ASSERT(idx >= 0 && idx < size_);
    return array_[idx];
  }
  TV_HOST_DEVICE_INLINE const T &operator[](int idx) const {
    TV_ASSERT(idx >= 0 && idx < size_);
    return array_[idx];
  }
#else
  TV_HOST_DEVICE_INLINE constexpr T &operator[](int idx) { return array_[idx]; }
  TV_HOST_DEVICE_INLINE constexpr const T &operator[](int idx) const {
    return array_[idx];
  }
#endif

  TV_HOST_DEVICE_INLINE void push_back(T s) {
#ifdef TV_DEBUG
    TV_ASSERT(size_ < N);
#endif
    array_[size_++] = s;
  }
  TV_HOST_DEVICE_INLINE void pop_back() {
#ifdef TV_DEBUG
    TV_ASSERT(size_ > 0);
#endif
    size_--;
  }

  TV_HOST_DEVICE_INLINE size_t size() const { return size_; }
  TV_HOST_DEVICE_INLINE constexpr size_t max_size() const { return N; }

  TV_HOST_DEVICE_INLINE const T *data() const { return array_; }
  TV_HOST_DEVICE_INLINE T *data() { return array_; }
  TV_HOST_DEVICE_INLINE size_t empty() const { return size_ == 0; }

  TV_HOST_DEVICE_INLINE iterator begin() { return iterator(array_); }
  TV_HOST_DEVICE_INLINE iterator end() { return iterator(array_ + size_); }
  TV_HOST_DEVICE_INLINE constexpr const_iterator begin() const {
    return const_iterator(array_);
  }

  TV_HOST_DEVICE_INLINE constexpr const_iterator end() const {
    return const_iterator(array_ + size_);
  }
  TV_HOST_DEVICE_INLINE constexpr const_iterator cbegin() const {
    return const_iterator(array_);
  }

  TV_HOST_DEVICE_INLINE constexpr const_iterator cend() const {
    return const_iterator(array_ + size_);
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

  iterator erase(const_iterator CIit) {
    // Just cast away constness because this is a non-const member function.
    iterator Iit = const_cast<iterator>(CIit);

    assert(Iit >= this->begin() && "Iterator to erase is out of bounds.");
    assert(Iit < this->end() && "Erasing at past-the-end iterator.");

    iterator Nit = Iit;
    // Shift all elts down one.
    std::move(Iit + 1, this->end(), Iit);
    // Drop the last elt.
    this->pop_back();
    return (Nit);
  }

#endif
  TV_HOST_DEVICE_INLINE constexpr reference front() noexcept { return *begin(); }

  TV_HOST_DEVICE_INLINE constexpr const_reference front() const noexcept { return array_[0]; }

  TV_HOST_DEVICE_INLINE reference back() noexcept { return size_ ? *(end() - 1) : *end(); }

  TV_HOST_DEVICE_INLINE const_reference back() const noexcept {
    return size_ ? array_[size_ - 1] : array_[0];
  }

protected:
  T array_[N];
  size_t size_ = 0;
};

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE bool operator==(const vecarray<T, N> &lfs, const vecarray<T, N> &rfs) {
  if (lfs.size() != rfs.size())
    return false;
  for (size_t i = 0; i < lfs.size(); ++i) {
    if (lfs[i] != rfs[i])
      return false;
  }
  return true;
}

template <typename T, size_t N>
TV_HOST_DEVICE_INLINE bool operator!=(const vecarray<T, N> &lfs, const vecarray<T, N> &rfs) {

  return !(lfs == rfs);
}

} // namespace tv
#pragma GCC diagnostic pop