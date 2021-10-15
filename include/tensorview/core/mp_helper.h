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

// disable weird warning when use alignas(0)

#ifndef MP_HELPER_H_
#define MP_HELPER_H_
#include "defs.h"

#ifdef __CUDACC_RTC__
#include "nvrtc_std.h"
#else
#include "cc17.h"
#include <functional>
#include <type_traits>
#include <utility>

#endif

namespace tv {

namespace detail {

template <std::size_t I, typename T, T N, T... Ns> struct mp_nth_c_impl {
  constexpr static T value = mp_nth_c_impl<I - 1, T, Ns...>::value;
};

template <typename T, T N, T... Ns> struct mp_nth_c_impl<0, T, N, Ns...> {
  constexpr static T value = N;
};

template <std::size_t I, typename T, class... Ts> struct mp_nth_impl {
  using type = typename mp_nth_impl<I - 1, Ts...>::type;
};

template <typename T, class... Ts> struct mp_nth_impl<0, T, Ts...> {
  using type = T;
};

template <std::size_t I, typename T> struct mp_at_impl;

template <std::size_t I, template <class...> class T, class... Args>
struct mp_at_impl<I, T<Args...>> {
  using type = typename mp_nth_impl<I, Args...>::type;
};

} // namespace detail

template <std::size_t I, typename T>
using mp_at = typename detail::mp_at_impl<I, T>::type;

template <std::size_t I, typename T>
constexpr auto mp_at_c = mp_at<I, T>::value;

template <std::size_t I, typename T, T... Ns>
constexpr T mp_nth_c = detail::mp_nth_c_impl<I, T, Ns...>::value;

template <std::size_t I, class... Ts>
using mp_nth_t = typename detail::mp_nth_impl<I, Ts...>::type;

template <class... T> struct mp_list {
  template <std::size_t I> using at = mp_nth_t<I, T...>;

  template <std::size_t I>
  static constexpr typename mp_nth_t<I, T...>::value_type at_c =
      mp_nth_t<I, T...>::value;

  static TV_HOST_DEVICE_INLINE constexpr size_t size() { return sizeof...(T); }
};

template <> struct mp_list<> {
  static TV_HOST_DEVICE_INLINE constexpr size_t size() { return 0; }
};

template <class T, T... I>
using mp_list_c = mp_list<std::integral_constant<T, I>...>;

template <int... I>
using mp_list_int = mp_list<std::integral_constant<int, I>...>;

namespace detail {
#ifndef __CUDACC_RTC__
template <class... Ts, class F>
constexpr F mp_for_each_impl(mp_list<Ts...>, F &&f) {
  using A = int[sizeof...(Ts)];
  return (void)A{((void)f(Ts()), 0)...}, std::forward<F>(f);

  // return (void)(std::initializer_list<int>{(f(Ts()), 0)...}),
  //        std::forward<F>(f);
}

template <class F>
constexpr F mp_for_each_impl(mp_list<>, F &&f) {
  return std::forward<F>(f);
}
#endif
} // namespace detail

template <class... T>
using mp_length = std::integral_constant<std::size_t, sizeof...(T)>;

namespace detail {

template <class A, template <class...> class B> struct mp_rename_impl {
  // An error "no type named 'type'" here means that the first argument to
  // mp_rename is not a list
};

template <template <class...> class A, class... T, template <class...> class B>
struct mp_rename_impl<A<T...>, B> {
  using type = B<T...>;
};

template <class A, template <class...> class B> struct mp_rename_v_impl {
  // An error "no type named 'type'" here means that the first argument to
  // mp_rename is not a list
};

template <template <class...> class A, class... T, template <class...> class B>
struct mp_rename_v_impl<A<T...>, B> {
  // unlike mp_rename_t, B is a templated type. we must use xx::value
  // because B can't be a templated non-type template parameter.
  static constexpr auto value = B<T...>::value;
};

template <class L, class T> struct mp_append_impl;

template <class... Ts, class T, template <class...> class L>
struct mp_append_impl<L<Ts...>, T> {
  using type = mp_list<Ts..., T>;
};

} // namespace detail

template <class L, class T>
using mp_append = typename detail::mp_append_impl<L, T>::type;

template <class A, template <class...> class B>
using mp_rename = typename detail::mp_rename_impl<A, B>::type;

template <class A, template <class...> class B>
constexpr auto mp_rename_v = detail::mp_rename_v_impl<A, B>::value;

template <class L> using mp_size = mp_rename<L, mp_length>;
#ifndef __CUDACC_RTC__
template <class L, class F>
constexpr F mp_for_each(F &&f) {
  return detail::mp_for_each_impl(mp_rename<L, mp_list>(), std::forward<F>(f));
}
#endif
template <unsigned N, unsigned... Ns> struct mp_prod_int {
  static constexpr unsigned value = N * mp_prod_int<Ns...>::value;
};

template <unsigned N> struct mp_prod_int<N> {
  static constexpr unsigned value = N;
};
namespace detail {

template <typename T, typename L> struct mp_append_impl;
template <typename T, class... Ts> struct mp_append_impl<T, mp_list<Ts...>> {
  using type = mp_list<Ts..., T>;
};

template <typename T, int N> struct mp_make_integer_sequence_impl {
  using type = typename mp_append_impl<
      typename mp_make_integer_sequence_impl<T, N - 1>::type,
      std::integral_constant<T, N - 1>>::type;
};

template <typename T> struct mp_make_integer_sequence_impl<T, 1> {
  using type = mp_list_c<T, 0>;
};

template <typename T> struct mp_make_integer_sequence_impl<T, 0> {
  using type = mp_list_c<T>;
};

template <typename T> struct mp_integer_sequence_impl;

template <class T, T... Ints>
struct mp_integer_sequence_impl<mp_list_c<T, Ints...>> {
  using type = mp_list<std::integral_constant<T, Ints>...>;
};
} // namespace detail

template <typename T, int N>
using mp_make_integer_sequence =
    typename detail::mp_make_integer_sequence_impl<T, N>::type;

template <class T, int N>
using mp_make_list_c_sequence = typename detail::mp_integer_sequence_impl<
    mp_make_integer_sequence<T, N>>::type;

namespace detail {}

namespace detail {

template <class T, T Start, T... Is>
constexpr auto TV_HOST_DEVICE_INLINE
mp_make_list_c_range_impl(mp_list_c<T, Is...> const &)
    -> decltype(mp_list_c<T, (Is + Start)...>{});

template <class T, T... Is>
constexpr auto TV_HOST_DEVICE_INLINE
mp_make_list_c_sequence_reverse_impl(mp_list_c<T, Is...> const &)
    -> decltype(mp_list_c<T, sizeof...(Is) - 1U - Is...>{});
} // namespace detail

template <int Start, int End>
using mp_list_int_range =
    decltype(detail::mp_make_list_c_range_impl<int, Start>(
        mp_make_list_c_sequence<int, (End - Start > 0 ? End - Start : 0)>{}));

template <class T, std::size_t N>
using mp_make_list_c_sequence_reverse =
    decltype(detail::mp_make_list_c_sequence_reverse_impl(
        mp_make_list_c_sequence<int, N>{}));

#ifndef __CUDACC_RTC__

namespace detail {

template <typename Ret, typename... Args>
std::integral_constant<size_t, sizeof...(Args)>
    func_argument_size_helper(Ret (*)(Args...));

template <typename Ret, typename F, typename... Args>
std::integral_constant<size_t, sizeof...(Args)>
    func_argument_size_helper(Ret (F::*)(Args...));

template <typename Ret, typename F, typename... Args>
std::integral_constant<size_t, sizeof...(Args)>
func_argument_size_helper(Ret (F::*)(Args...) const);

template <typename F>
decltype(func_argument_size_helper(&F::operator()))
    func_argument_size_helper(F);

template <typename Ret, typename Arg, typename... Rest>
Arg first_argument_helper(Ret (*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
Arg first_argument_helper(Ret (F::*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
Arg first_argument_helper(Ret (F::*)(Arg, Rest...) const);

template <typename Ret, typename Arg, typename... Rest>
Ret result_type_helper(Ret (*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
Ret result_type_helper(Ret (F::*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
Ret result_type_helper(Ret (F::*)(Arg, Rest...) const);

template <typename F>
decltype(first_argument_helper(&F::operator())) first_argument_helper(F);

template <typename F>
decltype(result_type_helper(&F::operator())) result_type_helper(F);

} // namespace detail

template <typename T>
using first_argument_t =
    decltype(detail::first_argument_helper(std::declval<T>()));
template <typename T>
using return_type_t = decltype(detail::result_type_helper(std::declval<T>()));

template <typename T>
constexpr size_t argument_size_v =
    decltype(detail::func_argument_size_helper(std::declval<T>()))::value;

#endif

} // namespace tv

#endif