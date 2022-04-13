#pragma once
#ifdef __CUDACC_RTC__
#include "type_traits.h"
#include <tensorview/core/defs.h>
#include <tensorview/core/mp_helper.h>

namespace std {
namespace detail {

template <size_t N, typename T> class head {
public:
  template <typename Targ>
  TV_HOST_DEVICE_INLINE constexpr explicit head(Targ &&num)
      : _value{std::forward<Targ>(num)} {}
  TV_HOST_DEVICE_INLINE constexpr explicit head():_value()  { }

  T _value;
};

template <typename Seq, typename... Ts> class tuple_impl {};

template <size_t... Ns, typename... Ts>
class tuple_impl<tv::mp_list_c<size_t, Ns...>, Ts...>
    : public head<Ns, Ts>... {
public:
  // FIXME if only one element in tuple, structure binding will raise error
  template <typename... Targs, typename = std::enable_if_t<sizeof...(Targs) == sizeof...(Ns)>>
  TV_HOST_DEVICE_INLINE constexpr tuple_impl(Targs &&...nums)
      : head<Ns, Ts>{std::forward<Targs>(nums)}... {}
    
   TV_HOST_DEVICE_INLINE constexpr tuple_impl()
    : head<Ns, Ts>{}... { }
};
} // namespace detail

template <typename... Ts>
using tuple =
    detail::tuple_impl<tv::mp_make_integer_sequence<size_t, sizeof...(Ts)>,
                         Ts...>;

template <size_t __i, class... Ts>
struct tuple_element<__i, tuple<Ts...>> {
  using type = typename tv::mp_nth_t<__i, Ts...>;
};

template <size_t N, typename... Ts>
TV_HOST_DEVICE_INLINE constexpr tuple_element_t<N, tuple<Ts...>> &
get(tuple<Ts...> &t) {
  using base = detail::head<N, tuple_element_t<N, tuple<Ts...>>>;
  return static_cast<base &>(t)._value;
}
template <size_t N, typename... Ts>
TV_HOST_DEVICE_INLINE constexpr const tuple_element_t<N, tuple<Ts...>> &
get(const tuple<Ts...> &t) {
  using base = detail::head<N, tuple_element_t<N, tuple<Ts...>>>;
  return static_cast<const base &>(t)._value;
}

template <size_t N, typename... Ts>
TV_HOST_DEVICE_INLINE constexpr tuple_element_t<N, tuple<Ts...>> &
get(tuple<Ts...> &&t) {
      typedef tuple_element_t<N, tuple<Ts...>> __element_type;
    return std::forward<__element_type&&>(get<N>(t));
}

template <size_t N, typename... Ts>
TV_HOST_DEVICE_INLINE constexpr const tuple_element_t<N, tuple<Ts...>> &
get(const tuple<Ts...> &&t) {
      typedef tuple_element_t<N, tuple<Ts...>> __element_type;
    return std::forward<const __element_type&&>(get<N>(t));
}

template <typename... _Elements>
struct tuple_size<tuple<_Elements...>>
    : public integral_constant<size_t, sizeof...(_Elements)> {};

template <typename _Tp>
constexpr size_t tuple_size_v = tuple_size<_Tp>::value;

template <typename... _Elements>
TV_HOST_DEVICE_INLINE constexpr tuple<
    typename __strip_reference_wrapper<_Elements>::__type...>
make_tuple(_Elements &&...__args) {
  typedef tuple<typename __strip_reference_wrapper<_Elements>::__type...>
      __result_type;
  return __result_type(std::forward<_Elements>(__args)...);
}

template <typename... _Elements>
TV_HOST_DEVICE_INLINE constexpr tuple<_Elements &...>
tie(_Elements &...__args) noexcept {
  return tuple<_Elements &...>(__args...);
}

} // namespace std
#endif
