// Copyright (C) 2008-2021 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.


#pragma once 
#ifdef __CUDACC_RTC__
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

namespace std {
typedef unsigned long size_t;
typedef long ptrdiff_t;


template <typename _Tp, _Tp __v> struct integral_constant {
  static constexpr _Tp value = __v;
  typedef _Tp value_type;
  typedef integral_constant<_Tp, __v> type;
  TV_HOST_DEVICE_INLINE constexpr operator value_type() const noexcept {
    return value;
  }
  TV_HOST_DEVICE_INLINE constexpr value_type operator()() const noexcept {
    return value;
  }
};
/// The type used as a compile-time boolean with true value.
typedef integral_constant<bool, true> true_type;

/// The type used as a compile-time boolean with false value.
typedef integral_constant<bool, false> false_type;

template <bool _Cond, typename _Iftrue, typename _Iffalse> struct conditional {
  typedef _Iftrue type;
};

// Partial specialization for false.
template <typename _Iftrue, typename _Iffalse>
struct conditional<false, _Iftrue, _Iffalse> {
  typedef _Iffalse type;
};

template <bool, typename, typename> struct conditional;

template <typename...> struct __or_;

template <> struct __or_<> : public false_type {};

template <typename _B1> struct __or_<_B1> : public _B1 {};

template <typename _B1, typename _B2>
struct __or_<_B1, _B2> : public conditional<_B1::value, _B1, _B2>::type {};

template <typename _B1, typename _B2, typename _B3, typename... _Bn>
struct __or_<_B1, _B2, _B3, _Bn...>
    : public conditional<_B1::value, _B1, __or_<_B2, _B3, _Bn...>>::type {};

template <typename...> struct __and_;

template <> struct __and_<> : public true_type {};

template <typename _B1> struct __and_<_B1> : public _B1 {};

template <typename _B1, typename _B2>
struct __and_<_B1, _B2> : public conditional<_B1::value, _B2, _B1>::type {};

template <typename _B1, typename _B2, typename _B3, typename... _Bn>
struct __and_<_B1, _B2, _B3, _Bn...>
    : public conditional<_B1::value, __and_<_B2, _B3, _Bn...>, _B1>::type {};

template <typename _Tp> struct remove_const { typedef _Tp type; };

template <typename _Tp> struct remove_const<_Tp const> { typedef _Tp type; };

template <typename _Tp> struct remove_volatile { typedef _Tp type; };

template <typename _Tp> struct remove_volatile<_Tp volatile> {
  typedef _Tp type;
};

template <typename _Tp> struct remove_cv {
  typedef typename remove_const<typename remove_volatile<_Tp>::type>::type type;
};

template <typename _Tp> struct remove_reference { typedef _Tp type; };

template <typename _Tp> struct remove_reference<_Tp &> { typedef _Tp type; };

template <typename _Tp> struct remove_reference<_Tp &&> { typedef _Tp type; };

/// remove_extent
template <typename _Tp> struct remove_extent { typedef _Tp type; };

template <typename _Tp, std::size_t _Size> struct remove_extent<_Tp[_Size]> {
  typedef _Tp type;
};

template <typename _Tp> struct remove_extent<_Tp[]> { typedef _Tp type; };

/// is_array
template <typename> struct is_array : public false_type {};

template <typename _Tp, std::size_t _Size>
struct is_array<_Tp[_Size]> : public true_type {};

template <typename _Tp> struct is_array<_Tp[]> : public true_type {};

template <typename> struct __is_pointer_helper : public false_type {};

template <typename _Tp> struct __is_pointer_helper<_Tp *> : public true_type {};

/// is_pointer
template <typename _Tp>
struct is_pointer
    : public __is_pointer_helper<typename remove_cv<_Tp>::type>::type {};
template <typename> struct is_function;

template <typename> struct is_function : public false_type {};

#define _GLIBCXX_NOEXCEPT_PARM , bool _NE
#define _GLIBCXX_NOEXCEPT_QUAL noexcept(_NE)

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) volatile _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) volatile & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) volatile && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) volatile _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) volatile & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) volatile && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const volatile _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const volatile & _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes...) const volatile && _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const volatile _GLIBCXX_NOEXCEPT_QUAL>
    : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const volatile &
                   _GLIBCXX_NOEXCEPT_QUAL> : public true_type {};

template <typename _Res, typename... _ArgTypes _GLIBCXX_NOEXCEPT_PARM>
struct is_function<_Res(_ArgTypes......) const volatile &&
                   _GLIBCXX_NOEXCEPT_QUAL> : public true_type {};

template <typename> struct is_lvalue_reference : public false_type {};

template <typename _Tp> struct is_lvalue_reference<_Tp &> : public true_type {};

template <typename _Up, bool _IsArray = is_array<_Up>::value,
          bool _IsFunction = is_function<_Up>::value>
struct __decay_selector;

// NB: DR 705.
template <typename _Up> struct __decay_selector<_Up, false, false> {
  typedef typename remove_cv<_Up>::type __type;
};

template <typename _Up> struct __decay_selector<_Up, true, false> {
  typedef typename remove_extent<_Up>::type *__type;
};

// add support for add_point need lots of code.
// template<typename _Up>
//   struct __decay_selector<_Up, false, true>
//   { typedef typename add_pointer<_Up>::type __type; };

/// decay
template <typename _Tp> class decay {
  typedef typename remove_reference<_Tp>::type __remove_type;

public:
  typedef typename __decay_selector<__remove_type>::__type type;
};

template <typename _Tp> using decay_t = typename decay<_Tp>::type;

template <typename _Tp>
constexpr _Tp &&
forward(typename std::remove_reference<_Tp>::type &__t) noexcept {
  return static_cast<_Tp &&>(__t);
}

/**
 *  @brief  Forward an rvalue.
 *  @return The parameter cast to the specified type.
 *
 *  This function is used to implement "perfect forwarding".
 */
template <typename _Tp>
constexpr _Tp &&
forward(typename std::remove_reference<_Tp>::type &&__t) noexcept {
  static_assert(!std::is_lvalue_reference<_Tp>::value,
                "template argument"
                " substituting _Tp is an lvalue reference type");
  return static_cast<_Tp &&>(__t);
}

template <typename T>
TV_HOST_DEVICE_INLINE constexpr const T &min(const T &a, const T &b) {
  if (b < a) {
    return b;
  }
  return a;
}

} // namespace std
#endif