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
#ifndef __APPLE__

#include "nvrtc/core.h"
#include "nvrtc/type_traits.h"
#include "nvrtc/limits.h"
#include "nvrtc/tuple.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#else 
#pragma METAL internals : enable

#include <metal_stdlib>
namespace std = metal;
namespace metal {
template <typename> struct is_array : false_type {};

template <typename _Tp, std::size_t _Size>
struct is_array<_Tp[_Size]> : true_type {};

template <typename _Tp> struct is_array<_Tp[]> : true_type {};

/// remove_extent
template <typename _Tp> struct remove_extent {
  typedef _Tp type;
};

template <typename _Tp, std::size_t _Size> struct remove_extent<_Tp[_Size]> {
  typedef _Tp type;
};

template <typename _Tp> struct remove_extent<_Tp[]> {
  typedef _Tp type;
};


template <typename _Up, bool _IsArray = is_array<_Up>::value,
          bool _IsFunction = is_function<_Up>::value>
struct __decay_selector;

template <class T>
struct is_metal_constant : false_type {};
template <class T>
struct is_metal_constant<constant T> : true_type {};

// NB: DR 705.
template <typename _Up> struct __decay_selector<_Up, false, false> {
  typedef conditional_t<is_metal_constant<_Up>::value, const typename remove_cv<_Up>::type, typename remove_cv<_Up>::type> __type;
};

template <typename _Up> struct __decay_selector<_Up, true, false> {
  typedef typename remove_extent<_Up>::type device *__type;
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
namespace detail {
template <class T>
struct remove_metal_address_space_helper;

template <class T>
struct remove_metal_address_space_helper<constant T&> {
    typedef T type;
};
template <class T>
struct remove_metal_address_space_helper<thread T&> {
    typedef T type;
};

template <class T>
struct remove_metal_address_space_helper<threadgroup T&> {
    typedef T type;
};

template <class T>
struct remove_metal_address_space_helper<device T&> {
    typedef T type;
};


}
template <class T>
struct remove_metal_address_space {
    typedef conditional_t<is_function_v<T>, T, typename detail::remove_metal_address_space_helper<constant std::conditional_t<is_metal_constant<decay_t<T>>::value, const remove_reference_t<T>, remove_reference_t<T>>&>::type> type;
};
template <class T>
using remove_metal_address_space_t = typename remove_metal_address_space<T>::type;

// template <class T>
// inline constant T&& forward(constant typename remove_reference<T>::type& t) TV_NOEXCEPT_EXCEPT_METAL
// {
//     return static_cast<constant T&&>(t);
// }
template <class T>
inline thread T&& forward(thread typename remove_reference<T>::type& t) TV_NOEXCEPT_EXCEPT_METAL
{
    return static_cast<thread T&&>(t);
}
template <class T>
inline threadgroup T&& forward(threadgroup typename remove_reference<T>::type& t) TV_NOEXCEPT_EXCEPT_METAL
{
    return static_cast<threadgroup T&&>(t);
}
template <class T>
inline device T&& forward(device typename remove_reference<T>::type& t) TV_NOEXCEPT_EXCEPT_METAL
{
    return static_cast<device T&&>(t);
}

// template <class T>
// inline constant T&& forward(constant typename remove_reference<T>::type&& t) TV_NOEXCEPT_EXCEPT_METAL
// {
//     static_assert(!is_lvalue_reference<T>::value,
//                   "Can not forward an rvalue as an lvalue.");
//     return static_cast<constant T&&>(t);
// }

template <class T>
inline thread T&& forward(thread typename remove_reference<T>::type&& t) TV_NOEXCEPT_EXCEPT_METAL
{
    static_assert(!is_lvalue_reference<T>::value,
                  "Can not forward an rvalue as an lvalue.");
    return static_cast<thread T&&>(t);
}
template <class T>
inline threadgroup T&& forward(threadgroup typename remove_reference<T>::type&& t) TV_NOEXCEPT_EXCEPT_METAL
{
    static_assert(!is_lvalue_reference<T>::value,
                  "Can not forward an rvalue as an lvalue.");
    return static_cast<threadgroup T&&>(t);
}
template <class T>
inline device T&& forward(device typename remove_reference<T>::type&& t) TV_NOEXCEPT_EXCEPT_METAL
{
    static_assert(!is_lvalue_reference<T>::value,
                  "Can not forward an rvalue as an lvalue.");
    return static_cast<device T&&>(t);
}

#ifndef M_PI
#define M_PI M_PI_F // metal don't support double for now
#endif
/*
for non-array, non-function, non-point type T,
std::decay_t<T> is:
  const constant T& -> constant T
  constant T& -> constant T
  thread T& -> thread T
  ...

when we use regular template partial specification, the 
`constant T` is actually `const constant T`, which isn't equal to `constant T`

 */
}
#endif
