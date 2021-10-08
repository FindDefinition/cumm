#pragma once 

namespace tv {

/// Defines the size of an element in bits - specialized for bin1_t

template <class T>
struct sizeof_bits {
  static constexpr auto value = sizeof(T) * 8;
};
using bin1_t = bool;

template <>
struct sizeof_bits<bin1_t> {
  static int const value = 1;
};


}
