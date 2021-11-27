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

#include <tensorview/tensorview.h>
#include <type_traits>

namespace tv {
namespace hash {

namespace detail {

template <int N, bool Signed> struct SizeToInt;

template <bool Signed> struct SizeToInt<1, Signed> {
  using type = std::conditional_t<Signed, int8_t, uint8_t>;
};

template <bool Signed> struct SizeToInt<2, Signed> {
  using type = std::conditional_t<Signed, int16_t, uint16_t>;
};

template <bool Signed> struct SizeToInt<4, Signed> {
  using type = std::conditional_t<Signed, int32_t, uint32_t>;
};

template <bool Signed> struct SizeToInt<8, Signed> {
  using type = std::conditional_t<Signed, int64_t, unsigned long long>;
};

template <typename T> struct MapTypeToInt {
  using type = typename SizeToInt<sizeof(T), true>::type;
};

template <typename T> struct MapTypeToUnsignedInt {
  using type = typename SizeToInt<sizeof(T), false>::type;
};
} // namespace detail

template <int K>
using itemsize_to_unsigned_t = typename detail::SizeToInt<K, false>::type;

template <typename K>
using to_unsigned_t = typename detail::MapTypeToUnsignedInt<K>::type;


template <typename K>
constexpr to_unsigned_t<K> empty_key_v =
    std::numeric_limits<to_unsigned_t<K>>::max();

template <typename K, typename V, to_unsigned_t<K> EmptyKey = empty_key_v<K>> struct pair {
  K first;
  V second;
  using key_type_uint = to_unsigned_t<K>;
  static constexpr auto empty_key = EmptyKey;
  // static constexpr auto empty_key_uint =
  //     *(reinterpret_cast<const key_type_uint *>(&empty_key));
  bool TV_HOST_DEVICE_INLINE empty() { return first == empty_key; }
};

template <typename T> size_t align_to_power2(T size) {
  size_t r = 0;
  size_t num_1_bit = size & 1 ? 1 : 0;
  while (size >>= 1) {
    r++;
    if (size & 1) {
      ++num_1_bit;
    }
  }
  if (num_1_bit == 1) {
    return 1 << r;
  } else {
    return 1 << (r + 1);
  }
}

} // namespace hash

} // namespace tv
