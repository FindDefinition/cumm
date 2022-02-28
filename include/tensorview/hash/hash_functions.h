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
#include "hash_core.h"
#include <tensorview/core/all.h>

namespace tv {
namespace hash {

template <size_t Nbits>
struct Morton;

template<>
struct Morton<32>{
  static constexpr int kNumBits = 32;
  TV_HOST_DEVICE_INLINE static uint32_t encode(uint32_t x, uint32_t y, uint32_t z){
    return (split_by_3bits(z) << 2) | (split_by_3bits(y) << 1) | (split_by_3bits(x) << 0);
  }
  TV_HOST_DEVICE_INLINE static tv::array<uint32_t, 3> decode(uint32_t d){
    return {get_third_bits(d), get_third_bits(d << 1), get_third_bits(d << 2)};
  }
  template <uint32_t Axis>
  TV_HOST_DEVICE_INLINE static uint32_t decode_axis(uint32_t d){
    return get_third_bits(d << Axis);
  }
private: 
  TV_HOST_DEVICE_INLINE static uint32_t split_by_3bits(uint32_t a){
    uint32_t x = a & 0x000003ff; // we only look at the first 10 bits
    x = (x | x << 16) & 0x30000ff; 
    x = (x | x << 8) & 0x0300f00f;
    x = (x | x << 4) & 0x30c30c3; 
    x = (x | x << 2) & 0x9249249;
    return x;
  }
  TV_HOST_DEVICE_INLINE static uint32_t get_third_bits(uint32_t a){
    uint32_t x = a & 0x9249249; // we only look at the first 10 bits
		x = (x ^ (x >> 2)) & 0x30c30c3;
		x = (x ^ (x >> 4)) & 0x0300f00f;
		x = (x ^ (x >> 8)) & 0x30000ff;
		x = (x ^ (x >> 16)) & 0x000003ff;
    return x;
  }
};

template<>
struct Morton<64>{
  static constexpr int kNumBits = 64;
  TV_HOST_DEVICE_INLINE static uint64_t encode(uint32_t x, uint32_t y, uint32_t z){
    return (split_by_3bits(z) << 2) | (split_by_3bits(y) << 1) | (split_by_3bits(x) << 0);
  }
  TV_HOST_DEVICE_INLINE static tv::array<uint32_t, 3> decode(uint64_t d){
    return {get_third_bits(d), get_third_bits(d << 1), get_third_bits(d << 2)};
  }
  template <uint32_t Axis>
  TV_HOST_DEVICE_INLINE static uint32_t decode_axis(uint64_t d){
    return get_third_bits(d << Axis);
  }
private: 
  TV_HOST_DEVICE_INLINE static uint64_t split_by_3bits(uint32_t a){
    uint64_t x = a & 0x1fffff; // we only look at the first 10 bits
    x = (x | x << 32) & 0x1f00000000ffff; 
    x = (x | x << 16) & 0x1f0000ff0000ff; 
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3; 
    x = (x | x << 2) & 0x1249249249249249;
    return x;
  }
  TV_HOST_DEVICE_INLINE static uint32_t get_third_bits(uint64_t a){
    uint64_t x = a & 0x1249249249249249; // we only look at the first 21 bits
		x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
		x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
		x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
		x = (x ^ (x >> 16)) & 0x1f00000000ffff;
		x = (x ^ (x >> 32)) & 0x1fffff;
    return uint32_t(x);
  }
};

template <typename K> struct Murmur3Hash {
  using key_type = tv::hash::to_unsigned_t<K>;

  TV_HOST_DEVICE_INLINE static key_type hash(key_type k) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
  }
  TV_HOST_DEVICE_INLINE static key_type encode(key_type x) {
    return x;
  }
};

template <typename K> struct SpatialHash {
  using key_type = tv::hash::to_unsigned_t<K>;
  TV_HOST_DEVICE_INLINE static key_type hash(uint32_t x, uint32_t y, uint32_t z) {
    return x * 73856096u ^ y * 193649663u ^ z * 83492791u;
  }
  TV_HOST_DEVICE_INLINE static key_type encode(uint32_t x, uint32_t y, uint32_t z) {
    return Morton<key_type>::encode(x, y, z);
  }
};


template <typename K> struct IdentityHash {
  using key_type = tv::hash::to_unsigned_t<K>;

  TV_HOST_DEVICE_INLINE static key_type hash(key_type k) { return k; }
  TV_HOST_DEVICE_INLINE static key_type encode(key_type x) {
    return x;
  }
};

namespace detail {

template <typename T> struct FNVInternal;
template <> struct FNVInternal<uint32_t> {
  constexpr static uint32_t defaultOffsetBasis = 0x811C9DC5;
  constexpr static uint32_t prime = 0x01000193;
};

template <> struct FNVInternal<uint64_t> {
  constexpr static uint64_t defaultOffsetBasis = 0xcbf29ce484222325;
  constexpr static uint64_t prime = 0x100000001b3;
};

static constexpr bool kIsUint64SameAsULL =
    std::is_same<uint64_t, unsigned long long>::value;

template <>
struct FNVInternal<
    std::conditional<kIsUint64SameAsULL, int64_t, unsigned long long>::type> {
  constexpr static unsigned long long defaultOffsetBasis = 0xcbf29ce484222325;
  constexpr static unsigned long long prime = 0x100000001b3;
};

} // namespace detail

template <typename K>
struct FNV1aHash : detail::FNVInternal<tv::hash::to_unsigned_t<K>> {
  using key_type = tv::hash::to_unsigned_t<K>;
  TV_HOST_DEVICE_INLINE static key_type hash(key_type key) {
    key_type ret = detail::FNVInternal<key_type>::defaultOffsetBasis;
    key_type key_u = *(reinterpret_cast<key_type *>(&key));
    // const char* key_ptr = reinterpret_cast<const char*>(&key);
    TV_PRAGMA_UNROLL
    for (size_t i = 0; i < sizeof(K); ++i) {
      ret ^= (key_u >> (i * 8)) & 0xff;
      // ret ^= key_ptr[i];
      ret *= detail::FNVInternal<key_type>::prime;
    }
    return ret;
  }
  TV_HOST_DEVICE_INLINE static key_type encode(key_type x) {
    return x;
  }
};

} // namespace hash
} // namespace tv