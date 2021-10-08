#pragma once
#include "hash_core.h"
#include <tensorview/tensorview.h>

namespace tv {
namespace hash {

template <typename K> struct Murmur3Hash {
  using key_type = tv::hash::to_unsigned_t<K>;

  key_type TV_HOST_DEVICE_INLINE operator()(key_type k) const {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
  }
};

template <typename K> struct IdentityHash {
  using key_type = tv::hash::to_unsigned_t<K>;

  key_type TV_HOST_DEVICE_INLINE operator()(key_type k) const {
    return k;
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

template <> struct FNVInternal<unsigned long long> {
  constexpr static unsigned long long defaultOffsetBasis = 0xcbf29ce484222325;
  constexpr static unsigned long long prime = 0x100000001b3;
};

} // namespace detail

template <typename K> struct FNV1aHash : detail::FNVInternal<tv::hash::to_unsigned_t<K>> {
  using key_type = tv::hash::to_unsigned_t<K>;
  key_type TV_HOST_DEVICE_INLINE operator()(key_type key) const {
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
};

} // namespace hash
} // namespace tv