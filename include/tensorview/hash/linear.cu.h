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
#include "hash.cu.h"
#include "hash_functions.h"
#include <tensorview/core/defs.h>

namespace tv {

namespace hash {

template <typename K, typename V, typename Hash = tv::hash::Murmur3Hash<K>, K EmptyKey = default_empty_key_v<K>,
          bool Power2 = false>
struct LinearHashTable {
public:
  using key_type_uint = to_unsigned_t<K>;
  using key_type = K;
  using value_type = pair<K, V, EmptyKey>;
  using mapped_type = V;
  using size_type = int32_t;
  using hash_type = Hash;
  // TODO no way to get constexpr uint repr of key for now.
  // static constexpr auto empty_key_uint = value_type::empty_key_uint;
  static constexpr auto empty_key = EmptyKey;

private:
  value_type *table_ = nullptr;
  size_type hash_size_;
  key_type_uint empty_key_uint;
  static constexpr auto kNumHashArgs = argument_size_v<decltype(hash_type::hash)>;

public:
  TV_HOST_DEVICE_INLINE explicit LinearHashTable(value_type *table,
                                                 size_type hash_size)
      : table_(table), hash_size_(hash_size) {
    auto eky = empty_key;
    empty_key_uint = *(reinterpret_cast<const key_type_uint *>(&eky));
  }

  template <class... Args>
  TV_DEVICE_INLINE key_type_uint hash(Args &&...keys) const {
    return hash_type::hash(keys...);
  }

  TV_DEVICE_INLINE int probe_length(int index, key_type key) {
    return (index - hash_type::hash_scalar(reinterpret_cast<key_type_uint&>(key)) % hash_size_);
  }

  TV_HOST_DEVICE_INLINE size_type size() const { return hash_size_; }

  TV_HOST_DEVICE_INLINE value_type *data() { return table_; }

  TV_HOST_DEVICE_INLINE const value_type *data() const { return table_; }

private:
  template <class... Args>
  TV_DEVICE_INLINE void insert_raw(const V &value, Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<const key_type_uint &>(std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(keys...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    while (true) {
      key_type_uint prev =
          atomicCAS(reinterpret_cast<key_type_uint *>(&table_[slot].first),
                    empty_key_uint, key_u);
      if (prev == empty_key_uint || prev == key_u) {
        table_[slot].second = value;
        break;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
  }

  template <class F, class... Args>
  TV_DEVICE_INLINE void insert_raw_custom_value(F&& f, Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<const key_type_uint &>(std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(keys...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    while (true) {
      key_type_uint prev =
          atomicCAS(reinterpret_cast<key_type_uint *>(&table_[slot].first),
                    empty_key_uint, key_u);
      if (prev == empty_key_uint || prev == key_u) {
        std::forward<F>(f)(&table_[slot].second);
        break;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
  }

  template <int... Inds>
  TV_DEVICE_INLINE void insert_raw(const array<K, kNumHashArgs>& key_arr, const V &value, mp_list_int<Inds...>) {
    return insert_raw(value, key_arr[Inds]...);
  }

  template <class F, int... Inds>
  TV_DEVICE_INLINE void insert_raw_custom_value(const array<K, kNumHashArgs>& key_arr, F&& f, mp_list_int<Inds...>) {
    return insert_raw_custom_value(std::forward<F>(f), key_arr[Inds]...);
  }

public:
  TV_DEVICE_INLINE void insert(const K& key, const V &value) {
    static_assert(kNumHashArgs == 1, "you must use tv::array if hash multiple values.");
    return insert_raw(value, key);
  }

  TV_DEVICE_INLINE void insert(const array<K, kNumHashArgs>& key_arr, const V &value) {
    return insert_raw(key_arr, value, mp_make_list_c_sequence<int, kNumHashArgs>{});
  }

  template <class F>
  TV_DEVICE_INLINE void insert_custom_value(const K& key, F&& f) {
    static_assert(kNumHashArgs == 1, "you must use tv::array if hash multiple values.");
    return insert_raw_custom_value(std::forward<F>(f), key);
  }

  template <class F>
  TV_DEVICE_INLINE void insert_custom_value(const array<K, kNumHashArgs>& key_arr, F&& f) {
    return insert_raw_custom_value(key_arr, std::forward<F>(f), mp_make_list_c_sequence<int, kNumHashArgs>{});
  }

  template <class... Args> TV_DEVICE_INLINE value_type lookup(Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<const key_type_uint &>(std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(keys...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    const value_type *table_ptr = table_;
    while (true) {
      auto val = table_ptr[slot];
      if (reinterpret_cast<key_type_uint &>(val.first) == key_u) {
        return val;
      }
      if (val.first == empty_key) {
        return {empty_key, V()};
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
    return {empty_key, V()};
  }

  template <class... Args>
  TV_DEVICE_INLINE size_type lookup_offset(Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<const key_type_uint &>(std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(keys...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    value_type val;
    while (true) {
      val = table_[slot];
      if (reinterpret_cast<key_type_uint &>(val.first) == key_u) {
        return slot;
      }
      if (val.first == empty_key) {
        return -1;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
    return -1;
  }
};

template <typename K, typename Hash = tv::hash::Murmur3Hash<K>, K EmptyKey = default_empty_key_v<K>,
          bool Power2 = false>
struct LinearHashSet {
public:
  using key_type_uint = to_unsigned_t<K>;
  using value_type = K;
  using size_type = int32_t;
  static constexpr auto empty_key = EmptyKey;
  using hash_type = Hash;

private:
  value_type *table_ = nullptr;
  size_type hash_size_;
  key_type_uint empty_key_uint;
  static constexpr auto kNumHashArgs = argument_size_v<decltype(hash_type::hash)>;

public:
  TV_HOST_DEVICE_INLINE explicit LinearHashSet(value_type *table,
                                               size_type hash_size)
      : table_(table), hash_size_(hash_size) {
    auto eky = empty_key;
    empty_key_uint = *(reinterpret_cast<const key_type_uint *>(&eky));
  }

  template <class... Args>
  TV_DEVICE_INLINE key_type_uint hash(Args &&...keys) const {
    return hash_type::hash(keys...);
  }
  TV_DEVICE_INLINE int probe_length(int index, value_type key) {
    return (index - hash_type::hash_scalar(reinterpret_cast<key_type_uint&>(key)) % hash_size_);
  }

  TV_HOST_DEVICE_INLINE size_type size() const { return hash_size_; }

  TV_HOST_DEVICE_INLINE value_type *data() { return table_; }

  TV_HOST_DEVICE_INLINE const value_type *data() const { return table_; }

  template <class... Args> TV_DEVICE_INLINE void insert(Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<const key_type_uint &>(std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(keys...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    while (true) {
      key_type_uint prev =
          atomicCAS(reinterpret_cast<key_type_uint *>(table_ + slot),
                    empty_key_uint, key_u);
      if (prev == empty_key_uint || prev == key_u) {
        break;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
  }

  template <class... Args>
  TV_DEVICE_INLINE size_type lookup_offset(Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<const key_type_uint &>(std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(keys...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    value_type val;
    int cnt = 0;
    while (cnt < 100) {
      cnt++;
      val = table_[slot];
      if (reinterpret_cast<key_type_uint &>(val) == key_u) {
        return slot;
      }
      // TODO
      if (val == empty_key) {
        return -1;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
    return -1;
  }
};

template <typename K, typename V, typename Hash = tv::hash::Murmur3Hash<K>, K EmptyKey = default_empty_key_v<K>,
          bool Power2 = false>
struct LinearHashTableSplit {
public:
  using key_type_uint = to_unsigned_t<K>;
  using key_type = K;
  // using value_type = pair<K, V, EmptyKey>;
  using mapped_type = V;
  using size_type = int32_t;
  // TODO no way to get constexpr uint repr of key for now.
  // static constexpr auto empty_key_uint = value_type::empty_key_uint;
  static constexpr auto empty_key = EmptyKey;
  using hash_type = Hash;

private:
  key_type *key_ptr_ = nullptr;
  mapped_type *value_ptr_ = nullptr;

  size_type hash_size_;
  key_type_uint empty_key_uint;

  static constexpr auto kNumHashArgs = argument_size_v<decltype(hash_type::hash)>;

public:
  TV_HOST_DEVICE explicit LinearHashTableSplit(key_type *key,
                                               mapped_type *value,
                                               size_type hash_size)
      : key_ptr_(key), value_ptr_(value), hash_size_(hash_size) {
    auto eky = empty_key;
    empty_key_uint = *(reinterpret_cast<const key_type_uint *>(&eky));
  }

  template <class... Args>
  TV_DEVICE_INLINE key_type_uint hash(Args &&...keys) const {
    return hash_type::hash(keys...);
  }

  TV_HOST_DEVICE_INLINE size_type size() const { return hash_size_; }

  TV_HOST_DEVICE_INLINE key_type *key_ptr() { return key_ptr_; }
  TV_HOST_DEVICE_INLINE const key_type *key_ptr() const { return key_ptr_; }
  TV_HOST_DEVICE_INLINE mapped_type *value_ptr() { return value_ptr_; }
  TV_HOST_DEVICE_INLINE const mapped_type *value_ptr() const {
    return value_ptr_;
  }
private:
  template <class... Args>
  TV_DEVICE_INLINE void insert_raw(const V &value, Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<const key_type_uint &>(std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(keys...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    while (true) {
      key_type_uint prev =
          atomicCAS(reinterpret_cast<key_type_uint *>(&key_ptr_[slot]),
                    empty_key_uint, key_u);
      if (prev == empty_key_uint || prev == key_u) {
        value_ptr_[slot] = value;
        break;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
  }

  template <class F, class... Args>
  TV_DEVICE_INLINE void insert_raw_custom_value(F&& f, Args &&...keys) {
    key_type_uint key_u = hash_type::encode(
        reinterpret_cast<const key_type_uint &>(std::forward<Args>(keys))...);
    key_type_uint hash_val = hash_type::hash(keys...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    while (true) {
      key_type_uint prev =
          atomicCAS(reinterpret_cast<key_type_uint *>(&key_ptr_[slot]),
                    empty_key_uint, key_u);
      if (prev == empty_key_uint || prev == key_u) {
        std::forward<F>(f)(value_ptr_ + slot);
        break;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
  }


  template <int... Inds>
  TV_DEVICE_INLINE void insert_raw(const array<K, kNumHashArgs>& key_arr, const V &value, mp_list_int<Inds...>) {
    return insert_raw(value, key_arr[Inds]...);
  }
  template <class F, int... Inds>
  TV_DEVICE_INLINE void insert_raw_custom_value(const array<K, kNumHashArgs>& key_arr, F&& f, mp_list_int<Inds...>) {
    return insert_raw_custom_value(std::forward<F>(f), key_arr[Inds]...);
  }

public:
  TV_DEVICE_INLINE void insert(const K& key, const V &value) {
    static_assert(kNumHashArgs == 1, "you must use tv::array if hash multiple values.");
    return insert_raw(value, key);
  }

  TV_DEVICE_INLINE void insert(const array<K, kNumHashArgs>& key_arr, const V &value) {
    return insert_raw(key_arr, value, mp_make_list_c_sequence<int, kNumHashArgs>{});
  }

  template <class F>
  TV_DEVICE_INLINE void insert_custom_value(const K& key, F&& f) {
    static_assert(kNumHashArgs == 1, "you must use tv::array if hash multiple values.");
    return insert_raw_custom_value(std::forward<F>(f), key);
  }

  template <class F>
  TV_DEVICE_INLINE void insert_custom_value(const array<K, kNumHashArgs>& key_arr, F&& f) {
    return insert_raw_custom_value(key_arr, std::forward<F>(f), mp_make_list_c_sequence<int, kNumHashArgs>{});
  }

  TV_DEVICE_INLINE int probe_length(int index, key_type key) {
    return (index - hash_type::hash_scalar(reinterpret_cast<key_type_uint&>(key)) % hash_size_);
  }

  template <class... Args>
  TV_DEVICE_INLINE void insert_key_only(Args &&...keys) {
    const key_type_uint key_u = hash_type::encode(std::forward<Args>(keys)...);
    key_type_uint hash_val = hash_type::hash(keys...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    while (true) {
      key_type_uint prev =
          atomicCAS(reinterpret_cast<key_type_uint *>(&key_ptr_[slot]),
                    empty_key_uint, key_u);
      if (prev == empty_key_uint || prev == key_u) {
        break;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
  }

  template <class... Args>
  TV_DEVICE_INLINE size_type lookup_offset(Args &&...keys) {
    key_type_uint key_u = hash_type::encode(std::forward<Args>(keys)...);
    key_type_uint hash_val = hash_type::hash(keys...);
    key_type_uint slot;
    if (Power2) {
      slot = hash_val & (hash_size_ - 1);
    } else {
      slot = hash_val % hash_size_;
    }
    while (true) {
      auto key_target = key_ptr_[slot];
      if (reinterpret_cast<key_type_uint &>(key_target) == key_u) {
        return slot;
      }
      if (key_target == empty_key) {
        return -1;
      }
      if (Power2) {
        slot = (slot + 1) & (hash_size_ - 1);
      } else {
        slot = (slot + 1) % hash_size_;
      }
    }
    return -1;
  }
};

} // namespace hash
} // namespace tv