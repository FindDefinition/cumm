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
