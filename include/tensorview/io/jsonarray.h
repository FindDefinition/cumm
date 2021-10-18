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
#include <tuple>
#include <vector>

#include <tensorview/thirdparty/nlohmann/json.hpp>
#include <string>
#include <tensorview/tensor.h>
#include <unordered_map>

namespace tv {
namespace io {
using json = nlohmann::json;
constexpr const char *kJsonArrayKey = "__tensorview_io_json_index";

struct JsonArray {
  json data;
  std::vector<tv::Tensor> tensors;
};

namespace detail {
template <typename K, typename V, typename Hash>
std::unordered_map<V, K, Hash>
inverse_map(const std::unordered_map<K, V, Hash> &dict) {
  std::unordered_map<V, K, Hash> res;
  for (auto &p : dict) {
    res[p.second] = p.first;
  }
  return res;
}

int64_t align_offset(int64_t offset, int64_t n) {
  if (n <= 0) {
    return offset;
  }
  return n * ((offset + n - 1) / n);
}

} // namespace detail


template <typename Tbuffer> JsonArray decode(const Tbuffer &buffer) {
  JsonArray res;
  auto buffer_ptr = buffer.data();
  if (buffer.size() < 16) {
    TV_THROW_RT_ERR("buffer length invalid");
  }
  int64_t content_end = *(reinterpret_cast<const int64_t *>(buffer_ptr));
  int64_t meta_end = *(reinterpret_cast<const int64_t *>(buffer_ptr + 8));
  TV_ASSERT_RT_ERR(content_end < (1 << 30) && meta_end < (1 << 30),
                   "must small than 1GB");
  TV_ASSERT_RT_ERR(content_end < meta_end, "error");

  auto meta =
      json::parse(buffer.begin() + content_end, buffer.begin() + meta_end);
  auto array_metas = meta["array"];
  auto data_skeleton = meta["data"];
  std::vector<tv::Tensor> res_tensors;
  for (auto it = array_metas.begin(); it < array_metas.end(); ++it) {
    auto &array_meta = (*it);
    auto shape = array_meta["shape"].template get<std::vector<int64_t>>();
    auto dtype = array_meta["dtype"].template get<int64_t>();
    auto offset = array_meta["offset"].template get<int64_t>();
    TV_ASSERT_RT_ERR(shape.size() <= tv::TensorShape::kMaxDim, "error");
    if (kJsonArrayTypeToTv.find(dtype) == kJsonArrayTypeToTv.end()) {
      TV_THROW_RT_ERR("dtype not found", dtype);
    }
    auto tv_dtype = kJsonArrayTypeToTv.at(dtype);
    tv::TensorShape shape_tensor;
    for (auto &c : shape) {
      shape_tensor.push_back(c);
    }
    tv::Tensor tensor =
        tv::from_blob(buffer_ptr + offset, shape_tensor, tv_dtype, -1);
    res_tensors.push_back(tensor.clone());
  }
  return {data_skeleton, res_tensors};
}

inline std::string encode(const std::vector<tv::Tensor> tensors, json &data_json) {
  json res_json;
  int64_t start = 16;
  constexpr int64_t align_size = 128;
  std::vector<int64_t> offsets;
  res_json["array"] = json::array();
  for (size_t i = 0; i < tensors.size(); ++i) {
    json array_meta;
    auto &tensor = tensors[i];
    auto &shape = tensor.shape();
  
    int64_t start_aligned = detail::align_offset(start, align_size);
    array_meta["shape"] = std::vector<int64_t>(shape.begin(), shape.end());
    array_meta["dtype"] = int(tensor.dtype());
    array_meta["offset"] = start_aligned;
    start = start_aligned + tensor.nbytes();
    res_json["array"].push_back(array_meta);
    offsets.push_back(start_aligned);
  }
  res_json["data"] = data_json;
  std::string meta_json = res_json.dump();
  size_t total_length = start + meta_json.size();
  std::string res;
  res.resize(total_length);

  auto res_data = &res[0];
  *(reinterpret_cast<int64_t *>(res_data)) = start;
  *(reinterpret_cast<int64_t *>(res_data + 8)) = total_length;

  for (size_t i = 0; i < tensors.size(); ++i) {
    auto &tensor = tensors[i];
    auto offset = offsets[i];
    tv::Dispatch<tv::detail::all_tensor_types_t>()(tensor.dtype(), [&](auto I) {
      using T = TV_DECLTYPE(I);
      std::copy(tensor.data<T>(), tensor.data<T>() + tensor.size(),
                reinterpret_cast<T *>(res_data + offset));
    });
  }
  std::copy(meta_json.begin(), meta_json.end(), res_data + start);
  return res;
}

inline json json_idx(size_t i) {
  json res;
  res[kJsonArrayKey] = i;
  return res;
}

inline json access_idx(json j) {
  return j[kJsonArrayKey].template get<int64_t>();
}


} // namespace codeai
} // namespace codeai