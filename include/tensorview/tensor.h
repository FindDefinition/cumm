// Copyright 2019-2021 Yan Yan
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

/*
tv::Tensor is a lightweight header-only tensor container
without template and annoying dependencies. no algorithm is implemented.
it should only be used when you want a no-template simple container but
dont want to link with libtorch.

If you can use libtorch, dont use tv::Tensor.
*/
#pragma once

#include "core/cc17.h"
#include "dtypes.h"
#include "mp_helper.h"
#include "tensorview.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <memory>
#include <type_traits>
#include "context.h"
#ifdef TV_CUDA
#include "cuda/driverops.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#endif
#if (CUDA_VERSION >= 11000 && defined(TV_CUDA))
#include <cuda_bf16.h>
#endif
#include <random>

namespace tv {
namespace detail {

using dtype_collection_t =
    mp_list_c<int, float32, int32, int16, int8, float64, bool_, uint8, float16,
              int64, uint16, uint32, uint64>;

#if defined(TV_CUDA) && CUDA_VERSION < 11000
using all_tensor_types_t =
    std::tuple<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
               uint16_t, uint32_t, uint64_t, bool, __half>;
#elif defined(TV_CUDA) && CUDA_VERSION >= 11000
using all_tensor_types_t =
    std::tuple<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
               uint16_t, uint32_t, uint64_t, bool, __half, __nv_bfloat16>;
#else
using all_tensor_types_t =
    std::tuple<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
               uint16_t, uint32_t, uint64_t, bool>;
#endif

using all_tensor_types_print_t =
    std::tuple<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
               uint16_t, uint32_t, uint64_t, bool>;
        
using all_int_tensor_types_t =
    std::tuple<int8_t, int16_t, int32_t, int64_t, uint8_t,
               uint16_t, uint32_t, uint64_t>;


template <typename T> class TensorStorage {
public:
  TensorStorage(size_t size, int device = -1, bool managed = false,
                bool pinned = false)
      : size_(size), device_(device), managed_(managed), pinned_(pinned) {
    if (size == 0) {
      ptr_ = nullptr;
    } else {
      if (device == -1) {
        if (pinned_) {
#ifdef TV_CUDA
          checkCudaErrors(cudaMallocHost(&ptr_, size * sizeof(T)));
#else
          TV_THROW_INVALID_ARG("you need to define TV_CUDA to use pinned");
#endif
        } else {
          ptr_ = new T[size];
        }
      } else {
#ifdef TV_CUDA
        if (managed) {
          checkCudaErrors(cudaMallocManaged(&this->ptr_, size * sizeof(T)));
        } else {
          checkCudaErrors(cudaMalloc(&ptr_, size * sizeof(T)));
        }
#else
        TV_THROW_INVALID_ARG("don't compiled with cuda");
#endif
      }
    }
  }
  TensorStorage(T *ptr, size_t size, int device)
      : size_(size), ptr_(ptr), from_blob_(true), device_(device) {}

  virtual ~TensorStorage() {
    if (empty()) {
      return;
    }
    if (from_blob_) {
      return;
    }
    if (device_ == -1) {
      if (pinned_) {
#ifdef TV_CUDA
        cudaFreeHost(ptr_);
#endif
      } else {
        delete[] ptr_;
      }
    } else {
#ifdef TV_CUDA
      cudaFree(ptr_);
#endif
    }
  };

  inline size_t size() const { return size_; }

  T *data() { return ptr_; }
  const T *data() const { return ptr_; }
  bool is_cpu() const { return device_ == -1; }

  bool empty() const { return ptr_ == nullptr || size_ == 0; }
  bool managed() const { return managed_; }
  bool pinned() const { return pinned_; }

  int device() const { return device_; }
  void zero_(int64_t offset, int64_t length, Context ctx = Context()) {
    TV_ASSERT_RT_ERR(length <= size_ - offset, "eror");
    if (device_ == -1) {
      std::memset(data() + offset, 0, length);
      // std::fill(data(), data() + size_, 0);
    } else {
#ifdef TV_CUDA
      if (ctx.has_cuda_stream()) {
        checkCudaErrors(cudaMemsetAsync(data() + offset * sizeof(T), 0,
                                        length * sizeof(T), ctx.cuda_stream()));
      } else {
        checkCudaErrors(
            cudaMemset(data() + offset * sizeof(T), 0, length * sizeof(T)));
      }
#else
      TV_THROW_INVALID_ARG("don't compiled with cuda");
#endif
    }
  }

private:
  size_t size_ = 0;
  T *ptr_ = nullptr;
  bool from_blob_ = false;
  int device_ = -1;
  bool managed_ = false;
  bool pinned_ = false;
};

template <typename T> size_t sizeof_dtype(T dtype) {
  switch (dtype) {
  case float32:
    return sizeof(float);
  case int8:
    return sizeof(int8_t);
  case int16:
    return sizeof(int16_t);
  case int32:
    return sizeof(int32_t);
  case float64:
    return sizeof(double);
  case int64:
    return sizeof(int64_t);
  case bool_:
    return sizeof(bool);
  case uint8:
    return sizeof(uint8_t);
  case uint16:
    return sizeof(uint16_t);
  case uint32:
    return sizeof(uint32_t);
  case uint64:
    return sizeof(uint64_t);
  case float16:
    return 2;
#if (CUDA_VERSION >= 11000 && defined(TV_CUDA))
  case bfloat16:
    return 2;
  case tf32:
    return sizeof(float);
#endif
  case custom16:
    return 2;
  case custom32:
    return 4;
  case custom48:
    return 6;
  case custom64:
    return 8;
  case custom80:
    return 10;
  case custom96:
    return 12;
  case custom128:
    return 16;
  default:
    TV_THROW_RT_ERR("unsupported dtype");
  }
  return 0;
}

template <class Tsrc, class Tdst>
struct ConvertTmpType {
  using type = Tdst;
  static constexpr bool kSpec = false;
};

#if (CUDA_VERSION >= 11000 && defined(TV_CUDA))
template <class T>
struct ConvertTmpType<T, __nv_bfloat16>{
  using type = float;
  static constexpr bool kSpec = true;
};
#endif
} // namespace detail
template <class... Ts, typename F> bool dispatch_noexcept(DType t, F &&f) {
  static_assert(sizeof...(Ts) > 0, "you need to provide at least one type");
  bool notFound = true;
  mp_for_each<mp_list<Ts...>>([=, &notFound, &f](auto I) {
    if (type_v<TV_DECLTYPE(I)> == t && notFound) {
      std::forward<F>(f)(TV_DECLTYPE(I)());
      notFound = false;
    }
  });
  return !notFound;
}

template <class... Ts, typename F> void dispatch(DType t, F &&f) {
  if (!dispatch_noexcept<Ts...>(t, std::forward<F>(f))) {
    std::stringstream ss;
    mp_for_each<mp_list<Ts...>>([=, &ss](auto I) {
      ss << type_s<std::decay_t<TV_DECLTYPE(I)>> << " ";
    });
    TV_THROW_RT_ERR("unknown type", dtype_str(t), ", available:", ss.str());
  }
}

template <typename T, T... Is, typename F> void dispatch_scalar(T idx, F &&f) {
  static_assert(sizeof...(Is) > 0,
                "you need to provide at least one candidate");
  bool notFound = true;
  mp_for_each<mp_list_c<T, Is...>>([=, &notFound, &f](auto I) {
    if (T(I) == idx && notFound) {
      std::forward<F>(f)(I);
      notFound = false;
    }
  });
  if (notFound) {
    std::stringstream ss;
    mp_for_each<mp_list_c<T, Is...>>([=, &ss](auto I) { ss << T(I) << " "; });
    TV_THROW_RT_ERR("unknown value", idx, ", available:", ss.str());
  }
}

template <int... Is, typename F> bool dispatch_int_noexcept(int idx, F &&f) {
  static_assert(sizeof...(Is) > 0,
                "you need to provide at least one candidate");
  bool notFound = true;
  mp_for_each<mp_list_c<int, Is...>>([=, &notFound, &f](auto I) {
    if (TV_DECLTYPE(I)::value == idx && notFound) {
      std::forward<F>(f)(I);
      notFound = false;
    }
  });
  return !notFound;
}

template <int... Is, typename F, class BinaryPredicate>
bool dispatch_int_noexcept(int idx, BinaryPredicate p, F &&f) {
  static_assert(sizeof...(Is) > 0,
                "you need to provide at least one candidate");
  bool notFound = true;
  mp_for_each<mp_list_c<int, Is...>>([=, &notFound, &f](auto I) {
    if (p(idx, TV_DECLTYPE(I)::value) && notFound) {
      std::forward<F>(f)(I);
      notFound = false;
    }
  });
  return !notFound;
}

template <int... Is, typename F> void dispatch_int(int idx, F &&f) {
  if (!dispatch_int_noexcept<Is...>(idx, std::forward<F>(f))) {
    std::stringstream ss;
    mp_for_each<mp_list_c<int, Is...>>(
        [=, &ss](auto I) { ss << TV_DECLTYPE(I)::value << " "; });
    TV_THROW_RT_ERR("unknown value", idx, ", available:", ss.str());
  }
}

template <int... Is, typename F, class BinaryPredicate>
void dispatch_int(int idx, BinaryPredicate p, F &&f) {
  // BinaryPredicate: BinaryPredicate(idx, candidate)
  if (!dispatch_int_noexcept<Is...>(idx, p, std::forward<F>(f))) {
    std::stringstream ss;
    mp_for_each<mp_list_c<int, Is...>>(
        [=, &ss](auto I) { ss << TV_DECLTYPE(I)::value << " "; });
    TV_THROW_RT_ERR("unknown value", idx, ", available:", ss.str());
  }
}

// Ts is pack of mp_list_c
template <class... Ts, typename Iterator, typename F>
bool dispatch_container_noexcept(Iterator begin, Iterator end, F &&f) {
  static_assert(sizeof...(Ts) > 0,
                "you need to provide at least one candidate");
  bool notFound = true;
  mp_for_each<mp_list<Ts...>>([=, &notFound, &f](auto I) {
    using val_lst_t = TV_DECLTYPE(I);
    auto val_lst_size = mp_size<val_lst_t>::value;
    bool equal = true;
    std::size_t count = 0;
    auto iter = begin;
    mp_for_each<val_lst_t>([&](auto E) {
      if (iter == end || !equal) {
        return;
      }
      if (count >= val_lst_size) {
        equal = false;
        return;
      }
      constexpr auto c = TV_DECLTYPE(E)::value;
      if (c != *iter) {
        equal = false;
      }
      ++count;
      std::advance(iter, 1);
    });
    if (count != val_lst_size || iter != end) {
      equal = false;
    }
    if (equal && notFound) {
      std::forward<F>(f)(I);
      notFound = false;
    }
  });

  return !notFound;
}

template <class... Ts, typename Iterator, typename F>
void dispatch_container(Iterator begin, Iterator end, F &&f) {
  if (!dispatch_container_noexcept<Ts...>(begin, end, std::forward<F>(f))) {
    std::stringstream ss;
    ss << "unknown value [";
    for (auto iter = begin; iter != end; std::advance(iter, 1)) {
      ss << *iter << ",";
    }
    ss << "], available: ";
    mp_for_each<mp_list<Ts...>>([=, &ss](auto I) {
      ss << "[";
      mp_for_each<TV_DECLTYPE(I)>(
          [=, &ss](auto E) { ss << TV_DECLTYPE(E)::value << ","; });
      ss << "]";
    });
    TV_THROW_RT_ERR(ss.str());
  }
}

/*
template <int... Is, typename F> void dispatch_int(int idx, F &&f) {
  return dispatch_scalar<int, Is...>(idx, f);
}
*/

template <class T> struct Dispatch;

template <template <class...> class T, class... Args>
struct Dispatch<T<Args...>> {
  template <typename F> inline void operator()(DType t, F &&f) {
    return dispatch<Args...>(t, std::forward<F>(f));
  }
};


template <class T> struct DispatchNoExcept;

template <template <class...> class T, class... Args>
struct DispatchNoExcept<T<Args...>> {
  template <typename F> inline bool operator()(DType t, F &&f) {
    return dispatch_noexcept<Args...>(t, std::forward<F>(f));
  }
};


template <class T> struct DispatchContainer;

template <template <class...> class T, class... Args>
struct DispatchContainer<T<Args...>> {
  template <typename Iterator, typename F>
  inline void operator()(Iterator begin, Iterator end, F &&f) {
    return dispatch_container<Args...>(begin, end, std::forward<F>(f));
  }
};

template <class T> struct DispatchContainerNoexcept;

template <template <class...> class T, class... Args>
struct DispatchContainerNoexcept<T<Args...>> {
  template <typename Iterator, typename F>
  inline bool operator()(Iterator begin, Iterator end, F &&f) {
    return dispatch_container_noexcept<Args...>(begin, end, std::forward<F>(f));
  }
};

template <class T> struct DispatchInt;

// Args should be std::integral_constant<int, value>
// you need to use type_container<std::integral_constant<int, value>...>
// as template parameter of DispatchInt.
// tv::mp_list_c is ok.
template <template <class...> class T, class... Args>
struct DispatchInt<T<Args...>> {
  template <typename F> inline void operator()(int t, F &&f) {
    return dispatch_int<Args::value...>(t, std::forward<F>(f));
  }
  template <typename F, typename BinaryPredicate>
  inline void operator()(int t, BinaryPredicate p, F &&f) {
    return dispatch_int<Args::value...>(t, p, std::forward<F>(f));
  }
};

template <class T> struct DispatchIntNoexcept;

template <template <class...> class T, class... Args>
struct DispatchIntNoexcept<T<Args...>> {
  template <typename F> inline bool operator()(int t, F &&f) {
    return dispatch_int_noexcept<Args::value...>(t, std::forward<F>(f));
  }
  template <typename F, typename BinaryPredicate>
  inline bool operator()(int t, BinaryPredicate p, F &&f) {
    return dispatch_int_noexcept<Args::value...>(t, p, std::forward<F>(f));
  }
};

constexpr size_t kTensorMaxDim = 10;
using TensorShape = ShapeBase<kTensorMaxDim, int64_t>;

struct Tensor {
  Tensor() {}
  Tensor(TensorShape shape, TensorShape stride, DType dtype, int device = -1,
         bool pinned = false, bool managed = false)
      : dtype_(dtype) {
    TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        shape.size() * detail::sizeof_dtype(dtype), device, managed, pinned);
    shape_ = shape;
    stride_ = stride;
  }

  Tensor(TensorShape shape, DType dtype, int device = -1, bool pinned = false,
         bool managed = false)
      : dtype_(dtype) {
    TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        shape.size() * detail::sizeof_dtype(dtype), device, managed, pinned);
    shape_ = shape;
    stride_ = shape.stride_rowmajor();
  }
  Tensor(void *ptr, TensorShape shape, TensorShape stride, DType dtype,
         int device = -1)
      : dtype_(dtype) {
    TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(ptr),
        shape.size() * detail::sizeof_dtype(dtype), device);
    shape_ = shape;
    stride_ = stride;
  }
  Tensor(void *ptr, TensorShape shape, DType dtype, int device = -1)
      : dtype_(dtype) {
    TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(ptr),
        shape.size() * detail::sizeof_dtype(dtype), device);
    shape_ = shape;
    stride_ = shape.stride_rowmajor();
  }

  Tensor(const void *ptr, TensorShape shape, TensorShape stride, DType dtype,
         int device = -1)
      : dtype_(dtype), writeable_(false) {
    TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(const_cast<void *>(ptr)),
        shape.size() * detail::sizeof_dtype(dtype), device);
    shape_ = shape;
    stride_ = stride;
  }
  Tensor(const void *ptr, TensorShape shape, DType dtype, int device = -1)
      : dtype_(dtype), writeable_(false) {
    TV_ASSERT_INVALID_ARG(!shape.empty(), "dont support empty shape");
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        reinterpret_cast<uint8_t *>(const_cast<void *>(ptr)),
        shape.size() * detail::sizeof_dtype(dtype), device);
    shape_ = shape;
    stride_ = shape.stride_rowmajor();
  }

  Tensor(std::initializer_list<int32_t> init)
      : Tensor({int(init.size())}, int32) {
    std::copy(init.begin(), init.end(), data<int32_t>());
  }
  Tensor(std::initializer_list<int64_t> init)
      : Tensor({int(init.size())}, int64) {
    std::copy(init.begin(), init.end(), data<int64_t>());
  }
  Tensor(std::initializer_list<float> init)
      : Tensor({int(init.size())}, float32) {
    std::copy(init.begin(), init.end(), data<float>());
  }
  Tensor(std::initializer_list<double> init)
      : Tensor({int(init.size())}, float64) {
    std::copy(init.begin(), init.end(), data<double>());
  }

  template <typename Iterator, typename = detail::_RequireInputIter<Iterator>>
  Tensor(Iterator first, Iterator second, bool pinned = false) {
    using T = typename std::iterator_traits<Iterator>::value_type;
    dtype_ = type_v<T>;
    shape_ = {second - first};
    stride_ = shape_.stride_rowmajor();
    storage_ = std::make_shared<detail::TensorStorage<uint8_t>>(
        shape_.size() * detail::sizeof_dtype(dtype_), -1, false, pinned);
    writeable_ = true;
    std::copy(first, second, data_ptr<T>());
  }

  template <typename T, int Rank = -1,
            template <class> class PtrTraits = DefaultPtrTraits,
            typename Tindex = TV_GLOBAL_INDEX>
  decltype(auto) tview() const {
    static_assert(Rank == -1 || Rank > 0, "error");
    template_dtype_check<T>();
    return if_constexpr<(Rank > 0)>(
        [&](auto _) {
          // detail::_if_constexpr_workaround<Rank, (Rank > 0)> _val;
          // TV_ASSERT_RT_ERR(_(_val).value == ndim(), "error");
          ShapeBase<Rank == -1 ? TV_MAX_DIM : Rank, Tindex> shape(_(Rank)),
              stride(_(Rank));
          for (int i = 0; i < Rank; ++i) {
            shape[i] = shape_[i];
            stride[i] = stride_[i];
          }
          return TensorView<const std::remove_const_t<T>, Rank, PtrTraits,
                            Tindex>(
              reinterpret_cast<const std::remove_const_t<T> *>(this->data<T>()),
              _(shape), _(stride));
        },
        [&](auto _) {
          ShapeBase<TV_MAX_DIM, Tindex> shape(this->ndim()),
              stride(this->ndim());
          for (int i = 0; i < int(this->ndim()); ++i) {
            shape[i] = shape_[i];
            stride[i] = stride_[i];
          }
          return TensorView<const std::remove_const_t<T>, Rank, PtrTraits,
                            Tindex>(
              reinterpret_cast<const std::remove_const_t<T> *>(this->data<T>()),
              _(shape), _(stride));
        });
  }

  template <typename T, int Rank = -1,
            template <class> class PtrTraits = DefaultPtrTraits,
            typename Tindex = TV_GLOBAL_INDEX>
  decltype(auto) tview() {
    static_assert(Rank == -1 || Rank > 0, "error");
    template_dtype_check<T>();
    return if_constexpr<(Rank > 0)>(
        [&](auto _) {
          // detail::_if_constexpr_workaround<Rank, (Rank > 0)> _val;
          // TV_ASSERT_RT_ERR(_(_val).value == ndim(), "error");
          ShapeBase<Rank == -1 ? TV_MAX_DIM : Rank, Tindex> shape(_(Rank)),
              stride(_(Rank));
          for (int i = 0; i < Rank; ++i) {
            shape[i] = shape_[i];
            stride[i] = stride_[i];
          }
          return TensorView<T, Rank, PtrTraits, Tindex>(
              reinterpret_cast<T *>(this->data<T>()), _(shape), _(stride));
        },
        [&](auto _) {
          ShapeBase<TV_MAX_DIM, Tindex> shape(this->ndim()),
              stride(this->ndim());
          for (int i = 0; i < int(this->ndim()); ++i) {
            shape[i] = shape_[i];
            stride[i] = stride_[i];
          }
          return TensorView<T, Rank, PtrTraits, Tindex>(
              reinterpret_cast<T *>(this->data<T>()), _(shape), _(stride));
        });
  }

  Tensor view(TensorShape shape) const {
    bool found_minus_1 = false;
    for (size_t i = 0; i < shape.ndim(); ++i) {
      if (!found_minus_1) {
        if (shape[i] == -1) {
          shape[i] = 1;
          shape[i] = size() / shape.size();
          found_minus_1 = true;
        } else {
          TV_ASSERT_INVALID_ARG(shape[i] > 0,
                                "shape except -1 must larger than 0");
        }
      } else {
        TV_ASSERT_INVALID_ARG(shape[i] > 0, "multiple -1 in your argument.");
      }
    }
    TV_ASSERT_RT_ERR(shape.size() == size(), "error");
    Tensor res(*this);
    res.shape_ = shape;
    res.stride_ = shape.stride_rowmajor();
    return res;
  }

  template <class... Inds> Tensor view(Inds... newShapes) const {
    static_assert(sizeof...(newShapes) > 0, "dont support empty for now");
    TensorShape shape{int(newShapes)...};
    return view(shape);
  }

  Tensor operator[](int64_t index) {
    TV_ASSERT_INVALID_ARG(ndim() > 1, "error");
    if (index < 0) {
      index += dim(0);
    }
    TV_ASSERT_INVALID_ARG(index < dim(0), "error");
    Tensor res = Tensor();
    res.dtype_ = dtype_;
    res.storage_ = storage_;
    res.shape_ = shape_.subshape(1);
    res.offset_ = offset_ + index * stride_[0] * itemsize();
    res.stride_ = stride_.subshape(1);
    res.writeable_ = writeable_;
    return res;
  }

  Tensor squeeze() const { return view(shape_.squeeze()); }

  Tensor squeeze(int axis) const {
    if (axis < 0) {
      axis = ndim() + axis;
    }
    return view(shape_.squeeze(axis));
  }

  Tensor unsqueeze(int axis) const {
    if (axis < 0) {
      axis = ndim() + axis + 1;
    }
    TV_ASSERT_INVALID_ARG(axis >= 0 && axis < ndim() + 1, "error");
    return view(shape_.unsqueeze(axis));
  }

  bool pinned() const { return storage_->pinned(); }

  Tensor slice_first_axis(int64_t start, int64_t end) const {
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    if (start < 0) {
      start = shape_[0] + start;
    }
    if (end < 0) {
      end = shape_[0] + end;
    }
    TV_ASSERT_INVALID_ARG(start < shape_[0], "start must small than dim 0",
                          shape_[0]);
    TV_ASSERT_INVALID_ARG(start < end, "start must small than end");
    size_t new_offset = offset_ + start * shape_.prod(1) * itemsize();
    Tensor res(*this);
    TensorShape newshape(shape_);
    newshape[0] = std::min(end - start, shape_[0]);
    res.shape_ = newshape;
    res.stride_ = stride_;
    res.offset_ = new_offset;
    res.writeable_ = writeable_;
    res.contiguous_ = contiguous_;
    return res;
  }

  bool empty() const { return !storage_ || storage_->empty(); }
  DType dtype() const { return dtype_; }
  int device() const { return storage_->device(); }
  size_t ndim() const { return shape_.ndim(); }

  const TensorShape &shape() const { return shape_; }
  const TensorShape &strides() const { return stride_; }
  int stride(int idx) const {
    if (idx < 0) {
      TV_ASSERT_RT_ERR(stride_.ndim() + idx < stride_.ndim(), idx, stride_);
      return stride_[stride_.ndim() + idx];
    } else {
      TV_ASSERT_RT_ERR(idx < int(stride_.ndim()), idx, stride_);
      return stride_[idx];
    }
  }
  const TensorShape &sizes() const { return shape_; }
  const TensorShape &stride() const { return stride_; }

  int dim(int idx) const {
    if (idx < 0) {
      TV_ASSERT_RT_ERR(shape_.ndim() + idx < shape_.ndim(), idx, shape_);
      return shape_[shape_.ndim() + idx];
    } else {
      TV_ASSERT_RT_ERR(idx < int(shape_.ndim()), idx, shape_);
      return shape_[idx];
    }
  }
  const uint8_t *raw_data() const { 
    if (empty()){
      return nullptr;
    }
    return storage_->data() + byte_offset(); 
  }
  size_t raw_size() const { return size() * itemsize(); }
  size_t nbytes() const { return raw_size(); }
  size_t size() const { return shape_.size(); }
  size_t size(int64_t idx) const { return dim(idx); }
  size_t storage_size() const { return storage_->size(); }

  size_t itemsize() const { return detail::sizeof_dtype(dtype_); }
  size_t byte_offset() const { return offset_; }
  Tensor &zero_(Context ctx = Context()) {
    writable_check();
    storage_->zero_(byte_offset(), raw_size(), ctx);
    return *this;
  }

  uint8_t *raw_data() {
    if (empty()){
      return nullptr;
    }
    writable_check();
    return storage_->data() + byte_offset();
  }
  template <typename T> Tensor &fill_template_(T val, Context ctx) {
    writable_check();
    if (this->device() == -1) {
      std::fill(this->data_ptr<T>(), this->data_ptr<T>() + this->size(), val);
    } else {
#ifdef TV_CUDA
      auto tview = this->tview<T, -1, tv::DefaultPtrTraits, int64_t>();
      if (ctx.has_cuda_stream()) {
        tv::FillDev<T, -1, tv::DefaultPtrTraits, int64_t>::run_async(
            tview, val, ctx.cuda_stream());
      } else {
        tv::FillDev<T, -1, tv::DefaultPtrTraits, int64_t>::run(tview, val);
      }
#else
      TV_THROW_INVALID_ARG("don't compiled with cuda");
#endif
    }
    return *this;
  }
  template <typename T> Tensor &fill_template_(T val) {
    return fill_template_<T>(val, Context());
  }

  Tensor &fill_(int val, Context ctx = Context()) {
    using int_types_t =
        std::tuple<int32_t, int16_t, int8_t, uint32_t, uint16_t, uint8_t>;
    Dispatch<int_types_t>()(dtype_, [&](auto I) -> void {
      using T = TV_DECLTYPE(I);
      fill_template_<T>(val, ctx);
    });
    return *this;
  }

  Tensor &fill_(float val, Context ctx = Context()) {
    using float_types_t = std::tuple<float>;
    Dispatch<float_types_t>()(dtype_, [&](auto I) -> void {
      using T = TV_DECLTYPE(I);
      fill_template_<T>(val, ctx);
    });
    return *this;
  }

  template <typename T> T *data() {
    if (empty()){
      return nullptr;
    }
    template_dtype_check<T>();
    writable_check();
    return reinterpret_cast<T *>(raw_data());
  }

  template <typename T> const T *data() const {
    if (empty()){
      return nullptr;
    }
    template_dtype_check<T>();
    return reinterpret_cast<const T *>(raw_data());
  }

  template <typename T> T *data_ptr() { return data<T>(); }

  template <typename T> const T *data_ptr() const { return data<T>(); }

  void *data_ptr() { return reinterpret_cast<void *>(raw_data()); }

  const void *data_ptr() const {
    return reinterpret_cast<const void *>(raw_data());
  }
  bool managed() const { return storage_->managed(); }
  void copy_(const Tensor &tensor, Context ctx = Context()) {
    writable_check();
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    TV_ASSERT_RT_ERR(!this->empty() && !tensor.empty(), "must not empty");
    TV_ASSERT_RT_ERR(this->size() == tensor.size(), "must have same size");
    TV_ASSERT_RT_ERR(this->dtype() == tensor.dtype(), "must have same dtype",
                     dtype_str(this->dtype()), dtype_str(tensor.dtype()));
    if (this->device() == -1 && tensor.device() == -1) {
#ifdef TV_CUDA
      // use memcpy instead to avoid cuda context init
      // host2host(raw_data(), tensor.raw_data(),
      //           size() * detail::sizeof_dtype(dtype_));
      std::copy(tensor.raw_data(),
                tensor.raw_data() + size() * detail::sizeof_dtype(dtype_),
                raw_data());
      // if (ctx.has_cuda_stream()) {
      //   host2host(this->raw_data(), tensor.raw_data(),
      //             this->size() * detail::sizeof_dtype(dtype_),
      //             ctx.cuda_stream());

      // } else {
      //   host2host(this->raw_data(), tensor.raw_data(),
      //             this->size() * detail::sizeof_dtype(dtype_));
      // }
#else
      std::copy(tensor.raw_data(),
                tensor.raw_data() + size() * detail::sizeof_dtype(dtype_),
                raw_data());
#endif
    }
#ifdef TV_CUDA
    else if (device() >= 0 && tensor.device() == -1) {
      if (ctx.has_cuda_stream()) {
        host2dev(raw_data(), tensor.raw_data(),
                 size() * detail::sizeof_dtype(dtype_), ctx.cuda_stream());

      } else {
        host2dev(raw_data(), tensor.raw_data(),
                 size() * detail::sizeof_dtype(dtype_));
      }

    } else if (device() == -1 && tensor.device() >= 0) {
      if (ctx.has_cuda_stream()) {
        dev2host(raw_data(), tensor.raw_data(),
                 size() * detail::sizeof_dtype(dtype_), ctx.cuda_stream());

      } else {
        dev2host(raw_data(), tensor.raw_data(),
                 size() * detail::sizeof_dtype(dtype_));
      }
    } else if (device() >= 0 && tensor.device() >= 0) {
      if (ctx.has_cuda_stream()) {
        dev2dev(raw_data(), tensor.raw_data(),
                size() * detail::sizeof_dtype(dtype_), ctx.cuda_stream());
      } else {
        dev2dev(raw_data(), tensor.raw_data(),
                size() * detail::sizeof_dtype(dtype_));
      }

    }
#endif
    else {
      TV_THROW_RT_ERR("only support cpu tensor");
    }
  }

  void copy_cpu_(const Tensor &tensor) {
    // this function exists for cloud environment that
    // use cheap non-gpu instance with library compiled
    // with cuda.
    writable_check();
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    TV_ASSERT_RT_ERR(!this->empty() && !tensor.empty(), "must not empty");
    TV_ASSERT_RT_ERR(this->size() == tensor.size(), "must have same size");
    TV_ASSERT_RT_ERR(this->dtype() == tensor.dtype(), "must have same dtype",
                     dtype_str(this->dtype()), dtype_str(tensor.dtype()));
    TV_ASSERT_RT_ERR(this->device() == -1 && tensor.device() == -1,
                     "all tensors must be cpu");

    Dispatch<detail::all_tensor_types_t>()(tensor.dtype(), [&](auto I) {
      using T = TV_DECLTYPE(I);
      std::copy(tensor.data<T>(), tensor.data<T>() + tensor.size(),
                this->data<T>());
    });
  }

  Tensor cpu(Context ctx = Context()) const {
    if (storage_->device() == -1) {
      // cpu() should always copy tensor.
      return clone();
    }
    Tensor res(shape_, stride_, dtype_, -1, storage_->managed());
    res.copy_(*this);
    return res;
  }

#ifdef TV_CUDA
  Tensor cuda(Context ctx = Context()) const {
    if (storage_->device() >= 0) {
      // cuda() should always copy tensor.
      return clone();
    }
    Tensor res(shape_, stride_, dtype_, 0, storage_->managed());
    res.copy_(*this, ctx);
    return res;
  }
#endif

  template <typename T> T item() const {
    TV_ASSERT_RT_ERR(size() == 1, "size must be 1");
    auto tensor = *this;
    if (storage_->is_cpu()) {
      tensor = cpu();
    }
    return *(tensor.data_ptr<T>());
  }

  template <typename T> void copy_(const TensorView<T> &tensor, int device) {
    writable_check();
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    Tensor src = from_blob(tensor, device);
    return copy_(src);
  }

  Tensor &operator=(const Tensor &tensor) {
    dtype_ = tensor.dtype_;
    storage_ = tensor.storage_;
    shape_ = tensor.shape_;
    writeable_ = tensor.writeable_;
    offset_ = tensor.offset_;
    stride_ = tensor.stride_;
    return *this;
  }

  Tensor(const Tensor &tensor) {
    dtype_ = tensor.dtype_;
    storage_ = tensor.storage_;
    shape_ = tensor.shape_;
    writeable_ = tensor.writeable_;
    offset_ = tensor.offset_;
    stride_ = tensor.stride_;
  }

  Tensor type_view(DType dtype, TensorShape newshape) const {
    TV_ASSERT_INVALID_ARG(detail::sizeof_dtype(dtype) * newshape.size() == itemsize() * this->size(),
                          "dtype itemsize multiple size must same");
    auto ten = *this;
    ten.dtype_ = dtype;
    ten.shape_ = newshape;
    ten.stride_ = newshape.stride_rowmajor();

    return ten;
  }

  Tensor type_view(DType dtype) const {
    auto dtype_size = detail::sizeof_dtype(dtype);
    auto self_dtype_size = itemsize();
    auto new_shape = shape_;
    if (dtype_size >= self_dtype_size){
      TV_ASSERT_INVALID_ARG(dtype_size % self_dtype_size == 0, "error", dtype_size, self_dtype_size);
      int rate = dtype_size / self_dtype_size;
      TV_ASSERT_INVALID_ARG(this->dim(this->ndim() - 1) % rate == 0, "error", this->dim(this->ndim() - 1), rate);
      auto new_shape = shape_;
      new_shape[this->ndim() - 1] = this->dim(this->ndim() - 1) / rate;
    }else{
      TV_ASSERT_INVALID_ARG(self_dtype_size % dtype_size == 0, "error", dtype_size, self_dtype_size);
      int rate = self_dtype_size / dtype_size;
      new_shape[this->ndim() - 1] = this->dim(this->ndim() - 1) * rate;
    }
    auto ten = *this;
    ten.dtype_ = dtype;
    ten.shape_ = new_shape;
    ten.stride_ = new_shape.stride_rowmajor();
    return ten;
  }

  Tensor clone(bool pinned = false, bool use_cpu_copy = false) const {
    if (empty()) {
      return Tensor();
    }
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    Tensor newtensor(shape_, stride_, dtype_, device(), pinned,
                     storage_->managed());
    if (!empty()) {
      if (use_cpu_copy) {
        TV_ASSERT_INVALID_ARG(device() == -1, "tensor must be cpu");
        newtensor.copy_cpu_(*this);
      } else {
        newtensor.copy_(*this);
      }
    }
    return newtensor;
  }

  Tensor rand_(int seed = -1) {
    std::random_device rd;
    if (seed == -1){
      seed = rd();
    }
    dispatch<float, double>(dtype_, [&](auto I) {
      using T = TV_DECLTYPE(I);
      TensorView<T> tensor_tv = this->tview<T>();
      if (this->device() == -1) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<T> distr(0, 1);
        for (size_t i = 0; i < this->size(); ++i) {
          tensor_tv[i] = distr(generator);
        }
      } else {
#ifdef TV_CUDA
        curandGenerator_t gen;
        auto status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        if (status != CURAND_STATUS_SUCCESS) {
          TV_THROW_RT_ERR("curand failed", status);
        }
        status = curandSetPseudoRandomGeneratorSeed(gen, seed);
        if (status != CURAND_STATUS_SUCCESS) {
          TV_THROW_RT_ERR("curand failed", status);
        }
        if_constexpr<std::is_same<T, float>::value>(
            [&](auto _) {
              status =
                  curandGenerateUniform(gen, _(this->data_ptr<T>()), size());
              if (status != CURAND_STATUS_SUCCESS) {
                TV_THROW_RT_ERR("curand failed", status);
              }
            },
            [&](auto _) {
              status = curandGenerateUniformDouble(gen, _(this->data_ptr<T>()),
                                                   size());
              if (status != CURAND_STATUS_SUCCESS) {
                TV_THROW_RT_ERR("curand failed", status);
              }
            });
        checkCudaErrors(curandDestroyGenerator(gen));
#else
        TV_THROW_RT_ERR("not implemented");
#endif
      }
    });
    return *this;
  }

  Tensor rand_int_(int begin, int end, int seed = -1) {
    std::random_device rd;
    if (seed == -1){
      seed = rd();
    }
    if (this->device() == -1) {
      using all_int16_32_64_t =
          std::tuple<int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t>;
      using all_int8_t =
          std::tuple<int8_t, uint8_t>;
      bool found = DispatchNoExcept<all_int16_32_64_t>()(dtype_, [&](auto I) {
        using T = TV_DECLTYPE(I);
        TensorView<T> tensor_tv = this->tview<T>();
        std::mt19937 generator(seed);
        std::uniform_int_distribution<T> distr(begin, end);
        for (size_t i = 0; i < this->size(); ++i) {
          tensor_tv[i] = distr(generator);
        }
      });
      if (!found){
        Dispatch<all_int8_t>()(dtype_, [&](auto I) {
          found = true;
          using T = TV_DECLTYPE(I);
          TensorView<T> tensor_tv = this->tview<T>();
          std::mt19937 generator(seed);
          constexpr int kTRange = int32_t(std::numeric_limits<T>::max()) - int32_t(std::numeric_limits<T>::min());
          std::uniform_int_distribution<int32_t> distr(begin, end);
          for (size_t i = 0; i < this->size(); ++i) {
            tensor_tv[i] = T((distr(generator) % kTRange) - int32_t(std::numeric_limits<T>::min()));
          }
        });
      }
    } else {
      TV_THROW_RT_ERR("not implemented");
    }
    return *this;
  }

  Tensor astype(DType dtype, bool use_cpu_copy = false) const {
    if (dtype == dtype_) {
      return this->clone(this->pinned(), use_cpu_copy);
    }
    TV_ASSERT_INVALID_ARG(this->device() == -1, "only support cpu tensor");
    TV_ASSERT_INVALID_ARG(!this->empty(), "can't be used in empty tensor");
    TV_ASSERT_INVALID_ARG(contiguous_, "only support contiguous for now");
    auto tensor = Tensor();
    Dispatch<detail::all_tensor_types_t>()(dtype, [&](auto Idst) {
      using Tdst = TV_DECLTYPE(Idst);
      Dispatch<detail::all_tensor_types_t>()(this->dtype_, [&](auto Icur) {
        using Tcur = TV_DECLTYPE(Icur);
        // if constexpr (std::is_convertible<Tcur, Tdst>::value){
        //   auto ptr = this->data_ptr<Tcur>();
        //   tensor =
        //       Tensor(this->shape_, this->stride_, dtype, this->device(),
        //               this->pinned(), this->storage_->managed());
        //   if constexpr (detail::ConvertTmpType<Tcur, Tdst>::kSpec){
        //     Tdst* tensor_data = tensor.data_ptr<Tdst>();
        //     using TmpType = typename detail::ConvertTmpType<Tcur, Tdst>::type;
        //     for (int i = 0; i < this->size(); ++i){
        //       tensor_data[i] = TmpType(ptr[i]);
        //     }
        //   }else{
        //     std::copy(ptr, ptr + this->size(), tensor.data<Tdst>());
        //   }
        // }else{
        //   TV_THROW_INVALID_ARG("not convertable from",
        //                         type_s<std::decay_t<Tcur>>, "to",
        //                         type_s<std::decay_t<Tdst>>);
        // }
        if_constexpr<std::is_convertible<Tcur, Tdst>::value>(
            [&](auto _) {
              auto ptr = this->data<Tcur>();
              tensor =
                  Tensor(this->shape_, this->stride_, dtype, this->device(),
                         this->pinned(), this->storage_->managed());
              std::copy(ptr, ptr + this->size(), tensor.data<Tdst>());
            },
            [&](auto _) {
              TV_THROW_INVALID_ARG("not convertable from",
                                   type_s<std::decay_t<Tcur>>, "to",
                                   type_s<std::decay_t<Tdst>>);
            });
      });
    });
    return tensor;
  }

protected:
  inline void writable_check() {
    TV_ASSERT_RT_ERR(writeable_,
                     "you cant do non-const operation when not writable");
  }

  template <typename T> inline void template_dtype_check() const {
    if (dtype_ >= custom16 && dtype_ <= custom128) {
      auto dsize = detail::sizeof_dtype(dtype_);
      TV_ASSERT_RT_ERR(dsize == sizeof(T), "expect size", sizeof(T),
                       "but sizeof(dtype_) =", dsize);
    } else {
      TV_ASSERT_RT_ERR(dtype_ == type_v<T>, "expect", type_s<T>,
                       "but dtype_ =", dtype_str(dtype_));
    }
  }

  DType dtype_;
  std::shared_ptr<detail::TensorStorage<uint8_t>> storage_;
  TensorShape shape_;
  size_t offset_ = 0;
  TensorShape stride_;

private:
  bool writeable_ = true;
  bool contiguous_ = true;
};

template <typename Os> Os &operator<<(Os &os, const Tensor &tensor) {
  TV_ASSERT_INVALID_ARG(tensor.device() == -1 ||
                            (tensor.device() == 0 && tensor.managed()),
                        "must be cpu tensor");
  Dispatch<detail::all_tensor_types_t>()(tensor.dtype(), [&](auto I) {
    using T = TV_DECLTYPE(I);
    std::stringstream ss;
    if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
      ss << std::setprecision(4);
    }
#if defined(TV_CUDA) && CUDA_VERSION < 11000
    if_constexpr<std::is_same<T, __half>::value>(
        [&](auto _) {
          auto tensorf = tensor.astype(float32);
          auto tview =
              tensorf.tview<const float, -1, DefaultPtrTraits, int64_t>();
          os << tview.repr(ss);
        },
        [&](auto _) {
          auto tview = tensor.tview<const T, -1, DefaultPtrTraits, int64_t>();
          os << tview.repr(ss);
        });
#elif defined(TV_CUDA) && CUDA_VERSION >= 11000
    if_constexpr<std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value>([&](auto _){
      auto tensorf = tensor.astype(float32);
      auto tview = tensorf.tview<const float, -1, DefaultPtrTraits, int64_t>();
      os << tview.repr(ss);
    }, [&](auto _){
      auto tview = tensor.tview<const T, -1, DefaultPtrTraits, int64_t>();
      os << tview.repr(ss);
    });
#else 
    auto tview = tensor.tview<const T, -1, DefaultPtrTraits, int64_t>();
    os << tview.repr(ss);
#endif
  });
  return os;
}

inline Tensor from_blob(void *ptr, TensorShape shape, DType dtype,
                        int device = -1) {
  return Tensor(ptr, shape, dtype, device);
}

inline Tensor from_blob(const void *ptr, TensorShape shape, DType dtype,
                        int device = -1) {
  return Tensor(ptr, shape, dtype, device);
}

inline Tensor empty(TensorShape shape, DType dtype, int device = -1,
                    bool pinned = false, bool managed = false) {
  return Tensor(shape, dtype, device, pinned, managed);
}

inline Tensor zeros(TensorShape shape, DType dtype, int device = -1,
                    bool pinned = false, bool managed = false) {
  return Tensor(shape, dtype, device, pinned, managed).zero_();
}

inline Tensor zeros_managed(TensorShape shape, DType dtype) {
  return Tensor(shape, dtype, 0, true, true).zero_();
}

inline Tensor full(TensorShape shape, int val, DType dtype, int device = -1,
                   bool pinned = false, bool managed = false) {
  return Tensor(shape, dtype, device, pinned, managed).fill_(val);
}

inline Tensor full(TensorShape shape, float val, DType dtype, int device = -1,
                   bool pinned = false, bool managed = false) {
  return Tensor(shape, dtype, device, pinned, managed).fill_(val);
}

inline Tensor cat_first_axis(std::vector<Tensor> tensors) {
  TV_ASSERT_RT_ERR(tensors.size() > 0, "error");
  int first_shape = 0;
  TensorShape remain_shape = tensors[0].shape();
  auto dtype = tensors[0].dtype();
  auto ndim = tensors[0].ndim();
  for (auto &t : tensors) {
    first_shape += t.dim(0);
    TV_ASSERT_RT_ERR(t.dtype() == dtype, "error");
    TV_ASSERT_RT_ERR(t.ndim() == ndim, "error");
    for (int i = 1; i < ndim; ++i) {
      TV_ASSERT_RT_ERR(t.dim(i) == remain_shape[i], "error");
    }
  }
  remain_shape[0] = first_shape;
  Tensor res(remain_shape, tensors[0].dtype(), tensors[0].device());
  int count = 0;
  for (auto &t : tensors) {
    if (t.dim(0) == 0) {
      continue;
    }
    res.slice_first_axis(count, count + t.dim(0)).copy_(t);
    count += t.dim(0);
  }
  return res;
}

} // namespace tv