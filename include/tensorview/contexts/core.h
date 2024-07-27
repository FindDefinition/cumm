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
#include <array>
#include <memory>
#include <tensorview/core/common.h>
#include <tensorview/core/defs.h>
#include <tensorview/cuda/driverops.h>
#include <unordered_map>
#if defined(TV_HARDWARE_ACC_CUDA)
#include <cuda.h>
#endif
#ifdef TV_HARDWARE_ACC_METAL
#include "Metal/Metal.hpp"
#endif

namespace tv {

enum class ContextType : int { kCudaStream = 1, kCublasLt = 2, kCudnn = 3, kAppleMetal = 4 };

namespace detail {
#ifdef TV_HARDWARE_ACC_METAL
template <class T>
std::unique_ptr<T, void (*)(T *)> make_apple_mtl_ptr(T *ptr) {
  return std::unique_ptr<T, void (*)(T *)>(ptr, [](T *ptr) {
    if (ptr) {
      ptr->release();
    }
  });
}

#endif 

}

#ifdef TV_HARDWARE_ACC_METAL

struct AppleMetalContext {
    MTL::Device* device_ptr_ = nullptr;
    // The command queue used to pass commands to the device.
    MTL::CommandQueue* command_queue_ptr_ = nullptr;
    AppleMetalContext() {
        device_ptr_ = MTL::CreateSystemDefaultDevice();
        TV_ASSERT_RT_ERR(device_ptr_, "Metal device not found");
        command_queue_ptr_ = device_ptr_->newCommandQueue();
        TV_ASSERT_RT_ERR(command_queue_ptr_, "Metal command queue not found");
    }
    ~AppleMetalContext() {
        if (command_queue_ptr_) {
            command_queue_ptr_->release();
            command_queue_ptr_ = nullptr;
        }
        if (device_ptr_) {
            device_ptr_->release();
            device_ptr_ = nullptr;
        }
    }
};

#endif

namespace detail {

struct ContextValue {
  std::uintptr_t ptr_int;
  bool from_blob;
};

struct ContextManager {
  // windows nvcc have bug when use std::function
  std::uintptr_t (*creater)();
  void (*deleter)(std::uintptr_t);
  // std::function<void(std::uintptr_t)> deleter;
};

struct ContextTypeHash {
  template <typename T> int operator()(T t) const {
    return static_cast<int>(t);
  }
};

struct ContextCore {
private:
  std::unordered_map<ContextType, ContextValue, ContextTypeHash> ctx_ptrs_;
  std::unordered_map<ContextType, ContextManager, ContextTypeHash> ctx_mgrs_;

public:
  ContextCore() {
#if defined(TV_HARDWARE_ACC_CUDA)
    ContextManager stream_mgr;
    stream_mgr.creater = []() -> std::uintptr_t {
      cudaStream_t stream_cuda;
      checkCudaErrors(cudaStreamCreate(&stream_cuda));
      return reinterpret_cast<std::uintptr_t>(stream_cuda);
    };
    stream_mgr.deleter = [](std::uintptr_t ptr_int) {
      cudaStreamDestroy(reinterpret_cast<cudaStream_t>(ptr_int));
    };
    ctx_mgrs_[ContextType::kCudaStream] = stream_mgr;
#endif
#ifdef TV_HARDWARE_ACC_METAL
    ContextManager mtl_compute_ctx_mgr;
    mtl_compute_ctx_mgr.creater = []() -> std::uintptr_t {
      auto* ptr = new AppleMetalContext();
      return reinterpret_cast<std::uintptr_t>(ptr);
    };
    mtl_compute_ctx_mgr.deleter = [](std::uintptr_t ptr_int) {
      auto* ptr = reinterpret_cast<AppleMetalContext*>(ptr_int);
      delete ptr;
    };
    ctx_mgrs_[ContextType::kAppleMetal] = mtl_compute_ctx_mgr;
#endif
  }
  void register_manager(ContextType type, ContextManager mgr) {
    TV_ASSERT_INVALID_ARG(ctx_mgrs_.find(type) == ctx_mgrs_.end(),
                          "manager exists");
    ctx_mgrs_[type] = mgr;
  }

  void create_item(ContextType type) {
    TV_ASSERT_RT_ERR(ctx_ptrs_.find(type) == ctx_ptrs_.end(),
                     "context item exists");
    TV_ASSERT_RT_ERR(
        ctx_mgrs_.find(type) != ctx_mgrs_.end(),
        "can't find context manager. call register_manager before.");

    ctx_ptrs_[type] = ContextValue{ctx_mgrs_[type].creater(), false};
  }

  void create_raw_item(ContextType type, std::uintptr_t handle) {
    TV_ASSERT_RT_ERR(ctx_ptrs_.find(type) == ctx_ptrs_.end(),
                     "context item exists");
    ctx_ptrs_[type] = ContextValue{handle, true};
  }

  bool has_item(ContextType type) {
    return ctx_ptrs_.find(type) != ctx_ptrs_.end();
  }

  std::uintptr_t get_item(ContextType type) {
    auto ptr = ctx_ptrs_.find(type);
    if (ptr != ctx_ptrs_.end()) {
      return ptr->second.ptr_int;
    }
    return 0;
  }

  ~ContextCore() {
    for (auto &pair : ctx_ptrs_) {
      if (!pair.second.from_blob) {
        ctx_mgrs_[pair.first].deleter(pair.second.ptr_int);
      }
    }
  }
};

} // namespace detail

struct Context {
protected:
  std::shared_ptr<detail::ContextCore> context_ptr_;
  void check_ptr_valid() {
    TV_ASSERT_RT_ERR(bool(context_ptr_), "context ptr must not empty");
  }

public:
  Context() : context_ptr_(std::make_shared<detail::ContextCore>()) {}

  bool has_cuda_stream() {
    check_ptr_valid();
    return context_ptr_->has_item(ContextType::kCudaStream);
  }

  Context &create_cuda_stream() {
    check_ptr_valid();
    context_ptr_->create_item(ContextType::kCudaStream);
    return *this;
  }

#if defined(TV_HARDWARE_ACC_CUDA)
  Context &set_cuda_stream(cudaStream_t stream) {
    check_ptr_valid();
    context_ptr_->create_raw_item(ContextType::kCudaStream,
                                  reinterpret_cast<std::uintptr_t>(stream));
    return *this;
  }
  cudaStream_t cuda_stream() {
    check_ptr_valid();
    return reinterpret_cast<cudaStream_t>(cuda_stream_int());
  }
#endif

  Context &set_cuda_stream_int(std::uintptr_t stream) {
#if defined(TV_HARDWARE_ACC_CUDA)
    check_ptr_valid();
    context_ptr_->create_raw_item(ContextType::kCudaStream,
                                  stream);
#endif
    return *this;
  }
  void synchronize_stream() {
#if defined(TV_HARDWARE_ACC_CUDA)
    check_ptr_valid();
    checkCudaErrors(cudaStreamSynchronize(cuda_stream()));
#endif
  }
  void synchronize() {
#if defined(TV_HARDWARE_ACC_CUDA)
    check_ptr_valid();
    checkCudaErrors(cudaStreamSynchronize(cuda_stream()));
#else 
#ifdef TV_HARDWARE_ACC_METAL
    check_ptr_valid();
    if (has_item(ContextType::kAppleMetal)) {
      auto* ptr = reinterpret_cast<AppleMetalContext*>(context_ptr_->get_item(ContextType::kAppleMetal));
      auto cb = detail::make_apple_mtl_ptr(ptr->command_queue_ptr_->commandBuffer());
      TV_ASSERT_INVALID_ARG(cb, "command buffer is null");
      cb->commit();
      cb->waitUntilCompleted();
    }
#endif
#endif
  }

  std::uintptr_t cuda_stream_int() {
    check_ptr_valid();
#if defined(TV_HARDWARE_ACC_CUDA)
    return reinterpret_cast<std::uintptr_t>(cuda_stream());
#else 
    return 0;
#endif 
  }
  bool has_item(ContextType type) {
    check_ptr_valid();
    return context_ptr_->has_item(type);
  }
  std::uintptr_t get_item(ContextType type) {
    check_ptr_valid();
    return context_ptr_->get_item(type);
  }
  Context &create_apple_metal_context() {
#ifdef TV_HARDWARE_ACC_METAL

    check_ptr_valid();
    context_ptr_->create_item(ContextType::kAppleMetal);
    return *this;
#else 
    TV_THROW_INVALID_ARG("Apple Metal is not supported in non-apple platform");
#endif
  }


};
} // namespace tv