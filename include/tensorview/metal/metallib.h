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

#include <memory>
#include <string>
#include <tensorview/tensor.h>

#include <unordered_map>
#include <vector>
#ifdef TV_HARDWARE_ACC_METAL
#include "Metal/Metal.hpp"
#endif
#include <tensorview/cuda/nvrtc.h>

namespace tv {
namespace detail {
#ifdef TV_HARDWARE_ACC_METAL
inline MTL::DataType tv_dtype_to_mtl_data_type(tv::DType t) {
  switch (t) {
  case DType::bool_:
    return MTL::DataType::DataTypeBool;
  case DType::float32:
    return MTL::DataType::DataTypeFloat;
  case DType::int8:
    return MTL::DataType::DataTypeChar;
  case DType::int16:
    return MTL::DataType::DataTypeShort;
  case DType::int32:
    return MTL::DataType::DataTypeInt;
  case DType::int64:
    return MTL::DataType::DataTypeLong;
  case DType::uint8:
    return MTL::DataType::DataTypeUChar;
  case DType::uint16:
    return MTL::DataType::DataTypeUShort;
  case DType::uint32:
    return MTL::DataType::DataTypeUInt;
  case DType::uint64:
    return MTL::DataType::DataTypeULong;
  case DType::float16:
    return MTL::DataType::DataTypeHalf;
  case DType::bfloat16:
    return MTL::DataType::DataTypeBFloat;

  default:
    TV_THROW_INVALID_ARG("not implemented dtype: ", dtype_str(t));
  }
}

#endif
} // namespace detail
class MetalModule {
public:
  using ArgType = NVRTCModule::ArgType;
  MetalModule(tv::Tensor binary) {
#ifdef TV_HARDWARE_ACC_METAL
    ptr_mtl_device_ = MTL::CreateSystemDefaultDevice();
    auto option =
        detail::make_apple_mtl_ptr(MTL::CompileOptions::alloc()->init());
    auto data = dispatch_data_create(binary.const_raw_data(), binary.raw_size(),
                                     dispatch_get_main_queue(),
                                     DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NS::Error *error = nullptr;
    mtl_lib_ = ptr_mtl_device_->newLibrary(data, &error);
#else
    TV_THROW_RT_ERR("you must compile with apple first to use metal program");
#endif
  }
  MetalModule(std::string code, std::vector<std::string> opts) {
#ifdef TV_HARDWARE_ACC_METAL
    ptr_mtl_device_ = MTL::CreateSystemDefaultDevice();
    auto codeNS = detail::make_apple_mtl_ptr(
        NS::String::string(code.c_str(), NS::ASCIIStringEncoding));
    auto option =
        detail::make_apple_mtl_ptr(MTL::CompileOptions::alloc()->init());
    NS::Error *error = nullptr;

    mtl_lib_ = ptr_mtl_device_->newLibrary(codeNS.get(), option.get(), &error);
#else
    TV_THROW_RT_ERR("you must compile with apple first to use metal program");
#endif
  }
  void run_kernel(std::string name, std::array<int, 3> blocks,
                  std::array<int, 3> threads, int smem_size, Context ctx,
                  std::vector<std::tuple<tv::Tensor, int>> args) {
#ifdef TV_HARDWARE_ACC_METAL
    bool sync_op = false;
    if (!ctx.has_item(ContextType::kAppleMetal)) {
      ctx = Context().create_apple_metal_context();
      sync_op = true;
    }
    TV_ASSERT_INVALID_ARG(
        ctx.has_item(ContextType::kAppleMetal),
        "you must use a context with metal created explicitly");
    auto *cmd_queue_ptr = reinterpret_cast<AppleMetalContext *>(
                              ctx.get_item(ContextType::kAppleMetal))
                              ->command_queue_ptr_;
    auto cb = detail::make_apple_mtl_ptr(cmd_queue_ptr->commandBuffer());
    TV_ASSERT_RT_ERR(cb, "command buffer is null");
    auto computeEncoder =
        std::unique_ptr<MTL::ComputeCommandEncoder, void(*)(MTL::ComputeCommandEncoder*)>(cb->computeCommandEncoder(), [](MTL::ComputeCommandEncoder *p) {
          p->endEncoding();
          p->release();
        });
    TV_ASSERT_RT_ERR(computeEncoder, "compute encoder is null");
    auto nameNS = detail::make_apple_mtl_ptr(
        NS::String::string(name.c_str(), NS::ASCIIStringEncoding));

    auto constants = detail::make_apple_mtl_ptr(
        MTL::FunctionConstantValues::alloc()->init());
    TV_ASSERT_RT_ERR(constants, "command buffer is null");
    int cnt = 0;
    int buffer_cnt = 0;
    for (auto &arg : args) {
      auto &ten = std::get<0>(arg);
      auto arg_type = std::get<1>(arg);
      // tv::ssprint(ten.shape(), ten.dtype(), arg_type);
      switch (arg_type) {
      case ArgType::kTensor: {
        if (ten.empty()) {
        } else {
          TV_ASSERT_INVALID_ARG(ten.device() == 0, "tensor must be GPU");
          computeEncoder->setBuffer(
              reinterpret_cast<MTL::Buffer *>(
                  ten.storage()->apple_metal_buffer_ptr()),
              ten.byte_offset(), buffer_cnt);
        }
        buffer_cnt += 1;
        break;
      }
      case ArgType::kArray: {
        TV_ASSERT_INVALID_ARG(ten.device() == -1, "array tensor must be CPU");
        // const check is performed in python
        computeEncoder->setBuffer(
            reinterpret_cast<MTL::Buffer *>(
                ten.storage()->apple_metal_buffer_ptr()),
            ten.byte_offset(), buffer_cnt);
        buffer_cnt += 1;
        break;
      }
      case ArgType::kScalar: {
        TV_ASSERT_INVALID_ARG(ten.device() == -1 && ten.size() == 1, "array tensor must be CPU and scalar");
        auto mtl_dtype = detail::tv_dtype_to_mtl_data_type(ten.dtype());
        constants->setConstantValue(ten.raw_data(), mtl_dtype, cnt);
        cnt += 1;
        break;
      }
      case ArgType::kTensorView: {
        TV_THROW_INVALID_ARG("not implemented");
        break;
      }
      default:
        TV_THROW_RT_ERR("not implemented");
      }
    }
    NS::Error *error = nullptr;
    auto computeFunction =
        detail::make_apple_mtl_ptr(mtl_lib_->newFunction(nameNS.get(), constants.get(), &error));
    TV_ASSERT_INVALID_ARG(computeFunction && error == nullptr, "can't find function", name);
    error = nullptr;
    auto func_pso =
        detail::make_apple_mtl_ptr(ptr_mtl_device_->newComputePipelineState(
            computeFunction.get(), &error));
    TV_ASSERT_INVALID_ARG(func_pso && error == nullptr, "can't create pso", name);
    computeEncoder->setComputePipelineState(func_pso.get());

    MTL::Size threadgroupSize = MTL::Size(threads[0], threads[1], threads[2]);
    MTL::Size gridSize = MTL::Size(blocks[0], blocks[1], blocks[2]);
    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();
    // Execute the command.
    cb->commit();
    if (sync_op) {
      cb->waitUntilCompleted();
    }

#endif
  }

  ~MetalModule() {
#ifdef TV_HARDWARE_ACC_METAL
    if (mtl_lib_) {
      mtl_lib_->release();
      mtl_lib_ = nullptr;
    }
    if (ptr_mtl_device_) {
      ptr_mtl_device_->release();
      ptr_mtl_device_ = nullptr;
    }
#endif
  }

private:
#ifdef TV_HARDWARE_ACC_METAL
  MTL::Device *ptr_mtl_device_ = nullptr;
  MTL::Library *mtl_lib_ = nullptr;

#endif
};

} // namespace tv
