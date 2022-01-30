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

#include <cuda.h>
#include <memory>
#include <nvrtc.h>
#include <string>
#include <tensorview/tensor.h>
#include <unordered_map>
#include <vector>
#ifdef TV_CUDA
#include <tensorview/cuda/driver.h>
#endif

#define TV_NVRTC_SAFE_CALL(x)                                                  \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      TV_THROW_RT_ERR("error: " #x " failed with error ",                      \
                      nvrtcGetErrorString(result));                            \
    }                                                                          \
  } while (0);

namespace tv {

class NVRTCProgram {
public:
  NVRTCProgram(std::string code,
               std::unordered_map<std::string, std::string> headers = {},
               std::vector<std::string> opts = {},
               std::string program_name = "kernel.cu",
               std::vector<std::string> name_exprs = {})
      : code_(code), headers_(headers), program_name_(program_name),
        name_exprs_(name_exprs) {
    std::vector<const char *> header_buffers;
    std::vector<const char *> header_names;
    std::vector<const char *> opts_ptrs;
    for (auto &opt : opts) {
      opts_ptrs.push_back(opt.c_str());
    }
    for (auto &pair : headers_) {
      header_names.push_back(pair.first.c_str());
      header_buffers.push_back(pair.second.c_str());
    }
    const char *const *header_ptr = nullptr;
    const char *const *header_name_ptr = nullptr;
    if (headers_.size() > 0) {
      header_ptr = header_buffers.data();
      header_name_ptr = header_names.data();
    }
#ifdef TV_CUDA
    TV_NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog_,                // prog
                                          code_.c_str(),         // buffer
                                          program_name_.c_str(), // name
                                          headers_.size(),       // numHeaders
                                          header_ptr,            // headers
                                          header_name_ptr));     // includeNames
    for (size_t i = 0; i < name_exprs_.size(); ++i) {
      TV_NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog_, name_exprs_[i].c_str()));
    }

    nvrtcResult compileResult =
        nvrtcCompileProgram(prog_,             // prog
                            opts.size(),       // numOptions
                            opts_ptrs.data()); // options

    size_t logSize;
    TV_NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog_, &logSize));
    std::string log(logSize, '0');
    TV_NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog_, &log[0]));
    tv::ssprint(log);
    TV_ASSERT_RT_ERR(compileResult == NVRTC_SUCCESS, "nvrtc compile failed.");
#else
    TV_THROW_RT_ERR("you must compile with CUDA first to use nvrtc program");
#endif
  }
  static std::shared_ptr<NVRTCProgram>
  create(std::string code,
         std::unordered_map<std::string, std::string> headers = {},
         std::vector<std::string> opts = {},
         std::string program_name = "kernel.cu",
         std::vector<std::string> name_exprs = {}) {
    return std::make_shared<NVRTCProgram>(code, headers, opts, program_name, name_exprs);
  }
  ~NVRTCProgram() {
#ifdef TV_CUDA
    if (prog_) {
      nvrtcDestroyProgram(&prog_);
    }
#endif
  }

  std::string ptx() const {
#ifdef TV_CUDA

    size_t ptxSize;
    TV_NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog_, &ptxSize));
    std::string ptx(ptxSize, '0');
    TV_NVRTC_SAFE_CALL(nvrtcGetPTX(prog_, &ptx[0]));

    return ptx;
#else
    return "";
#endif
  }

  std::string get_lowered_name(std::string name) {
#ifdef TV_CUDA
    const char* lowered_name;
    TV_NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog_,
                                        name.c_str(), // name expression
                                        &lowered_name // lowered name
                                        ));
    return std::string(lowered_name);
#else
    return "";
#endif
  }

private:
#ifdef TV_CUDA
  nvrtcProgram prog_ = nullptr;
#else
  void *prog_ = nullptr;
#endif
  std::string code_;
  std::unordered_map<std::string, std::string> headers_;
  std::string program_name_;
  std::vector<std::string> name_exprs_;
};

class NVRTCModule {
public:
  enum ArgType { kTensor = 0, kArray = 1, kScalar = 2 };

  NVRTCModule(std::shared_ptr<NVRTCProgram> program)
      : program_(program), module_(nullptr) {
    TV_ASSERT_RT_ERR(program, "program ptr must not empty");
#ifndef TV_CUDA
    TV_THROW_RT_ERR("you must compile with CUDA first to use NVRTCModule");
#endif
  }
  static std::shared_ptr<NVRTCModule>
  create(std::string code,
         std::unordered_map<std::string, std::string> headers = {},
         std::vector<std::string> opts = {},
         std::string program_name = "kernel.cu",
         std::vector<std::string> name_exprs = {}) {
    return std::make_shared<NVRTCModule>(
        NVRTCProgram::create(code, headers, opts, program_name, name_exprs));
  }

  NVRTCModule &load() {
#ifdef TV_CUDA
    if (module_ != nullptr) {
      TV_THROW_RT_ERR("this module is already compiled");
    }
    auto ptx = program_->ptx();
    TV_CUDA_RESULT_CHECK(
        wrapper_.cuModuleLoadDataEx(&module_, ptx.data(), 0, 0, 0));
#endif
    return *this;
  }
#ifdef TV_CUDA

  CUfunction kernel(std::string name) {

    TV_ASSERT_RT_ERR(module_ != nullptr, "moculde must be loaded");
    CUfunction k = nullptr;
    TV_CUDA_RESULT_CHECK(
        wrapper_.cuModuleGetFunction(&k, module_, name.c_str()));
    return k;
  }
#endif
  std::shared_ptr<NVRTCProgram> get_program() { return program_; }

  std::string get_lowered_name(std::string name) {
    TV_ASSERT_RT_ERR(program_ != nullptr, "program_ must not empty");
    return program_->get_lowered_name(name);
  }

  void run_kernel(std::string name, std::array<int, 3> blocks,
                  std::array<int, 3> threads, int smem_size,
                  std::uintptr_t stream_int,
                  std::vector<std::tuple<tv::Tensor, int>> args) {
#ifdef TV_CUDA
    if (module_ == nullptr) {
      load();
    }
    CUstream stream = reinterpret_cast<CUstream>(stream_int);
    std::vector<void *> params;
    std::vector<void *> tensor_ptrs;

    for (auto &arg : args) {
      auto &ten = std::get<0>(arg);
      auto arg_type = std::get<1>(arg);
      switch (arg_type) {
      case ArgType::kTensor: {
        TV_ASSERT_INVALID_ARG(ten.device() == 0, "tensor must be GPU");
        tensor_ptrs.push_back(ten.raw_data());
        params.push_back(tensor_ptrs.data() + tensor_ptrs.size() - 1);
        break;
      }
      case ArgType::kArray: {
        TV_ASSERT_INVALID_ARG(ten.device() == -1, "array tensor must be CPU");
        params.push_back(ten.raw_data());
        break;
      }
      case ArgType::kScalar: {
        TV_ASSERT_INVALID_ARG(ten.device() == -1, "array tensor must be CPU");
        params.push_back(ten.raw_data());
        break;
      }
      default:
        TV_THROW_RT_ERR("not implemented");
      }
    }
    TV_CUDA_RESULT_CHECK(wrapper_.cuLaunchKernel(
        kernel(name), blocks[0], blocks[1], blocks[2], threads[0], threads[1],
        threads[2], smem_size, stream, params.data(), 0));
#endif
  }

  ~NVRTCModule() {
#ifdef TV_CUDA
    if (module_ != nullptr) {
      wrapper_.cuModuleUnload(module_);
    }
#endif
  }

private:
  std::shared_ptr<NVRTCProgram> program_;
#ifdef TV_CUDA
  CUmodule module_ = nullptr;
  CUDADriverWrapper wrapper_;
#else
  void *module_ = nullptr;
#endif
};

} // namespace tv