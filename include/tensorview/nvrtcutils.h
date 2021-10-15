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
#include <tensorview/core/common.h>
#include <tensorview/cuda/driverops.h>
#include <unordered_map>
#include <vector>

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
               std::string program_name = "kernel.cu")
      : code_(code), headers_(headers), program_name_(program_name) {
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
    TV_NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog_,                // prog
                                          code_.c_str(),         // buffer
                                          program_name_.c_str(), // name
                                          headers_.size(),       // numHeaders
                                          header_ptr,            // headers
                                          header_name_ptr));     // includeNames
    nvrtcResult compileResult =
        nvrtcCompileProgram(prog_,             // prog
                            opts.size(),       // numOptions
                            opts_ptrs.data()); // options

    size_t logSize;
    TV_NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog_, &logSize));
    std::string log(logSize, '0');
    TV_NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog_, log.data()));
    tv::ssprint(log);
    TV_ASSERT_RT_ERR(compileResult == NVRTC_SUCCESS, "nvrtc compile failed.");
  }
  static std::shared_ptr<NVRTCProgram> create(std::string code,
               std::unordered_map<std::string, std::string> headers = {},
               std::vector<std::string> opts = {},
               std::string program_name = "kernel.cu"){
    return std::make_shared<NVRTCProgram>(code, headers, opts, program_name);
  }
  ~NVRTCProgram() {
    if (prog_) {
      nvrtcDestroyProgram(&prog_);
    }
  }

  std::string ptx() const {
    size_t ptxSize;
    TV_NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog_, &ptxSize));
    std::string ptx(ptxSize, '0');
    TV_NVRTC_SAFE_CALL(nvrtcGetPTX(prog_, ptx.data()));
    return ptx;
  }

private:
  nvrtcProgram prog_ = nullptr;
  std::string code_;
  std::unordered_map<std::string, std::string> headers_;
  std::string program_name_;
};

class NVRTCModule {
public:
  NVRTCModule(std::shared_ptr<NVRTCProgram> program)
      : program_(program), module_(nullptr) {
    TV_ASSERT_RT_ERR(program, "program ptr must not empty");
  }

  NVRTCModule &load() {
    if (module_ != nullptr) {
      TV_THROW_RT_ERR("this module is already compiled");
    }
    auto ptx = program_->ptx();
    checkCudaErrors(cuModuleLoadDataEx(&module_, ptx.data(), 0, 0, 0));
    return *this;
  }

  CUfunction kernel(std::string name) {
    TV_ASSERT_RT_ERR(module_ != nullptr, "moculde must be loaded");
    CUfunction k = nullptr;
    checkCudaErrors(cuModuleGetFunction(&k, module_, name.c_str()));
    return k;
  }

  ~NVRTCModule() {
    if (module_ != nullptr) {
      cuModuleUnload(module_);
    }
  }

private:
  CUmodule module_ = nullptr;
  std::shared_ptr<NVRTCProgram> program_;
};

} // namespace tv