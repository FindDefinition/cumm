/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#define CUDA_LIB_NAME "cuda"

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif // defined(WIN32_LEAN_AND_MEAN)
#include <windows.h>
#define dllOpen(name) (void *)LoadLibraryA("nv" name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name)                                                \
  GetProcAddress(static_cast<HMODULE>(handle), name)
#else
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so.1", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif

#include <cuda.h>
#include <tensorview/core/common.h>

namespace tv {

class CUDADriverWrapper {
public:
  CUDADriverWrapper() {
    handle = dllOpen(CUDA_LIB_NAME);
    TV_ASSERT_RT_ERR(handle != nullptr, "load CUDA Driver library failed!!! "
                                        "You must install cuda driver first.");

    auto load_sym = [](void *handle, const char *name) {
      void *ret = dllGetSym(handle, name);
      TV_ASSERT_RT_ERR(ret != nullptr, "load", name,
                       "from CUDA Driver library failed!!!");
      return ret;
    };

    *(void **)(&_cuModuleUnload) = load_sym(handle, "cuModuleUnload");
    *(void **)(&_cuModuleLoadDataEx) = load_sym(handle, "cuModuleLoadDataEx");
    *(void **)(&_cuModuleGetFunction) = load_sym(handle, "cuModuleGetFunction");
    *(void **)(&_cuLaunchKernel) = load_sym(handle, "cuLaunchKernel");
  }

  ~CUDADriverWrapper() {
    if (handle) {
      dllClose(handle);
    }
  }

  // Delete default copy constructor and copy assignment constructor
  CUDADriverWrapper(const CUDADriverWrapper &) = delete;
  CUDADriverWrapper &operator=(const CUDADriverWrapper &) = delete;

  CUresult cuModuleUnload(CUmodule hmod) const {
    return (*_cuModuleUnload)(hmod);
  }
  CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                              unsigned int numOptions, CUjit_option *options,
                              void **optionValues) const {
    return (*_cuModuleLoadDataEx)(module, image, numOptions, options,
                                  optionValues);
  }
  CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                               const char *name) const {
    return (*_cuModuleGetFunction)(hfunc, hmod, name);
  }
  CUresult cuLaunchKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
                          uint32_t gridDimZ, uint32_t blockDimX,
                          uint32_t blockDimY, uint32_t blockDimZ,
                          uint32_t sharedMemBytes, CUstream hStream,
                          void **kernelParams, void **extra) const {
    return (*_cuLaunchKernel)(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                              blockDimY, blockDimZ, sharedMemBytes, hStream,
                              kernelParams, extra);
  }

private:
  void *handle;
  CUresult (*_cuModuleUnload)(CUmodule);
  CUresult (*_cuModuleLoadDataEx)(CUmodule *module, const void *image,
                                  unsigned int numOptions,
                                  CUjit_option *options, void **optionValues);
  CUresult (*_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
  CUresult (*_cuLaunchKernel)(CUfunction f, uint32_t gridDimX,
                              uint32_t gridDimY, uint32_t gridDimZ,
                              uint32_t blockDimX, uint32_t blockDimY,
                              uint32_t blockDimZ, uint32_t sharedMemBytes,
                              CUstream hStream, void **kernelParams,
                              void **extra);
};

} // namespace tv
