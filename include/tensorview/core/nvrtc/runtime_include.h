#pragma once 
#include <tensorview/core/defs.h>

#if defined(TV_CUDA_CC)
#ifndef __CUDACC_RTC__
#include <cuda_runtime_api.h>
#endif
#endif