# Copyright 2022 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import dataclasses
from typing import List

@dataclasses.dataclass
class MathItem:
    name: str 
    f16name: str = ""
    f16vecname: str = ""


def main():
    items: List[MathItem] = [
        MathItem("sqrt", "hsqrt", "h2sqrt"),
        MathItem("rsqrt", "hrsqrt", "h2rsqrt"),
        MathItem("ceil", "hceil", "h2ceil"),
        MathItem("cos", "hcos", "h2cos"),
        MathItem("exp", "hexp", "h2exp"),
        MathItem("exp10", "hexp10", "h2exp10"),
        MathItem("exp2", "hexp2", "h2exp2"),
        MathItem("floor", "hfloor", "h2floor"),
        MathItem("log", "hlog", "h2log"),
        MathItem("log10", "hlog10", "h2log10"),
        MathItem("log2", "hlog2", "h2log2"),
        MathItem("rint", "hrint", "h2rint"),
        MathItem("sin", "hsin", "h2sin"),
        MathItem("trunc", "htrunc", "h2trunc"),

        MathItem("fabs", "__habs", "__habs2"),


        MathItem("tan"),
        MathItem("asin"),
        MathItem("acos"),
        MathItem("atan"),
        MathItem("round"),
        MathItem("sinh"),
        MathItem("cosh"),
        MathItem("tanh"),
        MathItem("asinh"),
        MathItem("acosh"),
        MathItem("atanh"),

    ]
    code_cpus = []
    code_cuda_fps = []
    code_cuda_hfs = []
    code_cuda_bfs = []
    code_cuda_hf2s = []
    code_cuda_bf2s = []
    no_cpu_vers = set(["rsqrt", "exp10"])
    for item in items:
        if item.name not in no_cpu_vers:
            code_cpus.append(f"""
    static T {item.name}(T x){{
        return std::{item.name}(x);
    }}
            """)

        code_cuda_fps.append(f"""
TV_HOST_DEVICE_INLINE static float {item.name}(float x){{
    return {item.name}f(x);
}}
        """)
        if item.f16name:
            code_cuda_hfs.append(f"""
    TV_DEVICE_INLINE static __half {item.name}(__half x){{
        return {item.f16name}(x);
    }}
            """)
            code_cuda_bfs.append(f"""
    TV_DEVICE_INLINE static __nv_bfloat16 {item.name}(__nv_bfloat16 x){{
        return {item.f16name}(x);
    }}
            """)
        if item.f16vecname:
            code_cuda_hf2s.append(f"""
    TV_DEVICE_INLINE static __half2 v2{item.name}(__half2 x){{
        return {item.f16vecname}(x);
    }}
            """)
            code_cuda_bf2s.append(f"""
    TV_DEVICE_INLINE static __nv_bfloat162 v2{item.name}(__nv_bfloat162 x){{
        return {item.f16vecname}(x);
    }}
            """)

    code_cpu = "\n".join(code_cpus)
    code_cuda_fp = "\n".join(code_cuda_fps)
    code_cuda_hf = "\n".join(code_cuda_hfs)
    code_cuda_hf2 = "\n".join(code_cuda_hf2s)
    code_cuda_bf = "\n".join(code_cuda_bfs)
    code_cuda_bf2 = "\n".join(code_cuda_bf2s)

    code = f"""
template <typename T>
struct MathScalarOp{{
#ifndef __CUDACC_RTC__
static T copysign(T x, T y){{
    return std::copysign(x, y);
}}

static T atan2(T y, T x){{
    return std::atan2(y, x);
}}

static T scalbn(T x, int n){{
    return std::scalbn(x, n);
}}

static T pow(T x, T n){{
    return std::pow(x, n);
}}

static T fmod(T x, T n){{
    return std::fmod(x, n);
}}

{code_cpu}
#endif
}};

#ifdef TV_CUDA_CC
template <>
struct MathScalarOp<float>{{

TV_HOST_DEVICE_INLINE static float copysign(float x, float y){{
    return copysignf(x, y);
}}

TV_HOST_DEVICE_INLINE static float atan2(float y, float x){{
    return atan2f(y, x);
}}

TV_HOST_DEVICE_INLINE static float scalbn(float x, int n){{
    return scalbnf(x, n);
}}

TV_HOST_DEVICE_INLINE static float pow(float x, float n){{
    return powf(x, n);
}}

TV_HOST_DEVICE_INLINE static float fmod(float x, float n){{
    return fmodf(x, n);
}}

{code_cuda_fp}
}};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
template <>
struct MathScalarOp<__half>{{
{code_cuda_hf}
{code_cuda_hf2}
}};
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
template <>
struct MathScalarOp<__nv_bfloat16>{{
{code_cuda_bf}
{code_cuda_bf2}
}};
#endif

#endif


    """
    print(code)

if __name__ == "__main__":

    main()