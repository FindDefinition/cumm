# Copyright 2021 Yan Yan
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

import collections
import importlib.util
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import pccm
from ccimport import compat

from cumm.constants import CUMM_CPU_ONLY_BUILD, TENSORVIEW_INCLUDE_PATH
from cumm.constants import PACKAGE_ROOT

def get_executable_path(executable: str) -> str:
    if compat.InWindows:
        cmd = ["powershell.exe", "(Get-Command {}).Path".format(executable)]
    elif compat.InLinux:
        cmd = ["which", executable]
    else:
        raise NotImplementedError
    try:
        out = subprocess.check_output(cmd)
    except subprocess.CalledProcessError:
        return ""
    return out.decode("utf-8").strip()


_GPU_NAME_TO_ARCH = {
    r"(NVIDIA )?H100": "90",
    r"(NVIDIA )?A100": "80",
    r"(NVIDIA )?RTX A[0-9]000": "86",
    r"(NVIDIA )?TESLA H100": "90",
    r"(NVIDIA )?TESLA A100": "80",
    r"(NVIDIA )?TESLA V100": "70",
    r"(NVIDIA )?TESLA P(100|4|40)": "60",
    r"(NVIDIA )?TESLA T4": "75",
    r"(NVIDIA )?QUADRO RTX A[0-9]000": "86",
    r"(NVIDIA )?QUADRO RTX [0-9]000": "75",
    r"(NVIDIA )?TITAN V": "70",
    r"(NVIDIA )?TITAN Xp": "60",
    r"(NVIDIA )?TITAN X": "52",
    r"(NVIDIA )?GeForce RTX 40[0-9]0( Ti)?": "89",
    r"(NVIDIA )?GeForce RTX 30[0-9]0( Ti)?": "86",
    r"(NVIDIA )?GeForce RTX 20[0-9]0( Ti)?": "75",
    r"(NVIDIA )?GeForce GTX 16[0-9]0( Ti)?":
    "75",  # FIXME GTX 1660 don't have Tensor Core?
    r"(NVIDIA )?GeForce GTX 10[0-9]0( Ti)?": "61",
    r"(NVIDIA )?GeForce GTX 9[0-9]0( Ti)?": "52",
}


def _get_cuda_arch_flags(is_gemm: bool = False) -> Tuple[List[str], List[Tuple[int, int]], bool]:
    r'''
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    '''
    # from pytorch cpp_extension.py
    # Note: keep combined names ("arch1+arch2") above single names, otherwise
    # string replacement may not do the right thing
    named_arches = collections.OrderedDict([
        ('Maxwell', '5.2+PTX'),
        ('Pascal', '6.0;6.1+PTX'),
        ('Volta', '7.0+PTX'),
        ('Turing', '7.5+PTX'),
        ('Ampere', '8.0;8.6+PTX'),
        ('Hopper', '9.0+PTX'),
        ('Lovelace', '8.9+PTX'),
    ])

    supported_arches = [
        '3.5', '3.7', '5.0', '5.2', '6.0', '6.1', '7.0', '7.2', '7.5', '8.0',
        '8.6', '8.9', '9.0'
    ]
    supported_arches += ['5.3', '6.2', '7.2', '8.7']
    valid_arch_strings = supported_arches + [
        s + "+PTX" for s in supported_arches
    ]

    # The default is sm_30 for CUDA 9.x and 10.x
    # First check for an env var (same as used by the main setup.py)
    # Can be one or more architectures, e.g. "6.1" or "3.5;5.2;6.0;6.1;7.0+PTX"
    # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
    _arch_list = os.getenv('CUMM_CUDA_ARCH_LIST', None)
    _cuda_version = os.getenv('CUMM_CUDA_VERSION', None)
    _enable_cross_compile_aarch64 = os.getenv('CUMM_CROSS_COMPILE_TARGET',
                                              None) == "aarch64"

    if _arch_list is not None and _arch_list.lower() == "all":
        msg = ("you must provide CUDA version by CUMM_CUDA_VERSION, "
               "for example, export CUMM_CUDA_VERSION=\"10.2\"")
        assert _cuda_version is not None, msg
        cuda_ver_tuple = _cuda_version.split(".")
        if len(cuda_ver_tuple) == 2:
            major = int(cuda_ver_tuple[0])
            minor = int(cuda_ver_tuple[1])
        else:
            num = int(_cuda_version)
            major = num // 10
            minor = num % 10
        assert (major, minor) >= (10, 2), "we only support cuda >= 10.2"
        if _enable_cross_compile_aarch64:
            # 6.2: TX2, 7.2: Xavier, 8.7: Orin
            if (major, minor) < (11, 5):
                _arch_list = "5.3;6.2;7.2+PTX"
            else:
                _arch_list = "5.3;6.2;7.2;8.7+PTX"
        else:
            if is_gemm:
                if (major, minor) < (11, 0):
                    _arch_list = "3.7;5.0;5.2;6.0;6.1;7.0;7.5+PTX"
                elif (major, minor) < (11, 8):
                    _arch_list = "5.2;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
                elif (major, minor) < (12, 0):
                    _arch_list = "6.0;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
                else:
                    # remove sm < 70 prebuilt gemm kernels in CUDA 12.
                    # these gemm kernels will be compiled via nvrtc.
                    _arch_list = "6.0;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
            else:
                # flag for non-gemm kernels, they are usually simple and small.
                if (major, minor) < (11, 0):
                    _arch_list = "3.5;3.7;5.0;5.2;6.0;6.1;7.0;7.5+PTX"
                elif (major, minor) < (11, 8):
                    _arch_list = "3.5;3.7;5.0;5.2;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
                elif (major, minor) < (12, 0):
                    _arch_list = "5.0;5.2;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
                else:
                    _arch_list = "5.0;5.2;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
    _all_arch = "5.2;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    for named_arch, archval in named_arches.items():
        _all_arch = _all_arch.replace(named_arch, archval)
    _all_arch_list = _all_arch.split(';')

    # If not given, determine what's best for the GPU / CUDA version that can be found
    if not _arch_list:
        arch_list = []
        # the assumption is that the extension should run on any of the currently visible cards,
        # which could be of different types - therefore all archs for visible cards should be included
        gpu_names = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name",
             "--format=csv,noheader"]).decode("utf-8")
        gpu_names = gpu_names.strip()
        gpu_names = gpu_names.split("\n")
        arch_found = True
        for i in range(len(gpu_names)):
            capability = ""
            for reg, sm in _GPU_NAME_TO_ARCH.items():
                # print(reg, gpu_names[i])
                if re.match(reg, gpu_names[i]):
                    capability = sm
            if capability == "":
                arch_list = _all_arch_list
                arch_found = False
                break
            capability_int = int(capability)
            major = capability_int // 10
            minor = capability_int % 10
            arch = f'{major}.{minor}'
            if arch not in arch_list:
                arch_list.append(arch)
        if arch_found:
            arch_list = sorted(arch_list)
            arch_list[-1] += '+PTX'
    else:
        # Deal with lists that are ' ' separated (only deal with ';' after)
        _arch_list = _arch_list.replace(' ', ';')
        # Expand named arches
        for named_arch, archval in named_arches.items():
            _arch_list = _arch_list.replace(named_arch, archval)

        arch_list = _arch_list.split(';')
    flags = []
    nums: List[Tuple[int, int]] = []
    if not arch_list:
        raise ValueError(
            "can't find arch or can't recogize your GPU. "
            "use env CUMM_CUDA_ARCH_LIST to specify your gpu arch. "
            "for example, export CUMM_CUDA_ARCH_LIST=\"8.0;8.6+PTX\"")
    have_ptx = False
    for arch in arch_list:
        if arch.endswith("+PTX"):
            arch_vers = arch.split("+")[0].split(".")
        else:
            arch_vers = arch.split(".")
        if arch not in valid_arch_strings:
            raise ValueError(
                f"Unknown CUDA arch ({arch}) or GPU not supported")
        else:
            num = arch_vers[0] + arch_vers[1]
            nums.append((int(arch_vers[0]), int(arch_vers[1])))
            if arch.endswith('+PTX'):
                have_ptx = True
                flags.append(f'-gencode=arch=compute_{num},code=[sm_{num},compute_{num}]')
            else:
                flags.append(f'-gencode=arch=compute_{num},code=sm_{num}')

                # flags.append(f'-gencode=arch=compute_{num},code=compute_{num}')

    return sorted(list(set(flags))), nums, have_ptx

def get_cuda_version_by_nvcc():
    nvcc_version = subprocess.check_output(["nvcc", "--version"
                                            ]).decode("utf-8").strip()
    nvcc_version_str = nvcc_version.split("\n")[3]
    version_str: str = re.findall(r"release (\d+.\d+)",
                                    nvcc_version_str)[0]
    return version_str

_CACHED_CUDA_INCLUDE_LIB: Optional[Tuple[Path, Path]] = None 

def _get_cuda_include_lib():
    global _CACHED_CUDA_INCLUDE_LIB
    if _CACHED_CUDA_INCLUDE_LIB is None:
        if compat.InWindows:
            nvcc_version = subprocess.check_output(["nvcc", "--version"
                                                    ]).decode("utf-8").strip()
            nvcc_version_str = nvcc_version.split("\n")[3]
            version_str: str = re.findall(r"release (\d+.\d+)",
                                        nvcc_version_str)[0]
            windows_cuda_root = Path(
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA")
            if not windows_cuda_root.exists():
                raise ValueError(f"can't find cuda in {windows_cuda_root}.")
            include = windows_cuda_root / f"v{version_str}\\include"
            lib64 = windows_cuda_root / f"v{version_str}\\lib\\x64"
        else:
            linux_cuda_root = Path("/usr/local/cuda")
            include = linux_cuda_root / f"include"
            lib64 = linux_cuda_root / f"lib64"
        _CACHED_CUDA_INCLUDE_LIB = (include, lib64)
        return _CACHED_CUDA_INCLUDE_LIB
    else:
        return _CACHED_CUDA_INCLUDE_LIB


class GemmKernelFlags(pccm.Class):
    """gemm support nvrtc, so we use less flags here.
    """
    def __init__(self):
        super().__init__()
        gpu_arch_flags, self.cuda_archs, self.has_ptx = _get_cuda_arch_flags(True)
        include, lib64 = _get_cuda_include_lib()
        self.build_meta.add_public_includes(include)
        self.build_meta.add_public_cflags("nvcc", *gpu_arch_flags)
        self.build_meta.add_public_cflags("nvcc",
            "-Xcudafe \"--diag_suppress=implicit_return_from_non_void_function\"",
        )

class GenericKernelFlags(pccm.Class):
    def __init__(self):
        super().__init__()
        gpu_arch_flags, self.cuda_archs, self.has_ptx = _get_cuda_arch_flags(False)
        include, lib64 = _get_cuda_include_lib()
        self.build_meta.add_public_includes(include)
        self.build_meta.add_public_cflags("nvcc", *gpu_arch_flags)
        # http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
        self.build_meta.add_public_cflags("nvcc",
            "-Xcudafe \"--diag_suppress=implicit_return_from_non_void_function\"",
        )

class CUDALibs(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(GenericKernelFlags)
        include, lib64 = _get_cuda_include_lib()
        self.build_meta.libraries.extend(["cudart"])
        self.build_meta.libpaths.append(lib64)

class TensorViewHeader(pccm.Class):
    def __init__(self):
        super().__init__()
        self.build_meta.add_public_includes(TENSORVIEW_INCLUDE_PATH)

class TensorViewCPU(pccm.Class):
    def __init__(self):
        super().__init__()
        self.build_meta.add_public_includes(TENSORVIEW_INCLUDE_PATH)
        self.add_include("array")
        self.add_include("tensorview/core/all.h")
        self.add_include("tensorview/tensor.h")
        self.add_include("tensorview/tools.h")
        self.add_include("tensorview/check.h")
        self.add_include("tensorview/profile/all.h")


class ThrustLib(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("thrust/device_vector.h")
        self.add_include("thrust/device_ptr.h")
        self.add_include("thrust/sort.h")
        self.add_include("thrust/unique.h")
        # workaround for windows CI, thrust may not exist in windows CUDA
        thrust_include = os.getenv("CUMM_THRUST_INCLUDE", "")
        if thrust_include:
            self.build_meta.add_public_includes(thrust_include)


class PyTorchLibNoPybind(pccm.Class):
    def __init__(self):
        super().__init__()
        spec = importlib.util.find_spec("torch")
        if spec is None:
            raise ValueError(
                "you need to install torch python")
        origin = Path(spec.origin)
        libtorch = origin.parent
        self.add_dependency(CUDALibs, TensorView)

        self.build_meta.add_public_includes(str(libtorch / "include"))
        self.build_meta.add_public_includes(str(libtorch / "include/torch/csrc/api/include"))
        torch_lib_paths = [str(libtorch / "lib")]
        torch_libs = ["c10", "torch", 'torch_cpu', 'torch_python']
        torch_cuda_libs = ["c10_cuda", "torch_cuda"]
        self.build_meta.libraries.extend(torch_libs + torch_cuda_libs)
        self.build_meta.libpaths.extend(torch_lib_paths)
        self.build_meta.add_public_cflags("nvcc,clang++,g++", "-D_GLIBCXX_USE_CXX11_ABI=0")
        self.add_include("torch/script.h")
        self.add_include("ATen/cuda/CUDAContext.h")
        self.add_include("ATen/ATen.h")

class PyTorchLib(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(PyTorchLibNoPybind)
        self.add_include("torch/extension.h") # include this to add pybind for torch.Tensor
        self.add_include("tensorview/torch_utils.h")

class TensorView(pccm.Class):
    def __init__(self):
        super().__init__()
        # any project depend on TensorView will add global nvcc flags:
        self.build_meta.add_global_cflags("nvcc", "--expt-relaxed-constexpr")
        self.build_meta.add_global_cflags("cl",  "/O2")
        self.build_meta.add_global_cflags("g++,clang++", "-O3")

        if not CUMM_CPU_ONLY_BUILD:
            self.add_dependency(CUDALibs, TensorViewCPU)
            self.build_meta.add_public_cflags("nvcc,clang++,g++", "-DTV_CUDA")
            self.build_meta.add_public_cflags("cl", "/DTV_CUDA")
        else:
            self.add_dependency(TensorViewCPU)

class TensorViewParallel(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_include("tensorview/parallel/all.h")
        self.build_meta.add_cflags("nvcc", "--extended-lambda")

class ExtendedLambda(pccm.Class):
    def __init__(self):
        super().__init__()
        self.build_meta.add_cflags("nvcc", "--extended-lambda")


class CummNVRTCLib(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_include("tensorview/cuda/nvrtc.h")
        self.add_include("tensorview/gemm/core/nvrtc_bases.h")

class CummNVRTCLink(pccm.Class):
    def __init__(self):
        super().__init__()
        self.build_meta.add_libraries("nvrtc")
        if compat.InLinux:
            self.build_meta.add_ldflags("g++", "-Wl,--no-as-needed", "nvrtc-builtins")
            self.build_meta.add_ldflags("clang++", "-Wl,--no-as-needed", "nvrtc-builtins")
            self.build_meta.add_ldflags("nvcc", "-Wl,--no-as-needed", "nvrtc-builtins")

class _CudaInclude(pccm.Class):
    def __init__(self):
        super().__init__()
        if not CUMM_CPU_ONLY_BUILD:
            include, lib64 = _get_cuda_include_lib()
            self.add_include("cuda.h")
            self.build_meta.add_private_includes(include)

class CompileInfo(pccm.Class):
    def __init__(self):
        super().__init__()
        if not CUMM_CPU_ONLY_BUILD:
            include, lib64 = _get_cuda_include_lib()
            _, self.cuda_archs, self.has_ptx = _get_cuda_arch_flags()
            self.ptx_arch = self.cuda_archs[-1]
            _, self.gemm_cuda_archs, self.gemm_has_ptx = _get_cuda_arch_flags(True)
            self.gemm_ptx_arch = self.gemm_cuda_archs[-1]
        else:
            self.has_ptx = False 
            self.gemm_has_ptx = False 
            self.ptx_arch = (0, 0)
            self.gemm_ptx_arch = (0, 0)
            self.cuda_archs = []
            self.gemm_cuda_archs = []
        self.add_include("vector", "tuple")
        self.add_include("string")
    
    if CUMM_CPU_ONLY_BUILD:
        _STATIC_FUNC = pccm.static_function
    else:
        _STATIC_FUNC = pccm.cuda.static_function

    @pccm.pybind.mark
    @_STATIC_FUNC
    def get_compiled_cuda_version(self):
        code = pccm.code()
        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"return std::make_tuple(-1, -1);")
        else:
            code.add_dependency(_CudaInclude)
            code.raw(f"""
            #ifdef __CUDACC_VER_MAJOR__
            return std::make_tuple(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
            #else
            int ver = CUDA_VERSION; // from cuda.h
            return std::make_tuple(ver / 1000, (ver % 1000) / 10);
            #endif
            """)
        return code.ret("std::tuple<int, int>")

    @pccm.pybind.mark
    @pccm.static_function
    def get_compiled_cuda_arch(self):
        code = pccm.code()
        code.raw(f"""
        std::vector<std::tuple<int, int>> res;
        """)
        for ar0, ar1 in self.cuda_archs:
            code.raw(f"res.push_back(std::make_tuple({ar0}, {ar1}));")
        code.raw(f"return res;")
        return code.ret("std::vector<std::tuple<int, int>>")

    @pccm.pybind.mark
    @pccm.static_function
    def get_compiled_gemm_cuda_arch(self):
        code = pccm.code()
        code.raw(f"""
        std::vector<std::tuple<int, int>> res;
        """)
        for ar0, ar1 in self.gemm_cuda_archs:
            code.raw(f"res.push_back(std::make_tuple({ar0}, {ar1}));")
        code.raw(f"return res;")
        return code.ret("std::vector<std::tuple<int, int>>")

    @pccm.pybind.mark
    @pccm.static_function
    def arch_is_compiled(self):
        code = pccm.code()
        code.arg("arch", "std::tuple<int, int>")
        code.raw(f"""
        """)
        for ar0, ar1 in self.cuda_archs:
            code.raw(f"""
            if (arch == std::make_tuple({ar0}, {ar1})){{
                return true;
            }}
            """)
        code.raw(f"return false;")
        return code.ret("bool")

    @pccm.pybind.mark
    @pccm.static_function
    def arch_is_compiled_gemm(self):
        code = pccm.code()
        code.arg("arch", "std::tuple<int, int>")
        code.raw(f"""
        """)
        for ar0, ar1 in self.gemm_cuda_archs:
            code.raw(f"""
            if (arch == std::make_tuple({ar0}, {ar1})){{
                return true;
            }}
            """)
        code.raw(f"return false;")
        return code.ret("bool")

    @pccm.pybind.mark
    @pccm.static_function
    def arch_is_compatible(self):
        code = pccm.code()
        code.arg("arch", "std::tuple<int, int>")
        if self.has_ptx:
            code.raw(f"""
            if (arch > std::make_tuple({self.ptx_arch[0]}, {self.ptx_arch[1]})){{
                return true;
            }}
            """)
        code.raw(f"""
        return arch_is_compiled(arch);
        """)
        return code.ret("bool")

    @pccm.pybind.mark
    @pccm.static_function
    def arch_is_compatible_gemm(self):
        code = pccm.code()
        code.arg("arch", "std::tuple<int, int>")
        if self.gemm_has_ptx:
            code.raw(f"""
            if (arch > std::make_tuple({self.gemm_ptx_arch[0]}, {self.gemm_ptx_arch[1]})){{
                return true;
            }}
            """)
        code.raw(f"""
        return arch_is_compiled_gemm(arch);
        """)
        return code.ret("bool")

    @pccm.pybind.mark
    @pccm.static_function
    def algo_can_use_ptx(self):
        code = pccm.code()
        code.arg("min_arch", "std::tuple<int, int>")
        code.arg("arch", "std::tuple<int, int>")
        ptx_arch = self.ptx_arch
        if self.has_ptx:
            code.raw(f"""
            auto ptx_arch = std::make_tuple({ptx_arch[0]}, {ptx_arch[1]});
            return min_arch <= ptx_arch && arch >= ptx_arch;
            """)
        code.raw(f"return false;")
        return code.ret("bool")

    @pccm.pybind.mark
    @pccm.static_function
    def gemm_algo_can_use_ptx(self):
        code = pccm.code()
        code.arg("min_arch", "std::tuple<int, int>")
        code.arg("arch", "std::tuple<int, int>")

        ptx_arch = self.gemm_ptx_arch
        if self.gemm_has_ptx:
            code.raw(f"""
            auto ptx_arch = std::make_tuple({ptx_arch[0]}, {ptx_arch[1]});
            return min_arch <= ptx_arch && arch >= ptx_arch;
            """)
        code.raw(f"return false;")
        return code.ret("bool")

    @pccm.pybind.mark
    @pccm.static_function
    def algo_can_be_nvrtc_compiled(self):
        code = pccm.code()
        code.arg("min_arch", "std::tuple<int, int>")
        cuda_ver_to_max_arch = [
            ((11, 8), (9, 0)),
            ((11, 1), (8, 6)),
            ((11, 0), (8, 0)),
            ((10, 2), (7, 5)),
        ]
        code.raw(f"""
        auto cuda_ver = get_compiled_cuda_version();
        """)
        for ver, max_arch in cuda_ver_to_max_arch:
            code.raw(f"""
            if (cuda_ver >= std::make_tuple({ver[0]}, {ver[1]})){{
                return min_arch <= std::make_tuple({max_arch[0]}, {max_arch[1]});
            }}
            """)
        code.raw(f"""
        return false;
        """)
        return code.ret("bool")

class TensorViewKernel(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("tensorview/cuda/kernel_all.h")
        self.add_include("tensorview/cuda/device_ops.h")
        self.add_include("tensorview/gemm/debug.h")


class TensorViewNVRTC(pccm.Class):
    """a class that contains all tensorview features with nvrtc support.
    """
    def __init__(self):
        super().__init__()
        self.add_include("tensorview/core/all.h")
        self.add_include("tensorview/tensorview.h")
        self.add_include("tensorview/core/arrayops/simple.h")
        if not CUMM_CPU_ONLY_BUILD:
            # here we can't depend on GemmKernelFlags
            # because nvrtc don't support regular arch flags.
            include, lib64 = _get_cuda_include_lib()
            self.build_meta.add_public_includes(include, TENSORVIEW_INCLUDE_PATH)
            self.add_include("tensorview/cuda/kernel_utils.h")
            self.build_meta.add_public_cflags("nvcc", "-DTV_CUDA")
            # if compat.InLinux:
            #     nvrtc_include = PACKAGE_ROOT / "nvrtc_include"
            #     self.build_meta.add_public_includes(nvrtc_include)

class TensorViewViewClass(pccm.Class):
    """a class that contains tensorview/tensorview.h.
    """
    def __init__(self):
        super().__init__()
        self.add_include("tensorview/tensorview.h")

class TensorViewCore(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewHeader)
        self.add_include("tensorview/core/all.h")
        self.add_include("tensorview/core/arrayops/simple.h")


class TensorViewArrayLinalg(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewCore)
        self.add_include("tensorview/core/arrayops/linalg.h")

class EigenLib(pccm.Class):
    def __init__(self):
        super().__init__()
        eigen_include = "/usr/include/eigen3/"
        self.build_meta.add_public_includes(eigen_include)
        self.add_include("Eigen/Core")
        self.add_include("Eigen/Dense")
        self.add_include("Eigen/Geometry")
        self.add_include("Eigen/StdVector")

class TensorViewNVRTCKernel(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.add_include("tensorview/cuda/device_ops.h")
        self.add_include("tensorview/gemm/debug.h")


class TensorViewHashKernel(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewKernel)
        self.add_include("tensorview/hash/all.cu.h")

class TensorViewNVRTCHashKernel(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.add_include("tensorview/hash/linear.cu.h")

class TensorViewMath(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("tensorview/math/all.h")


class GemmDTypes(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("tensorview/gemm/dtypes/all.h")


class GemmBasic(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("tensorview/core/nvrtc/runtime_include.h"
                         )  # for __global__, __host__ and other cuda attrs.
        self.add_include("tensorview/gemm/core/constants.h")
        self.add_include("tensorview/gemm/core/utils.h")
        self.add_include("tensorview/gemm/core/nvrtc_params.h")
        self.add_include("tensorview/gemm/dtypes/all.h")

class GemmBasicHost(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("tensorview/gemm/core/params.h")
        self.add_include("tensorview/gemm/core/nvrtc_bases.h")

class GemmBasicKernel(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(GemmBasic)
        self.add_include("tensorview/gemm/arch/memory.h")
        self.add_include("tensorview/gemm/arch/transpose.h")
        self.add_include("tensorview/gemm/arch/semaphore.h")


class PyBind11(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("pybind11/stl.h")
        self.add_include("pybind11/pybind11.h")
        self.add_include("pybind11/numpy.h")
        # self.add_include("pybind11/eigen.h")
        self.add_include("pybind11/stl_bind.h")


class BoostGeometryLib(pccm.Class):
    def __init__(self):
        super().__init__()
        boost_root = os.getenv("BOOST_ROOT", None)
        assert boost_root is not None, f"can't find BOOST_ROOT env"
        boost_root_p = Path(boost_root)
        assert (boost_root_p / "boost" / "geometry").exists()
        self.add_include("boost/geometry.hpp")


class NlohmannJson(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("tensorview/thirdparty/nlohmann/json.hpp")


class TslRobinMap(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("tensorview/thirdparty/tsl/robin_map.h")

class CppTimer(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_include("tensorview/profile/cuda_profiler.h")
        self.build_meta.add_public_cflags("g++,clang++,nvcc", "-DTV_USE_LIBRT")
        self.build_meta.add_libraries("rt")

class TensorViewCPULLVM(pccm.Class):
    def __init__(self):
        super().__init__()
        self.build_meta.add_public_includes(TENSORVIEW_INCLUDE_PATH)
        self.add_include("array")

        self.add_include("tensorview/core/all.h")
        self.add_include("tensorview/tensor.h")
        self.add_include("tensorview/check.h")
        self.add_include("tensorview/profile/all.h")
        # self.build_meta.add_global_cflags("g++,clang++", "-g")
        # currently llvmlite can't handle iostream, so
        # we must disable it by -DTV_LLVM_JIT for tensorview library
        self.build_meta.add_global_cflags("g++,clang++", "-DTV_LLVM_JIT")

class TensorViewLLVM(pccm.Class):
    def __init__(self):
        super().__init__()
        self.build_meta.add_global_cflags("nvcc", "--expt-relaxed-constexpr")
        self.build_meta.add_global_cflags("cl",  "/O2")
        self.build_meta.add_global_cflags("g++,clang++", "-O3")

        if not CUMM_CPU_ONLY_BUILD:
            self.add_dependency(CUDALibs, TensorViewCPULLVM)
            self.build_meta.add_public_cflags("nvcc,clang++,g++", "-DTV_CUDA")
            self.build_meta.add_public_cflags("cl", "/DTV_CUDA")
        else:
            self.add_dependency(TensorViewCPULLVM)

class TensorViewNVRTCDev(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)

    @pccm.cuda.static_function(device=True)
    def device_function(self):
        code = pccm.code()
        code.arg("a, b", "float")
        code.raw("return a + b;")
        return code.ret("float")