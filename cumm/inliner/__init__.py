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
""" nvrtc inliner with capture.

don't forget to install a highlight extension to show captures in string!

my vscode highlight extension config:

    "highlight.regexes": {
        "(\\$)([_a-zA-Z][_a-zA-Z0-9]*)": {
            "filterFileRegex": ".*.py$",
            "decorations": [
                {
                    "color": "#c678dd",
                    "fontWeight": "bold"
                },
                {
                    "color": "#e06c75",
                    "fontWeight": "bold"
                }

            ]
        },
        "(\\$)(\\()(.*?)(\\))": {
            "filterFileRegex": ".*.py$",
            "decorations": [
                {
                    "color": "#c678dd",
                    "fontWeight": "bold"
                },
                {
                    "color": "#fbf755",
                    "fontWeight": "bold"
                },
                {
                    "color": "#e06c75",
                    "fontWeight": "bold"
                },
                {
                    "color": "#fbf755",
                    "fontWeight": "bold"
                },
            ]
        },

    },


"""

import pccm
from pccm.builder.inliner import InlineBuilder, InlineBuilderPlugin, PCCM_INLINE_NAMESPACE, PCCM_INLINE_FUNCTION_NAME
from cumm.nvrtc import CummNVRTCModule, CummNVRTCModuleBase, create_nvrtc_code
from pathlib import Path
from cumm.common import TensorViewKernel
import enum
from pccm.utils import get_qualname_of_type
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import numpy as np
from cumm import dtypes, tensorview as tv
from cumm.common import TensorViewNVRTC, GemmBasic
from cumm.gemm.codeops import div_up
from cumm.tensorview import nullcontext
_TORCH_DTYPE_TO_TV: Dict[Any, int] = {}
TORCH_TENSOR_NAME = "torch.Tensor"


class LaunchParam:
    def __init__(self,
                 blocks: Tuple[int, ...],
                 threads: Tuple[int, ...],
                 smem: int = 0,
                 stream: int = 0) -> None:
        self.blocks = list(blocks)
        self.threads = list(threads)
        assert len(blocks) == 3
        assert len(threads) == 3
        self.smem = smem
        self.stream = stream


def _cached_get_torch_dtype_to_tv():
    import torch
    if not _TORCH_DTYPE_TO_TV:
        _TORCH_DTYPE_TO_TV.update({
            torch.float32: tv.float32,
            torch.float64: tv.float64,
            torch.float16: tv.float16,
            torch.int32: tv.int32,
            torch.int64: tv.int64,
            torch.int8: tv.int8,
            torch.int16: tv.int16,
            torch.uint8: tv.uint8,
        })
    return _TORCH_DTYPE_TO_TV


class CUDAMode(enum.Enum):
    Kernel1D = "Kernel1D"
    KernelRaw = "KernelRaw"


def torch_tensor_to_tv(ten,
                       dtype: Optional[int] = None,
                       shape: Optional[List[int]] = None):
    assert ten.is_contiguous(), "must be contiguous tensor"
    ptr = ten.data_ptr()
    device = ten.device
    if device.type == "cpu":
        tv_device = -1
    elif device.type == "cuda":
        tv_device = 0
    else:
        raise NotImplementedError
    if shape is None:
        shape = list(ten.shape)
    if dtype is None:
        dtype = _cached_get_torch_dtype_to_tv()[ten.dtype]
    return tv.from_blob_strided(ptr, shape, list(ten.stride()), dtype,
                                tv_device)


def get_current_stream():
    import torch
    return torch.cuda.current_stream().cuda_stream


class TVTensorPlugin(InlineBuilderPlugin):
    QualifiedName = get_qualname_of_type(tv.Tensor)

    def handle_captured_type(
            self,
            name: str,
            code: pccm.FunctionCode,
            obj: Any,
            user_arg: Optional[Any] = None) -> Optional[Tuple[str, str]]:
        return None

    def type_conversion(self, obj: Any, user_arg: Optional[Any] = None):
        if get_qualname_of_type(type(obj)) == TORCH_TENSOR_NAME:
            return torch_tensor_to_tv(obj)
        return obj

    def get_cpp_type(self, obj: Any, user_arg: Optional[Any] = None) -> str:
        if get_qualname_of_type(type(obj)) == TORCH_TENSOR_NAME:
            obj = torch_tensor_to_tv(obj)
        prefix = ""
        if obj.is_readonly():
            prefix = "const "
        return f"{prefix}{dtypes.get_dtype_from_tvdtype(obj.dtype, True)}*"


_NPDTYPE_TO_LIST_TYPE_STR: Dict[np.dtype, str] = {
    np.dtype(np.float16): "__half",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.int8): "int8_t",
    np.dtype(np.int16): "int16_t",
    np.dtype(np.int32): "int32_t",
    np.dtype(np.int64): "int64_t",
    np.dtype(np.uint8): "uint8_t",
    np.dtype(np.uint16): "uint16_t",
    np.dtype(np.uint32): "uint32_t",
    np.dtype(np.uint64): "uint64_t",
    np.dtype(np.bool_): "bool",
}


class NumpyPlugin(InlineBuilderPlugin):
    """for nvrtc, we capture np.ndarray as tv::array, so the shape must fixed and small.
    """
    QualifiedName = get_qualname_of_type(np.ndarray)

    def handle_captured_type(self,
                             name: str,
                             code: pccm.FunctionCode,
                             obj: Any,
                             user_arg: Optional[Any] = None) -> Optional[str]:
        return

    def type_conversion(self, obj: np.ndarray, user_arg: Optional[Any] = None):
        return obj.tolist()

    def get_cpp_type(self,
                     obj: np.ndarray,
                     user_arg: Optional[Any] = None) -> str:
        ndim = obj.ndim
        dtype = obj.dtype
        if dtype is None:
            dtype = obj.dtype
        cpp_type = _NPDTYPE_TO_LIST_TYPE_STR[dtype]
        array_type = ""
        array_type += "tv::array<" * ndim
        array_type += f"{cpp_type}, "
        shape_rev = obj.shape[::-1]
        array_type += ">, ".join(map(str, shape_rev))
        return array_type + ">"


class _NVRTCInlineParams:
    def __init__(self,
                 mode: CUDAMode,
                 launch: tv.LaunchParam,
                 verbose: bool = False,
                 verbose_path: str = "") -> None:
        self.mode = mode
        self.launch = launch
        self.verbose = verbose
        self.verbose_path = verbose_path


_NVRTC_FUNC_NAME = f"{PCCM_INLINE_NAMESPACE}::{PCCM_INLINE_FUNCTION_NAME}"

_DEFAULT_KERNEL_PLUGINS: Dict[str, InlineBuilderPlugin] = {
    "numpy.ndarray": NumpyPlugin(),
    TVTensorPlugin.QualifiedName: TVTensorPlugin(),
    TORCH_TENSOR_NAME: TVTensorPlugin(),
}

_CUMM_KERNEL_1D_SIZE_NAME = "_cumm_pccm_inline_size"


class NVRTCInlineBuilder(InlineBuilder):
    def __init__(
            self,
            deps: List[Type[pccm.Class]],
            plugins: Optional[Dict[str, InlineBuilderPlugin]] = None,
            index_name: str = "i",
            root: Optional[Path] = None,
            build_root: Optional[Path] = None,
            build_kwargs: Optional[Dict[str, Any]] = None,
            param_deps: Optional[List[pccm.ParameterizedClass]] = None,
            measure_build_time: bool = False,
            reload_when_code_change: bool = False,
            remote_addr: str = ""):
        if plugins is None:
            plugins = _DEFAULT_KERNEL_PLUGINS
        deps.extend([TensorViewNVRTC, GemmBasic])
        if build_kwargs is None:
            build_kwargs = {}
        super().__init__(deps=deps,
                         plugins=plugins,
                         build_kwargs=build_kwargs,
                         root=root,
                         build_root=build_root,
                         param_deps=param_deps,
                         reload_when_code_change=reload_when_code_change)
        self.index_name = index_name
        self.maximum_1d_threads = 512
        self.measure_build_time = measure_build_time
        self._remote_addr = remote_addr

    def get_nvrtc_module(self, name: str) -> Optional[CummNVRTCModule]:
        for k, v in self.modules.items():
            if name == k[1]:
                return v.func
        return None

    def get_save_root(self,
                      path: Path,
                      root: Optional[Path] = None,
                      build_root: Optional[Path] = None):
        # override get_save_root because we don't need it for nvrtc.
        return Path()

    def get_base_class(self):
        return pccm.Class()

    def handle_container_code(self, code_str: str, code: pccm.FunctionCode,
                              arg: Optional[_NVRTCInlineParams]):
        meta = pccm.cuda.CudaGlobalFunctionMeta(attrs=["__global__"])

        if arg is None:
            code.raw(code_str)
            return meta
        if arg.mode == CUDAMode.KernelRaw:
            code.raw(code_str)
            return meta

        with code.for_(
                f"auto {self.index_name} : tv::KernelLoopX<int>({_CUMM_KERNEL_1D_SIZE_NAME})"
        ):
            code.raw(code_str)
        return meta

    def build(self,
              pccm_cls: pccm.Class,
              mod_root: Path,
              name: str,
              timeout: float,
              user_arg: Optional[_NVRTCInlineParams] = None):
        verbose = False
        verbose_path = ""
        if user_arg is not None:
            verbose = user_arg.verbose
            verbose_path = user_arg.verbose_path
        ctx = nullcontext()
        if self.measure_build_time:
            ctx = tv.measure_and_print(f"{name} nvrtc build time")
        # with tv.measure_and_print("INLINE"):
        params = create_nvrtc_code([pccm_cls])
        if self._remote_addr != "":
            # this should be used only you want to debug
            # different cuda version.
            import tensorpc 
            with tensorpc.RemoteManager(self._remote_addr) as robj:
                mod = robj.remote_call("NVRTCCompiler.compile_nvrtc", params)
        else:
            with ctx:
                if verbose:
                    mod = CummNVRTCModule([pccm_cls],
                                            verbose=verbose,
                                            verbose_path=verbose_path)
                else:
                    mod = CummNVRTCModuleBase.from_params(params)
        
        return mod

    def run_func(self,
                 func: CummNVRTCModuleBase,
                 *args,
                 user_args: Optional[_NVRTCInlineParams] = None):
        assert user_args is not None
        launch = user_args.launch
        return func.run_kernel(_NVRTC_FUNC_NAME, launch, *args)

    def kernel_raw(self,
                   name: str,
                   param: tv.LaunchParam,
                   code: Union[str, pccm.FunctionCode],
                   verbose_path: str = "",
                   disable_cache: bool = False):
        verbose = verbose_path != ""
        user_arg = _NVRTCInlineParams(CUDAMode.KernelRaw, param, verbose,
                                      verbose_path)
        return self.inline(name,
                           code,
                           ".cu",
                           _frame_cnt=2,
                           user_arg=user_arg,
                           disable_cache=disable_cache)

    def kernel_1d(self,
                  name: str,
                  num: int,
                  stream: int,
                  code: Union[str, pccm.FunctionCode],
                  verbose_path: str = "",
                  disable_cache: bool = False):
        verbose = verbose_path != ""
        num = int(num)
        user_arg = _NVRTCInlineParams(CUDAMode.Kernel1D,
                                      self.get_1d_param(num, stream=stream),
                                      verbose, verbose_path)
        additional_args = {
            _CUMM_KERNEL_1D_SIZE_NAME: num,
        }
        return self.inline(name,
                           code,
                           ".cu",
                           additional_args,
                           _frame_cnt=2,
                           user_arg=user_arg,
                           disable_cache=disable_cache)

    def get_1d_param(self, num: int, smem: int = 0, stream: int = 0):
        if num > self.maximum_1d_threads:
            threads = self.maximum_1d_threads
        else:
            threads = div_up(num, 32) * 32
        blocks = div_up(num, threads)
        return tv.LaunchParam((blocks, 1, 1), (threads, 1, 1), smem, stream)


def main():
    import torch
    INLINE = NVRTCInlineBuilder([], reload_when_code_change=True)
    from cumm import tensorview as tv
    print(1)
    a = tv.zeros([2], tv.float32, 0)
    print(2)
    INLINE.kernel_1d("hahaha", a.dim(0), 0, f"""
    $a[i] = 5;
    """)
    b = tv.zeros([1000, 3], tv.float32, 0)
    bbb = torch.rand(1000, 3).float().cuda()
    t = np.eye(4)
    for i in range(10):
        INLINE.kernel_1d(
            "hahaha2", a.dim(0), 0, f"""
        auto bb = $bbb + i * 3;
        auto x = bb[0] * $t[0][0] + bb[1] * $t[0][1] + bb[2] * $t[0][1] + $t[0][3];
        auto y = bb[0] * $t[1][0] + bb[1] * $t[1][1] + bb[2] * $t[1][1] + $t[1][3];
        auto z = bb[0] * $t[2][0] + bb[1] * $t[2][1] + bb[2] * $t[2][1] + $t[2][3];

        bb[0] = x;
        bb[1] = y;
        bb[2] = z;
        """)

    print(a.cpu().numpy())


if __name__ == "__main__":
    main()
