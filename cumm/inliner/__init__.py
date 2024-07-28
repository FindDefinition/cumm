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

import contextlib
import re
import pccm
from pccm.builder.inliner import InlineBuilder, InlineBuilderPlugin, PCCM_INLINE_NAMESPACE, PCCM_INLINE_FUNCTION_NAME, PCCM_INLINE_FUNCTION_NAME_FORMAT
from cumm.nvrtc import CummLLVMModule, CummMetalModule, CummNVRTCModule, CummNVRTCModuleBase, create_nvrtc_code
from pathlib import Path
from cumm.common import TensorViewKernel
import enum
from pccm.utils import get_qualname_of_type
from typing import Any, Callable, ContextManager, Dict, List, Optional, Set, Tuple, Type, Union
import numpy as np
from cumm import dtypes, tensorview as tv
from cumm.common import TensorViewNVRTC, GemmBasic, TensorViewViewClass
from cumm.gemm.codeops import div_up
from cumm.tensorview import nullcontext
from ccimport import compat
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
            torch.bfloat16: tv.bfloat16,
        })
    return _TORCH_DTYPE_TO_TV


class CUDAMode(enum.Enum):
    Kernel1D = "Kernel1D"
    KernelRaw = "KernelRaw"


def torch_tensor_to_tv(ten,
                       dtype: Optional[int] = None,
                       shape: Optional[List[int]] = None):
    # assert ten.is_contiguous(), "must be contiguous tensor"
    device = ten.device
    if device.type == "cpu":
        tv_device = -1
    elif device.type == "cuda":
        tv_device = 0
    elif device.type == "mps":
        tv_device = 0
    else:
        raise NotImplementedError
    if device.type == "mps":
        # mps data ptr is MTLBuffer, not real ptr
        # if we use ten.data_ptr(), the result will be 
        # MTLBuffer + data_offset, which will cause
        # segfault.
        ptr = ten.untyped_storage().data_ptr()
        offset = ten.storage_offset()
    else:
        ptr = ten.data_ptr()
        offset = 0
    if shape is None:
        shape = list(ten.shape)
    if dtype is None:
        dtype = _cached_get_torch_dtype_to_tv()[ten.dtype]
    return tv.from_blob_strided(ptr, shape, list(ten.stride()), dtype,
                                tv_device, offset)


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
            # print("??????", obj.shape, obj.dtype)
            obj = torch_tensor_to_tv(obj)
        prefix = ""
        if obj.is_readonly():
            prefix = "const "
        if user_arg is not None and isinstance(user_arg, _NVRTCInlineParams):
            if user_arg.capture_tensor_as_tview:
                return f"tv::TensorView<{prefix}{dtypes.get_dtype_from_tvdtype(obj.dtype, True)}, {obj.ndim}>"
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
        # if isinstance(user_arg, _NVRTCInlineParams):
        #     if user_arg.is_cpu:
        #         return obj.reshape(-1).tolist()
        assert obj.nbytes <= 256, "we only support capture small numpy which will be passed by value to kernel"
        return obj

    @staticmethod
    def get_cpp_type_static(
                     obj: np.ndarray,
                     user_arg: Optional[Any] = None) -> Union[str, Tuple[str, int]]:
        ndim = obj.ndim
        dtype = obj.dtype
        cpp_type = _NPDTYPE_TO_LIST_TYPE_STR[dtype]
        # if isinstance(user_arg, _NVRTCInlineParams):
        #     if user_arg.is_cpu:
        #         return cpp_type, obj.size
        array_type = ""
        array_type += "tv::array<" * ndim
        array_type += f"{cpp_type}, "
        shape_rev = obj.shape[::-1]
        array_type += ">, ".join(map(str, shape_rev))
        res = array_type + ">"
        return res

    def get_cpp_type(self,
                     obj: np.ndarray,
                     user_arg: Optional[Any] = None) -> Union[str, Tuple[str, int]]:
        
        return self.get_cpp_type_static(obj, user_arg)


class _NVRTCInlineParams:
    def __init__(self,
                 mode: CUDAMode,
                 launch: tv.LaunchParam,
                 verbose: bool = False,
                 verbose_path: str = "",
                 measure_time: bool = False,
                 is_cpu: bool = False,
                 capture_tensor_as_tview: bool = False,
                 perf_context: Optional[ContextManager] = None,
                 run_in_process: bool = False) -> None:
        self.mode = mode
        self.launch = launch
        self.verbose = verbose
        self.verbose_path = verbose_path
        self.measure_time = measure_time
        self.is_cpu = is_cpu
        self.capture_tensor_as_tview = capture_tensor_as_tview
        self.perf_context = perf_context
        self.run_in_process = run_in_process


_NVRTC_FUNC_NAME = f"{PCCM_INLINE_NAMESPACE}::{PCCM_INLINE_FUNCTION_NAME}"
_NVRTC_FUNC_NAME_FORMAT = f"{PCCM_INLINE_NAMESPACE}::{PCCM_INLINE_FUNCTION_NAME_FORMAT}"

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
            remote_addr: str = "",
            default_deps: Optional[List[Type[pccm.Class]]] = None,
            std: str = "c++14",
            context: Optional[tv.Context] = None):
        if plugins is None:
            plugins = _DEFAULT_KERNEL_PLUGINS
        if default_deps is None:
            deps.extend([TensorViewNVRTC, ])
            if not compat.InMacOS:
                deps.append(GemmBasic)
        else:
            deps.extend(default_deps)
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
        self.cpp_std = std
        self.maximum_1d_threads = 512
        self.measure_build_time = measure_build_time
        self._remote_addr = remote_addr
        
        self.ctx = tv.Context() if context is None else context
        if compat.InMacOS and not self.ctx.has_apple_metal_context():
            self.ctx.create_apple_metal_context()

    
    def synchronize(self):
        self.ctx.synchronize()

    def synchronize_if_apple_metal(self):
        """This method is used for mixed torch and cumm inline cuda code.
        cuda have a default stream and context but metal not, at least we
        can't acquire that context in pytorch. this means
        we have two metal command queue in single process, metal ops exec 
        order on two queue is undefined. so if you mix torch and inline cuda
        code, run sync before/after any torch operation. you can use 
        `inliner.synchronize_if_apple_metal` to achieve this.
        """
        if compat.IsAppleSiliconMacOs:
            self.ctx.synchronize()

    def get_nvrtc_module(self, name: str) -> Optional[CummNVRTCModule]:
        for k, v in self.modules.items():
            if name == k[1]:
                return v.func
        return None
    
    def get_nvrtc_kernel_attrs(self, name: str) -> Dict[str, int]:
        nvrtc_mod = self.get_nvrtc_module(name)
        assert nvrtc_mod is not None
        return nvrtc_mod.get_kernel_attrs(nvrtc_mod.get_lowered_name(_NVRTC_FUNC_NAME_FORMAT.format(name)))

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
        is_cpu = False
        meta = pccm.cuda.CudaGlobalFunctionMeta(attrs=["__global__"])
        if arg is not None:
            is_cpu = arg.is_cpu
        if is_cpu:
            meta = pccm.cuda.ExternalFunctionMeta()
        if not is_cpu and compat.InMacOS:
            meta = pccm.cuda.CudaGlobalFunctionMeta(attrs=["kernel"])
        if arg is None:
            code.raw(code_str)
            return meta
        if not is_cpu and compat.InMacOS:
            # metal 
            func_const_cnt = 0
            new_args: List[pccm.Argument] = []
            func_constants: List[str] = []
            for prev_arg in code.arguments:
                if isinstance(prev_arg.userdata, np.ndarray):
                    new_name = f"__cumm_metal_arg_{prev_arg.name}"
                    cpp_type_scalar = _NPDTYPE_TO_LIST_TYPE_STR[prev_arg.userdata.dtype]
                    cpp_type_arr = NumpyPlugin.get_cpp_type_static(prev_arg.userdata)
                    prev_arg.type_str = f"constant {cpp_type_scalar}*"
                    code.raw(f"thread {cpp_type_arr} {prev_arg.name} = reinterpret_cast<constant {cpp_type_arr}*>({new_name})[0];")
                    prev_arg.name = new_name
                    new_args.append(prev_arg)
                elif get_qualname_of_type(type(prev_arg.userdata)) == TORCH_TENSOR_NAME:
                    prev_arg.type_str = f"device {prev_arg.type_str}"
                    new_args.append(prev_arg)
                elif isinstance(prev_arg.userdata, tv.Tensor):
                    prev_arg.type_str = f"device {prev_arg.type_str}"
                    new_args.append(prev_arg)
                else:
                    assert isinstance(prev_arg.userdata, (int, float, bool))
                    dtype = "int64_t"
                    if isinstance(prev_arg.userdata, float):
                        dtype = "float"
                    elif isinstance(prev_arg.userdata, bool):
                        dtype = "bool"
                    func_constants.append(f"constant {dtype} {prev_arg.name} [[function_constant({func_const_cnt})]];")
                    func_const_cnt += 1
            if arg.mode != CUDAMode.KernelRaw:
                code.arguments = new_args + [
                    pccm.Argument(self.index_name, "uint32_t", attributes=["thread_position_in_grid"])
                ]
            code.code_before_func_def = "\n".join(func_constants)
        trycatch_ctx = contextlib.nullcontext()
        if is_cpu:
            code.raw(f"""
            int __pccm_error_code = 0;
            """)
            trycatch_ctx = code.block("try", end=f"""
            }}catch (const std::exception& e){{
                tv::printf2("LLVM Inline Function Exception!. Error:", e.what());
                __pccm_error_code = 1;
            }}
            """)
        with trycatch_ctx:
            if arg.mode == CUDAMode.KernelRaw:
                code.raw(code_str)
            else:
                if not is_cpu:
                    if compat.InMacOS:
                        code.raw(code_str)
                    else:
                        with code.for_(
                                f"auto {self.index_name} : tv::KernelLoopX<int>({_CUMM_KERNEL_1D_SIZE_NAME})"
                        ):
                            code.raw(code_str)
                else:
                    with code.for_(
                            f"size_t i = 0; i <{_CUMM_KERNEL_1D_SIZE_NAME}; ++i"
                    ):
                        code.raw(code_str)
        if is_cpu:
            code.raw("return __pccm_error_code;")
            code.ret("int")
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
        params = create_nvrtc_code([pccm_cls], std=self.cpp_std)
        is_cpu = False
        if user_arg is not None:
            is_cpu = user_arg.is_cpu
        if self._remote_addr != "":
            # this should be used only you want to debug
            # different cuda version.
            import tensorpc
            with tensorpc.RemoteManager(self._remote_addr) as robj:
                mod = robj.remote_call("NVRTCCompiler.compile_nvrtc", params)
        else:
            with ctx:
                if is_cpu:
                    if not compat.InLinux and not compat.InMacOS:
                        raise NotImplementedError("cpu jit only support linux")
                    mod = CummLLVMModule([pccm_cls],
                                         verbose=verbose,
                                         verbose_path=verbose_path,
                                         std=self.cpp_std)
                else:
                    if compat.InMacOS:
                        mod = CummMetalModule([pccm_cls],
                                            verbose=verbose,
                                            verbose_path=verbose_path,
                                            std=self.cpp_std)
                    else:
                        if verbose:
                            mod = CummNVRTCModule([pccm_cls],
                                                verbose=verbose,
                                                verbose_path=verbose_path,
                                                std=self.cpp_std)
                        else:
                            mod = CummNVRTCModuleBase.from_params(params)

        return mod

    def _get_nvrtc_inline_func_name_for_debug(self, name: str):
        return _NVRTC_FUNC_NAME_FORMAT.format(re.sub('[^0-9a-zA-Z]', '_', name))

    def run_func(self,
                name: str,
                 func: Union[CummNVRTCModuleBase, CummMetalModule],
                 *args,
                 user_args: Optional[_NVRTCInlineParams] = None):
        assert user_args is not None
        launch = user_args.launch.copy()
        launch.ctx = self.ctx
        if isinstance(func, CummMetalModule):
            return func.run_kernel(self._get_nvrtc_inline_func_name_for_debug(name), launch, *args, perf_context=user_args.perf_context)

        else:
            if user_args.run_in_process:
                return func.run_kernel_in_spawn_process(self._get_nvrtc_inline_func_name_for_debug(name), launch, *args)
            else:
                return func.run_kernel(self._get_nvrtc_inline_func_name_for_debug(name), launch, *args, perf_context=user_args.perf_context)

    def kernel_raw(self,
                   name: str,
                   param: tv.LaunchParam,
                   code: Union[str, pccm.FunctionCode],
                   verbose_path: str = "",
                   disable_cache: bool = False,
                   capture_tensor_as_tview: bool = False,
                   perf_context: Optional[ContextManager] = None,
                   run_in_process: bool = False,
                   *,
                   _frame_cnt: int = 2):
        verbose = verbose_path != ""
        user_arg = _NVRTCInlineParams(CUDAMode.KernelRaw, param, verbose,
                                      verbose_path, capture_tensor_as_tview=capture_tensor_as_tview,
                                      perf_context=perf_context, run_in_process=run_in_process)
        if capture_tensor_as_tview:
            if not isinstance(code, pccm.FunctionCode):
                code_pccm = pccm.code()
                code_pccm.raw(code)
                code = code_pccm
            code.add_dependency(TensorViewViewClass)
        return self.inline(name,
                           code,
                           ".cu",
                           _frame_cnt=_frame_cnt,
                           user_arg=user_arg,
                           disable_cache=disable_cache)
    
    def kernel_raw_capture_tview(self,
                                name: str,
                                param: tv.LaunchParam,
                                code: Union[str, pccm.FunctionCode],
                                verbose_path: str = "",
                                disable_cache: bool = False):
        """same as kernel_raw except all tensors (tv.Tensor, torch.Tensor)
        are captured as tv::TensorView with shape/stride support inside kernel.
        """
        return self.kernel_raw(name=name, 
                              param=param, 
                              code=code, 
                              verbose_path=verbose_path, 
                              disable_cache=disable_cache, 
                              capture_tensor_as_tview=True,
                              _frame_cnt=3)

    def kernel_1d(self,
                  name: str,
                  num: int,
                  stream: int,
                  code: Union[str, pccm.FunctionCode],
                  verbose_path: str = "",
                  disable_cache: bool = False,
                  capture_tensor_as_tview: bool = False,
                  perf_context: Optional[ContextManager] = None,
                  run_in_process: bool = False,
                  *,
                  _frame_cnt: int = 2,
                  maximum_1d_threads: Optional[int] = None):
        verbose = verbose_path != ""
        num = int(num)
        user_arg = _NVRTCInlineParams(CUDAMode.Kernel1D,
                                      self.get_1d_param(num, stream=stream, maximum_1d_threads=maximum_1d_threads),
                                      verbose, verbose_path,
                                      capture_tensor_as_tview=capture_tensor_as_tview,
                                      perf_context=perf_context, run_in_process=run_in_process)
        additional_args = {
            _CUMM_KERNEL_1D_SIZE_NAME: num,
        }
        if compat.InMacOS:
            additional_args.clear()
        if capture_tensor_as_tview:
            if not isinstance(code, pccm.FunctionCode):
                code_pccm = pccm.code()
                code_pccm.raw(code)
                code = code_pccm
            code.add_dependency(TensorViewViewClass)
        return self.inline(name,
                           code,
                           ".cu",
                           additional_args,
                           _frame_cnt=_frame_cnt,
                           user_arg=user_arg,
                           disable_cache=disable_cache)

    def kernel_1d_capture_tview(self,
                                name: str,
                                num: int,
                                stream: int,
                                code: Union[str, pccm.FunctionCode],
                                verbose_path: str = "",
                                disable_cache: bool = False):
        """same as kernel_1d except all tensors (tv.Tensor, torch.Tensor)
        are captured as tv::TensorView with shape/stride support inside kernel.
        """
        return self.kernel_1d(name=name, 
                              num=num, 
                              stream=stream, 
                              code=code, 
                              verbose_path=verbose_path, 
                              disable_cache=disable_cache, 
                              capture_tensor_as_tview=True,
                              _frame_cnt=3)

    def cpu_kernel_1d(self,
                      name: str,
                      num: int,
                      code: Union[str, pccm.FunctionCode],
                      verbose_path: str = "",
                      disable_cache: bool = False):
        verbose = verbose_path != ""
        num = int(num)
        user_arg = _NVRTCInlineParams(CUDAMode.Kernel1D,
                                      self.get_1d_param(num, stream=0),
                                      verbose, verbose_path,
                                      is_cpu=True)
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

    def cpu_kernel_raw(self,
                       name: str,
                       #    param: tv.LaunchParam,
                       code: Union[str, pccm.FunctionCode],
                       verbose_path: str = "",
                       disable_cache: bool = False):
        verbose = verbose_path != ""
        user_arg = _NVRTCInlineParams(CUDAMode.KernelRaw,
                                      tv.LaunchParam(
                                          (1, 1, 1), (1, 1, 1)), verbose,
                                      verbose_path, is_cpu=True)
        return self.inline(name,
                           code,
                           ".cu",
                           _frame_cnt=2,
                           user_arg=user_arg,
                           disable_cache=disable_cache)

    def cpu_prepare_libraries(self, *libs: str):
        import llvmlite.binding as llvm
        for l in libs:
            llvm.load_library_permanently(l)

    def get_1d_param(self, num: int, smem: int = 0, stream: int = 0, maximum_1d_threads: Optional[int] = None):
        if maximum_1d_threads is None:
            maximum_1d_threads = self.maximum_1d_threads
        if num > maximum_1d_threads:
            threads = maximum_1d_threads
        else:
            threads = div_up(num, 32) * 32
        if compat.InMacOS:
            blocks = num # for metal, the grid size is the same as block size
        else:
            blocks = div_up(num, threads)
        return tv.LaunchParam((blocks, 1, 1), (threads, 1, 1), smem, stream)


def main():
    import torch
    from cumm import tensorview as tv
    from cumm.common import TensorViewArrayLinalg, TensorViewNVRTCHashKernel
    INLINE = NVRTCInlineBuilder(
        [TensorViewArrayLinalg, TensorViewNVRTCHashKernel], reload_when_code_change=True)

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
        bb[1] = std::numeric_limits<float>::denorm_min();
        bb[2] = z;
        """)

    print(a.cpu().numpy())


if __name__ == "__main__":
    main()
