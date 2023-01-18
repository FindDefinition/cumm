import dataclasses
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import subprocess

import pccm
from ccimport import compat
from pccm import Argument
from pccm.core import CodeGenerator
from pccm.core.buildmeta import BuildMeta
from pccm.targets.cuda import CudaGlobalFunctionMeta
from pccm.core import ExternalFunctionMeta
import ctypes
from cumm import PACKAGE_ROOT
from cumm import tensorview as tv
from cumm.common import TensorViewKernel, _get_cuda_include_lib
from cumm.core_cc.tensorview_bind import Tensor
import numpy as np 

LLVM_IS_INITED = False

_cudadevrt_libname = "libcudadevrt.a"
if compat.InWindows:
    _cudadevrt_libname = "cudadevrt.lib"

def get_cudadevrt_path():
    _, lib64 = _get_cuda_include_lib()
    _cudadevrt_path_candidates = [
        PACKAGE_ROOT / "lib" / _cudadevrt_libname,  # pip package
        lib64 / _cudadevrt_libname,  # pip package
    ]
    for c in _cudadevrt_path_candidates:
        if c.exists():
            return c
    return None

def cufilt(name: str):
    res = tv.cufilt(name)
    if res:
        return res 
    else:
        # if fail, use cu++filt.
        # in windows, gcc-style demangle isn't available.
        # so we can only use subprocess before cuda 11.4.
        res = subprocess.check_output(["cu++filt", name]).decode("utf-8").strip()
        return res 

def _lazy_load_llvm():
    import llvmlite.binding as llvm
    global LLVM_IS_INITED
    if not LLVM_IS_INITED:
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()  # yes, even this one
        LLVM_IS_INITED = True 

def create_execution_engine():
    """
    Create an ExecutionEngine suitable for JIT code generation on
    the host CPU.  The engine is reusable for an arbitrary number of
    modules.
    """
    import llvmlite.binding as llvm
    _lazy_load_llvm()
    # Create a target machine representing the host
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    # And an execution engine with an empty backing module
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def compile_ir(engine, llvm_ir):
    """
    Compile the LLVM IR string with the given engine.
    The compiled module object is returned.
    """
    import llvmlite.binding as llvm
    _lazy_load_llvm()
    # Create a LLVM module object from the IR
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    # Now add the module and make sure it is ready for execution
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod

def compile_bitcode(engine, llvm_bitcode):
    """
    Compile the LLVM IR string with the given engine.
    The compiled module object is returned.
    """
    import llvmlite.binding as llvm
    _lazy_load_llvm()
    # Create a LLVM module object from the IR
    mod = llvm.parse_bitcode(llvm_bitcode)
    mod.verify()
    # Now add the module and make sure it is ready for execution
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod

@dataclasses.dataclass
class NVRTCModuleParams:
    code: str
    headers: Dict[str, str]
    opts: List[str]
    name_exprs: List[str]
    name_to_meta: Dict[str, tv.NVRTCKernelMeta]
    program_name: str = "kernel.cu"
    cudadevrt_path: str = ""
    includes: List[str] = dataclasses.field(default_factory=list)

def create_nvrtc_code(cus: List[pccm.Class],
                 namespace_root: Optional[Union[str, Path]] = None,
                 cudadevrt_path: str = "",
                 custom_names: Optional[List[str]] = None,
                 std: str = "c++14",
                 cpu_code: bool = False) -> NVRTCModuleParams:
    cg = CodeGenerator([])
    user_cus = cg.build_graph(cus, namespace_root)
    # iterate cus and get all kernels
    name_to_meta: Dict[str, tv.NVRTCKernelMeta] = {}
    for cu in user_cus:
        cu_ns = cu.canonical_namespace
        for decl in cu._function_decls:
            meta = decl.meta
            assert meta.name is not None
            meta_cls = ExternalFunctionMeta
            if not cpu_code:
                meta_cls = CudaGlobalFunctionMeta
            if isinstance(meta, meta_cls):
                if decl.code.is_template():
                    # don't support template kernel
                    continue
                # is global function. firstly check types
                meta = tv.NVRTCKernelMeta(meta.name, cu_ns,
                                            decl.code.arguments)
                func_qualname = meta.name
                if cu_ns:
                    func_qualname = f"{cu_ns}::{meta.name}"
                name_to_meta[func_qualname] = meta

    # generate code for nvrtc
    header_dict, _, _ = cg.code_generation(user_cus,
                                            global_header_only=True)
    header_code_dict = {k: v.to_string() for k, v in header_dict.items()}
    final_code = ""
    for k, _ in header_dict.items():
        final_code += f"#include <{k}>\n"
    extern_build_meta = BuildMeta()
    for cu in user_cus:
        extern_build_meta += cu.build_meta
    # this function will call driver init
    opts = [f"-std={std}"]
    global_cflags = extern_build_meta.get_global_cflags()
    local_cflags = extern_build_meta.get_local_cflags()
    if not cpu_code:
        arch = tv.get_compute_capability()
        opts.append(f"--gpu-architecture=sm_{arch[0]}{arch[1]}")
        if cudadevrt_path and Path(cudadevrt_path).exists():
            opts.append("--relocatable-device-code=true")
        if "nvcc" in global_cflags:
            opts.extend(global_cflags["nvcc"])
        if "nvcc" in local_cflags:
            opts.extend(local_cflags["nvcc"])
    else:
        if "g++" in global_cflags:
            opts.extend(global_cflags["g++"])
        if "clang++" in global_cflags:
            opts.extend(global_cflags["clang++"])
        if "g++" in local_cflags:
            opts.extend(local_cflags["g++"])
        if "clang++" in local_cflags:
            opts.extend(local_cflags["clang++"])
    # print(opts)
    opts = list(set(opts))

    includes = []
    for inc in extern_build_meta.get_global_includes():
        opts.append("-I")
        opts.append(str(inc))
        includes.append(str(inc))
    for inc in extern_build_meta.get_local_includes():
        opts.append("-I")
        opts.append(str(inc))
        includes.append(str(inc))
    name_exprs = list(name_to_meta.keys())
    if custom_names is not None:
        name_exprs.extend(custom_names)
    return NVRTCModuleParams(final_code, header_code_dict,
        opts, name_exprs, name_to_meta, "kernel.cu", cudadevrt_path,
        includes=includes)

class CummNVRTCModuleBase(tv.NVRTCModule):
    def __init__(self,
                 code: str,
                 headers: Optional[Dict[str, str]] = None,
                 opts: Optional[List[str]] = None,
                 program_name: str = "kernel.cu",
                 name_exprs: Optional[List[str]] = None,
                 name_to_meta: Optional[Dict[str, tv.NVRTCKernelMeta]] = None,
                 cudadevrt_path: str = "") -> None:
        super().__init__(code,
                         headers,
                         opts,
                         program_name=program_name,
                         name_exprs=name_exprs,
                         name_to_meta=name_to_meta,
                         cudadevrt_path=cudadevrt_path)
        # extract meta data from ptx
        self.kernel_metas = list(name_to_meta.values())
        ptx = self.program.ptx()
        const_values: Dict[str, int] = {}
        for line in ptx.split("\n"):
            line = line.strip()
            if line.startswith(".global .align"):
                parts = line.split(" ")
                name = parts[4]
                if "[" in name:
                    continue
                if len(parts) == 7:
                    name = cufilt(name)
                    if not name:
                        continue
                    name = "::".join(name.split("::")[1:])
                    const_values[name] = int(parts[-1].replace(";", " "))
                elif len(parts) == 5:
                    name = cufilt(name[:-1])
                    if not name:
                        continue
                    name = "::".join(name.split("::")[1:])
                    const_values[name] = 0
            elif line.startswith(".visible .const .align"):
                parts = line.split(" ")
                name = parts[5]
                if "[" in name:
                    continue
                if len(parts) == 8:
                    name = cufilt(name)
                    if not name:
                        continue
                    const_values[name] = int(parts[-1].replace(";", " "))
                elif len(parts) == 6:
                    name = cufilt(name[:-1])
                    if not name:
                        continue
                    const_values[name] = 0
            elif line.startswith(".const .align"):
                parts = line.split(" ")
                name = parts[4]
                if "[" in name:
                    continue
                if len(parts) == 7:
                    name = cufilt(name)
                    if not name:
                        continue
                    const_values[name] = int(parts[-1].replace(";", " "))
                elif len(parts) == 5:
                    name = cufilt(name[:-1])
                    if not name:
                        continue
                    const_values[name] = 0


        self.const_values = const_values

    @classmethod
    def from_params(cls, params: NVRTCModuleParams):
        return cls(params.code,
                params.headers,
                params.opts,
                name_exprs=params.name_exprs,
                name_to_meta=params.name_to_meta,
                cudadevrt_path=params.cudadevrt_path)

    @property
    def kernels(self):
        assert self.name_to_meta is not None
        return list(self.name_to_meta.keys())

class CummNVRTCModule(CummNVRTCModuleBase):
    def __init__(self,
                 cus: List[pccm.Class],
                 namespace_root: Optional[Union[str, Path]] = None,
                 verbose: bool = False,
                 cudadevrt_path: str = "",
                 custom_names: Optional[List[str]] = None,
                 verbose_path: str = "",
                 std: str = "c++14") -> None:
        mod_params = create_nvrtc_code(cus, namespace_root, cudadevrt_path, custom_names, std)
        if verbose:
            if verbose_path:
                verbose_path_p = Path(verbose_path)
                for k, v in mod_params.headers.items():
                    code_path = verbose_path_p / k
                    code_path.parent.mkdir(exist_ok=True, parents=True)
                    with code_path.open("w") as f:
                        f.write(v)
            else:
                for k, v in mod_params.headers.items():
                    print(k)
                    print(v)
        super().__init__(mod_params.code,
                         mod_params.headers,
                         mod_params.opts,
                         name_exprs=mod_params.name_exprs,
                         name_to_meta=mod_params.name_to_meta,
                         cudadevrt_path=cudadevrt_path)

class CummLLVMModule:
    def __init__(self,
                 cus: List[pccm.Class],
                 namespace_root: Optional[Union[str, Path]] = None,
                 verbose: bool = False,
                 verbose_path: str = "",
                 std: str = "c++17") -> None:
        self.mod_params: NVRTCModuleParams = create_nvrtc_code(cus, namespace_root, "", [], std, cpu_code=True)
        if verbose:
            if verbose_path:
                verbose_path_p = Path(verbose_path)
                for k, v in self.mod_params.headers.items():
                    code_path = verbose_path_p / k
                    code_path.parent.mkdir(exist_ok=True, parents=True)
                    with code_path.open("w") as f:
                        f.write(v)
            else:
                for k, v in self.mod_params.headers.items():
                    print(k)
                    print(v)
        self._llvm_mod: Optional[Any] = None
        self._llvm_engine: Optional[Any] = None
        self._use_llvm_bit_code = True

        self.name_to_meta = self.mod_params.name_to_meta

    def load(self):
        import llvmlite.binding as llvm
        _lazy_load_llvm()
        # use clang++ to get ir
        opts = self.mod_params.opts

        with tempfile.TemporaryDirectory() as fdir:
            inc_dir = Path(fdir) / "include"
        
            for k, v in self.mod_params.headers.items():
                code_path = Path(inc_dir) / k
                code_path.parent.mkdir(exist_ok=True, parents=True)
                with code_path.open("w") as f:
                    f.write(v)
            print(inc_dir)
            out_name = Path(fdir) / "ir.ll"
            with tempfile.NamedTemporaryFile("w", suffix=".cc") as f2:
                f2.seek(0)
                f2.write(self.mod_params.code)
                f2.flush()
                # print(["clang++", "-S", "-emit-llvm", f2.name, *opts, "-o", f.name])
                # subprocess.check_output(["clang++", "-S", "-emit-llvm", "/home/yy/Projects/cumm/cumm/nvrtc/wtf.cc", "-o", "wtf.o"])
                # breakpoint()
                print(opts)
                if self._use_llvm_bit_code:
                    subprocess.check_output(["clang++", "-emit-llvm", "-c", f2.name, *opts, "-O3", "-I", str(inc_dir), "-o", str(out_name)])
                else:
                    subprocess.check_output(["clang++", "-S", "-emit-llvm", "-c", f2.name, *opts, "-O3", "-I", str(inc_dir), "-o", str(out_name)])
                breakpoint()

                # breakpoint()
                # print(1)
            # read ir and pass to llvmlite
            with out_name.open("rb" if self._use_llvm_bit_code else "r") as f:
                llvm_ir = f.read()
        # print(llvm_ir)
        # breakpoint()

        self._llvm_engine = create_execution_engine()
        if self._use_llvm_bit_code:
            self._llvm_mod = compile_bitcode(self._llvm_engine, llvm_ir)
        else:
            self._llvm_mod = compile_ir(self._llvm_engine, llvm_ir)

        # breakpoint()
    
    def _get_loaded_llvm_mod(self):
        assert self._llvm_mod is not None 
        return self._llvm_mod

    def _get_loaded_llvm_engine(self):
        assert self._llvm_engine is not None 
        return self._llvm_engine

    def run_kernel(self, name: str, launch: tv.LaunchParam,
                   *args: Union[Tensor, int, float, List[int], List[float],
                                Tuple[float, ...], Tuple[int, ...]]):
        if self._llvm_mod is None:
            self.load()
        metas: List[tv.NVRTCArgMeta] = [tv.NVRTCArgMeta(False, -1, [])] * len(args)
        if self.name_to_meta:
            assert name in self.name_to_meta, f"can't find your kernel {name}, available: {self.name_to_meta.keys()}"
            assert len(args) == len(self.name_to_meta[name].args)
            metas = self.name_to_meta[name].arg_metas        
        from pccm.builder.inliner import PCCM_INLINE_FUNCTION_NAME
        mod = self._get_loaded_llvm_mod()
        func_mangle_name = ""
        for func in mod.functions:
            if PCCM_INLINE_FUNCTION_NAME in func.name:
                func_mangle_name = func.name
        assert func_mangle_name != ""
        func_ptr = self._get_loaded_llvm_engine().get_function_address(func_mangle_name)
        # breakpoint()
        assert func_ptr != 0, f"get function {name} failed."
        # cfunc = ctypes.CFUNCTYPE(c_double, c_double, c_double)(func_ptr)
        cfunc_types = []
        cfunc_args = []
        for arg, meta in zip(args, metas):
            if meta.valid:
                # print(meta.shape)
                if meta.is_simple_ptr:
                    if not isinstance(arg, Tensor):
                        raise ValueError("your arg must be tensor")
                    if not arg.dtype == meta.simple_type:
                        cur_dtype = tv.get_npdtype_from_tvdtype(arg.dtype)
                        expected_dtype = tv.get_npdtype_from_tvdtype(
                            meta.simple_type)
                        raise ValueError(
                            f"your tensor {arg.shape}|{cur_dtype}"
                            f" dtype not equal to {expected_dtype}")
                    cfunc_types.append(ctypes.c_void_p)
                    if arg.empty():
                        cfunc_args.append(0)
                    else:
                        cfunc_args.append(arg.byte_pointer())
                    continue
                else:
                    # we can't ensure arg isn't tv::Tensor.
                    if not isinstance(arg, Tensor):
                        if not meta.is_scalar:
                            assert not isinstance(arg, Tensor)
                            dtype = tv.get_npdtype_from_tvdtype(meta.simple_type)
                            arg_array = np.array(arg, dtype=dtype)
                            if not arg_array.shape:
                                arg_array = arg_array.reshape(1)
                            assert list(arg_array.shape) == meta.shape
                            ctype = np.ctypeslib.as_ctypes_type(dtype) * arg_array.size
                            cfunc_args.append(ctype(*arg_array.reshape(-1)))
                            # auto dtype cast
                            # TODO prevent floats assigned to ints
                            cfunc_types.append(ctype)
                            continue
                        else:
                            assert not isinstance(arg, Tensor)
                            dtype = tv.get_npdtype_from_tvdtype(meta.simple_type)
                            ctype = np.ctypeslib.as_ctypes_type(dtype)
                            cfunc_types.append(ctype)
                            cfunc_args.append(arg)
                            continue
            # meta isn't valid, use regular dtypes.
            if isinstance(arg, (int, float)):
                ctype = ctypes.c_float
                if isinstance(arg, int):
                    ctype = ctypes.c_int64
                cfunc_types.append(ctype)
                cfunc_args.append(arg)
            elif isinstance(arg, (list, tuple)):
                dtype = np.float32
                if isinstance(arg[0], int):
                    dtype = np.int64
                arg_np = np.array(arg, dtype=dtype)
                ctype = np.ctypeslib.as_ctypes_type(dtype) * arg_np.size
                cfunc_args.append(ctype(*arg_np))
                # auto dtype cast
                # TODO prevent floats assigned to ints
                cfunc_types.append(ctype)
            else:
                assert isinstance(arg, Tensor)
                cfunc_types.append(ctypes.c_void_p)
                if arg.empty():
                    cfunc_args.append(0)
                else:
                    cfunc_args.append(arg.byte_pointer())
        # print(cfunc_types, cfunc_args, func_ptr)
        # breakpoint()
        cfunc_type = ctypes.CFUNCTYPE(ctypes.c_int, *cfunc_types)(func_ptr)
        res = cfunc_type(*cfunc_args)
        # print(res)

        return res 

