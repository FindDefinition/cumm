from pathlib import Path
from typing import Dict, List, Optional, Union
from build import subprocess

import pccm
from ccimport import compat
from pccm import Argument
from pccm.core import CodeGenerator
from pccm.core.buildmeta import BuildMeta
from pccm.targets.cuda import CudaGlobalFunctionMeta

from cumm import PACKAGE_ROOT
from cumm import tensorview as tv
from cumm.common import TensorViewKernel, _get_cuda_include_lib

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
    # in windows, gcc-style demangle isn't available.
    # so we can only use subprocess before cuda 11.4.
    if compat.InWindows:
        res = subprocess.check_output(["cu++filt", name]).decode("utf-8").strip()
        return res 
    raise NotImplementedError

class CummNVRTCModule(tv.NVRTCModule):
    def __init__(self,
                 cus: List[pccm.Class],
                 namespace_root: Optional[Union[str, Path]] = None,
                 verbose: bool = False,
                 cudadevrt_path: str = "",
                 custom_names: Optional[List[str]] = None) -> None:
        cg = CodeGenerator([], verbose=verbose)
        user_cus = cg.build_graph(cus, namespace_root)
        # iterate cus and get all kernels
        self.kernel_metas: List[tv.NVRTCKernelMeta] = []
        name_to_meta: Dict[str, tv.NVRTCKernelMeta] = {}
        for cu in user_cus:
            cu_ns = cu.canonical_namespace
            for decl in cu._function_decls:
                meta = decl.meta
                assert meta.name is not None
                if isinstance(meta, CudaGlobalFunctionMeta):
                    if decl.code.is_template():
                        # don't support template kernel
                        continue
                    # is global functino. firstly check types
                    meta = tv.NVRTCKernelMeta(meta.name, cu_ns,
                                              decl.code.arguments)
                    self.kernel_metas.append(meta)
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
        arch = tv.get_compute_capability(0)
        opts = ["-std=c++14", f"--gpu-architecture=sm_{arch[0]}{arch[1]}"]
        if Path(cudadevrt_path).exists():
            opts.append("--relocatable-device-code=true")
        if "nvcc" in extern_build_meta.compiler_to_cflags:
            opts.extend(extern_build_meta.compiler_to_cflags["nvcc"])
        for inc in extern_build_meta.includes:
            opts.append("-I")
            opts.append(str(inc))
        if verbose:
            for k, v in header_code_dict.items():
                print(k)
                print(v)
        # print(header_code_dict)
        name_exprs = list(name_to_meta.keys())
        if custom_names is not None:
            name_exprs.extend(custom_names)
        # print(opts)
        # breakpoint()
        super().__init__(final_code,
                         header_code_dict,
                         opts,
                         name_exprs=name_exprs,
                         name_to_meta=name_to_meta,
                         cudadevrt_path=cudadevrt_path)
        # extract meta data from ptx
        ptx = self.program.ptx()
        const_values: Dict[str, int] = {}
        for line in ptx.split("\n"):
            line = line.strip()
            if line.startswith(".global .align"):
                parts = line.split(" ")
                name = parts[4]
                if len(parts) == 7:
                    name = cufilt(name)
                    name = "::".join(name.split("::")[1:])
                    const_values[name] = int(parts[-1].replace(";", " "))
        # print(const_values)
        # breakpoint()
        self.const_values = const_values

    @property
    def kernels(self):
        assert self.name_to_meta is not None
        return list(self.name_to_meta.keys())
