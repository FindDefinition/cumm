from typing import Dict, List, Optional, Union
from pccm import Argument
import pccm
from pathlib import Path
from pccm.targets.cuda import CudaGlobalFunctionMeta
from pccm.core import CodeGenerator
from pccm.core.buildmeta import BuildMeta

from cumm import tensorview as tv
from cumm.common import TensorViewKernel


class CummNVRTCModule(tv.NVRTCModule):
    def __init__(self,
                 cus: List[pccm.Class],
                 namespace_root: Optional[Union[str, Path]] = None,
                 verbose: bool = False) -> None:
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
        opts = ["-std=c++14"]
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
        super().__init__(final_code,
                         header_code_dict,
                         opts,
                         name_exprs=name_exprs,
                         name_to_meta=name_to_meta)
    
    
    @property
    def kernels(self):
        assert self.name_to_meta is not None 
        return list(self.name_to_meta.keys())