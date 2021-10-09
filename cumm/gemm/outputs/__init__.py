from cumm.gemm.bases import GemmInputIterator, GemmOutSmemLoader, GemmOutWarpIterator, GemmOutputIterator, GemmOutputOp, GemmSmemIterator, GemmWarpIterator
from cumm import tensorview as tv 
from cumm.core_cc.csrc.arrayref import ArrayPtr
import numpy as np
import pccm
from cumm import dtypes
from cumm.constants import CUTLASS_MODE
from cumm import cudasim
from cumm.gemm import (constants, layout, mask_iters,
                         out_iters, thread_map, volta_iters, volta_out_iters)
from cumm.common import (GemmBasic, GemmBasicKernel, TensorView,
                                TensorViewKernel)
from typing import Dict, List, Union, Optional, Type, Tuple
from cumm.gemm.core import metaseq, seq, MetaArray, array_type

from cumm.gemm.algospec import bases

def div_up(a, b):
    return (a + b - 1) // b

class OutputSmemStorage(pccm.ParameterizedClass):
    def __init__(self, shape: MetaArray[int], smem_padding_mn: MetaArray[int],
                 dtype_acc: dtypes.DType, frag_per_iter: int):
        super().__init__()
        self.add_dependency(TensorView)
        self.shape = shape
        self.smem_padding_mn = smem_padding_mn

        self.dtype_acc = dtype_acc
        self.frag_per_iter = frag_per_iter
        self.storage_shape = seq((shape[0] + smem_padding_mn[0]) * frag_per_iter, shape[1] + smem_padding_mn[1])
        self.smem_alignment = 16

        self.smem_size = self.storage_shape.prod() * dtype_acc.itemsize()
        self.smem_size = div_up(self.smem_size, self.smem_alignment) * self.smem_alignment

        self.add_member(
            "smem",
            f"tv::alignedarray<{dtype_acc}, {self.storage_shape.prod()}, 16>")



class Output(pccm.ParameterizedClass):
    def __init__(self, dtype_acc: dtypes.DType, warp_count_shape: MetaArray[int], partk: int, spec: bases.Output, smem_storage: OutputSmemStorage):
        super().__init__()
        self.spec = spec 
        self.dtype_acc = dtype_acc 
        self.smem_storage = smem_storage 
        self.warp_count_shape = warp_count_shape
        self.add_param_class("out_ns_frag", spec.frag_iter, "FragIter")
        self.add_param_class("out_ns_warp", spec.warp_store_iter, "OutWarpIter")
        self.add_param_class("out_ns_smem", spec.smem_loader, "SmemLoader")
        self.add_param_class("out_ns_out", spec.out_iter, "OutIter")
        self.add_param_class("out_ns_out_const", spec.const_out_iter, "ConstOutIter")

        self.add_param_class("out_ns_sto", smem_storage, "OutputStorage")
        self.add_param_class("out_ns_op", spec.output_op, "OutputOp")
        self.add_param_class("out_ns_apply", spec.apply_op, "ApplyOp")
        self.accumulator_fragment = array_type(dtype_acc, spec.mma_spec.accumulator_size)
        self.partk=  partk
        self.out_num_tile = self.spec.frag_per_iter if self.spec.frag_per_iter > 1 else partk
        self.out_tile_size = smem_storage.smem_size // dtype_acc.itemsize() // self.out_num_tile
        # print(smem_storage.smem_size, self.out_tile_size, self.out_num_tile)
        # raise NotImplementedError

        self.add_member("warp_iter", "OutWarpIter")
        self.add_member("smem_loader", "SmemLoader")
        # cudasim
        self.warp_iter : Optional[GemmOutWarpIterator] = None 
        self.smem_loader : Optional[GemmOutSmemLoader] = None 

    @pccm.cuda.constructor(device=True, forceinline=True)
    def ctor(self):
        code = pccm.FunctionCode()
        code.arg("smem_storage", "OutputStorage*")
        code.arg("thread_idx,warp_idx_k,warp_m,warp_n,lane_idx", "int")
        code.ctor_init("warp_iter", f"smem_storage->smem.data(), warp_idx_k * {self.warp_count_shape[0]} + warp_m, warp_n, lane_idx")
        code.ctor_init("smem_loader", f"smem_storage->smem.data(), thread_idx")
        return code 

    def python_ctor(self, smem_ptr: ArrayPtr, thread_idx: int, warp_idx_k: int, warp_m: int, warp_n: int, lane_idx: int):
        new_obj = Output(self.dtype_acc, self.warp_count_shape, self.partk, self.spec, self.smem_storage)
        new_obj.warp_iter = self.spec.warp_store_iter.python_ctor(smem_ptr, warp_idx_k * self.warp_count_shape[0] + warp_m, warp_n, lane_idx)
        new_obj.smem_loader = self.spec.smem_loader.python_ctor(smem_ptr, thread_idx)
        return new_obj 

    def call_template(self, have_source: bool, self_reduce: bool):
        code = pccm.FunctionCode()
        code.arg("output_op", f"OutputOp const&")
        code.arg("accumulators", f"{self.accumulator_fragment} const&")
        code.arg("out_iter", f"OutIter&")
        if have_source:
            code.arg("source_iter", f"ConstOutIter&")

        if CUTLASS_MODE:
            platform_math = "cutlass"
        else:
            platform_math = "tv::math"
        if have_source:
            code.raw(f"""
            if (!output_op.is_source_needed()){{
                return run(output_op, accumulators, out_iter);
            }}
            {self.spec.out_iter.fragment_t} source_frag;
            source_frag.clear();
            """)
        elif self_reduce:
            code.raw(f"""
            {self.spec.out_iter.fragment_t} source_frag;
            source_frag.clear();
            """)
        code.raw(f"FragIter out_acc_iter(accumulators.data());")

        with code.range_("iter", str(self.spec.num_out_iters), "TV_PRAGMA_UNROLL"):
            if have_source:
                code.raw(f"""
                source_iter.load(source_frag);
                ++source_iter;
                """)
            elif self_reduce:
                code.raw(f"""
                out_iter.load(source_frag);
                """)
            code.raw(f"""
        
            __syncthreads();
            TV_PRAGMA_UNROLL
            for (int p = 0; p < {self.spec.frag_per_iter}; ++p){{
                {self.spec.frag_iter.fragment_t} acc_frag;
                out_acc_iter.load(acc_frag);
                ++out_acc_iter;
                warp_iter.store(acc_frag);
                if (p < {self.spec.frag_per_iter} - 1){{
                    warp_iter.add_pointer_offset({self.out_tile_size});
                }}
            }}
            if ({self.spec.frag_per_iter} > 1){{
                warp_iter.add_pointer_offset({self.out_tile_size * (1 - self.spec.frag_per_iter)});
            }}

            __syncthreads();
            """)
            with code.range_("p", self.spec.frag_per_iter, "TV_PRAGMA_UNROLL"):
                code.raw(f"""
                {self.spec.smem_loader.fragment_t} smem_frags[{self.partk}];
                smem_loader.load(smem_frags[0]);

                if (p < {self.spec.frag_per_iter} - 1){{
                    smem_loader.add_pointer_offset({self.out_tile_size});
                }}
                else if ({self.partk} > 1){{
                    TV_PRAGMA_UNROLL
                    for (int partk_idx = 1; partk_idx < {self.partk}; ++partk_idx){{
                        smem_loader.add_pointer_offset({self.out_tile_size});
                        smem_loader.load(smem_frags[partk_idx]);
                        {platform_math}::plus<{self.spec.smem_loader.fragment_t}> accer;
                        smem_frags[0] = accer(smem_frags[0], smem_frags[partk_idx]);
                    }}
                    smem_loader.add_pointer_offset({self.out_tile_size * (1 - self.partk)});
                }}
                {self.spec.out_iter.fragment_t} out_frag;
                """)
                if have_source or self_reduce:
                    code.raw(f"""
                    ApplyOp::apply_output_operator(out_frag, output_op, smem_frags[0], source_frag);
                    """)
                else:
                    code.raw(f"""
                    ApplyOp::apply_output_operator_no_source(out_frag, output_op, smem_frags[0]);
                    """)
                code.raw(f"""
                out_iter.store(out_frag);
                ++out_iter;
                """)
            code.raw(f"""
            if ({self.spec.frag_per_iter} > 1){{
                smem_loader.add_pointer_offset({self.out_tile_size * (1 - self.spec.frag_per_iter)});
            }}
            """)
        return code 


    @pccm.cuda.member_function(device=True, forceinline=True, name="run")
    def run_source(self):
        return self.call_template(True, False) 

    @pccm.cuda.member_function(device=True, forceinline=True, name="run")
    def run_no_source(self):
        return self.call_template(False, False) 

    @pccm.cuda.member_function(device=True, forceinline=True, name="run_self_reduce")
    def run_self_reduce(self):
        return self.call_template(False, True) 

    async def __call__(self, output_op: GemmOutputOp, accumulators: ArrayPtr, out_iter: GemmOutputIterator, source_iter: Optional[GemmOutputIterator] = None, self_reduce: bool = False):
        out_warp_iter = self.warp_iter
        out_smem_loader = self.smem_loader
        assert out_warp_iter is not None 
        assert out_smem_loader is not None 
        source_frag = ArrayPtr(out_iter.dtype.tv_dtype,
                                out_iter.element_count)
        if source_iter is not None:
            source_frag.clear()

        out_acc_iter = self.spec.frag_iter.python_ctor(accumulators)
        smem_save_list = []
        smem_load_list = []
        for out_idx in range(0, self.spec.num_out_iters, self.spec.frag_per_iter):
            if source_iter is not None:
                source_iter.load_python(source_frag)
                source_iter.increment_python()
            await cudasim.syncthreads()
            for p in range(self.spec.frag_per_iter):
                acc_frag = ArrayPtr(self.dtype_acc.tv_dtype,
                                    self.spec.frag_iter.element_count)
                out_acc_iter.load_python(acc_frag)
                out_acc_iter.increment_python()
                ptrs = await out_warp_iter.store_python(acc_frag)
                if out_idx == 0:
                    smem_save_list.append(ptrs)
                if p != self.spec.frag_per_iter - 1:
                    out_warp_iter.add_pointer_offset_python(self.out_tile_size)
            if self.spec.frag_per_iter > 1:
                out_warp_iter.add_pointer_offset_python(self.out_tile_size * (1 - self.spec.frag_per_iter))
            
            await cudasim.syncthreads()
            # if cudasim.threadIdx().x == 0:
            #     smem_data = smem_out_ptr.data.numpy_view()[:64]
            #     print("SMEMM", smem_data[:16])
            #     print("SMEM", smem_data.mean(), smem_data.max(), smem_data.min())
            for p in range(self.spec.frag_per_iter):
                smem_frags = [ArrayPtr(self.dtype_acc.tv_dtype,
                                    out_smem_loader.element_count) for _ in range(self.partk)]
                ptrs = await out_smem_loader.load_python(smem_frags[0])
                if out_idx == 0:

                    smem_load_list.append(ptrs)

                if p != self.spec.frag_per_iter - 1:
                    out_smem_loader.add_pointer_offset_python(self.out_tile_size)
                elif self.partk > 1:
                    for partk_idx in range(1, self.partk):

                        out_smem_loader.add_pointer_offset_python(self.out_tile_size)
                        ptrs = await out_smem_loader.load_python(smem_frags[partk_idx])
                        if out_idx == 0:

                            smem_load_list.append(ptrs)
                        data_i = smem_frags[partk_idx].data.numpy_view()
                        smem_frags[0].data.numpy_view()[:] += data_i
                    out_smem_loader.add_pointer_offset_python(self.out_tile_size * (1 - self.partk))
                out_frag = ArrayPtr(out_iter.dtype.tv_dtype,
                                    out_iter.element_count)
                self.spec.apply_op.apply_output_operator_no_source_python(out_frag, output_op, smem_frags[0])
                out_iter.store_python(out_frag)
                out_iter.increment_python()
            if self.spec.frag_per_iter > 1:
                out_smem_loader.add_pointer_offset_python(self.out_tile_size * (1 - self.spec.frag_per_iter))
        res = {
            "Output": {
                "smem_save_coords": smem_save_list,
                "smem_load_coords": smem_load_list,
                "smem_save_epa": out_warp_iter.element_per_acc,
                "smem_load_epa": out_smem_loader.element_per_acc,
                "smem_shape": self.smem_storage.storage_shape,
            }
        }
        return res 
