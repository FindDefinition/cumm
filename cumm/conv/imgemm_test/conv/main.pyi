from typing import (Any, Callable, Dict, List, Optional, Set, Tuple, Type,
                    Union, overload)

from pccm.stubs import EnumClassValue, EnumValue

from cumm.tensorview import Tensor

class ConvMainUnitTest:
    @staticmethod
    def implicit_gemm(input: Tensor, weight: Tensor, output: Tensor, padding: List[int], stride: List[int], dilation: List[int], ndim: int, iter_algo_: int, op_type_: int, i_ltype_: int, w_ltype_: int, o_ltype_: int, ts: Tuple[int, int, int], wts: Tuple[int, int, int], num_stage: int, dacc: int, dcomp: int, algo: str, tensorop: List[int], i_interleave: int = 1, w_interleave: int = 1, o_interleave: int = 1, alpha: float = 1, beta: float = 0, split_k_slices: int = 1, workspace: Tensor =  Tensor(), mask_sparse: bool = False, increment_k_first: bool = False, mask: Tensor =  Tensor(), mask_argsort: Tensor =  Tensor(), indices: Tensor =  Tensor(), mask_output: Tensor =  Tensor()) -> None: 
        """
        Args:
            input: 
            weight: 
            output: 
            padding: 
            stride: 
            dilation: 
            ndim: 
            iter_algo_: 
            op_type_: 
            i_ltype_: 
            w_ltype_: 
            o_ltype_: 
            ts: 
            wts: 
            num_stage: 
            dacc: 
            dcomp: 
            algo: 
            tensorop: 
            i_interleave: 
            w_interleave: 
            o_interleave: 
            alpha: 
            beta: 
            split_k_slices: 
            workspace: 
            mask_sparse: 
            increment_k_first: 
            mask: 
            mask_argsort: 
            indices: 
            mask_output: 
        """
        ...