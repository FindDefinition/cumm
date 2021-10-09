from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
class GemmMainUnitTest:
    @staticmethod
    def matmul(a: Tensor, b: Tensor, c: Tensor, ta: bool, tb: bool, tc: bool, ts: Tuple[int, int, int], wts: Tuple[int, int, int], num_stage: int, dacc: int, dcomp: int, algo: str, tensorop: List[int], split_k_slices: int = 1, workspace: Tensor =  Tensor(), shuffle_type: str = "NS", a_inds: Tensor =  Tensor(), b_inds: Tensor =  Tensor(), c_inds: Tensor =  Tensor()) -> None: 
        """
        Args:
            a: 
            b: 
            c: 
            ta: 
            tb: 
            tc: 
            ts: 
            wts: 
            num_stage: 
            dacc: 
            dcomp: 
            algo: 
            tensorop: 
            split_k_slices: 
            workspace: 
            shuffle_type: 
            a_inds: 
            b_inds: 
            c_inds: 
        """
        ...
