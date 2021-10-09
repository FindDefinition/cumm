from cumm.dtypes import DType
from typing import Union
from .metaarray import seq, MetaArray, metaseq
from cumm.constants import CUTLASS_MODE

def array_type(dtype: Union[str, DType], count: int, align: int = 0):
    if not CUTLASS_MODE:
        return f"tv::array<{dtype}, {count}, {align}>"
    else:
        return f"cutlass::Array<{dtype}, {count}>"


def aligned_array_type(dtype: Union[str, DType], count: int, align: int = 0):
    if not CUTLASS_MODE:
        return f"tv::alignedarray<{dtype}, {count}, {align}>"
    else:
        return f"cutlass::AlignedArray<{dtype}, {count}, {align}>"