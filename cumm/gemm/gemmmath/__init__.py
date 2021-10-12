import numpy as np
import pccm

from cumm import dtypes
from cumm.common import TensorView
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import core


class Clamp(pccm.ParameterizedClass):
    def __init__(self, dtype: dtypes.DType, out_dtype: dtypes.DType,
                 num_element: int):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_include("tensorview/gemm/math/all.h")
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.num_element = num_element

    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               name="operator()")
    def call_operator(self):
        code = pccm.FunctionCode()
        argument_t = core.array_type(self.dtype, self.num_element)
        code.arg("src", f"const {argument_t} &")
        code.raw(f"""
        constexpr {self.dtype} kClamp = {self.dtype}((1U << (sizeof({self.out_dtype}) * 8 - 1)) - 1);
        tv::math::minimum<{argument_t}> min_op;
        tv::math::maximum<{argument_t}> max_op;
        {argument_t} intermediate = max_op(src, -kClamp - {self.dtype}(1));
        intermediate = min_op(intermediate, kClamp);
        return intermediate;
        """)
        code.ret(argument_t)
        return code

    def python_ctor(self):
        return self

    def __call__(self, src: ArrayPtr):
        kClamp = int(((1 << (self.out_dtype.itemsize() * 8 - 1)) - 1))
        data = src.data.numpy_view()
        data = np.maximum(data, -kClamp - 1)
        intermediate = src.copy()
        intermediate_data = intermediate.data.numpy_view()
        intermediate_data[:] = np.minimum(data, kClamp)
        return intermediate


class UnaryIdentity(pccm.ParameterizedClass):
    def __init__(self, dtype: dtypes.DType, out_dtype: dtypes.DType,
                 num_element: int):
        super().__init__()
        self.add_dependency(TensorView)
        self.add_include("tensorview/gemm/math/all.h")
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.num_element = num_element

    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               name="operator()")
    def call_operator(self):
        code = pccm.FunctionCode()
        argument_t = core.array_type(self.dtype, self.num_element)
        code.arg("src", f"const {argument_t} &")
        code.raw(f"""
        return src;
        """)
        code.ret(argument_t)
        return code

    def python_ctor(self):
        return self

    def __call__(self, src: ArrayPtr):
        return src.copy()
