import pccm

from cumm.common import TensorView
from cumm.gemm.core.metaarray import MetaArray


class GemmUtils(pccm.ParameterizedClass):
    def __init__(self, tile_shape: MetaArray[int]):
        super().__init__()
        self.add_dependency(TensorView)
        self.tile_shape = tile_shape

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def get_gemm_k_size_per_split(self):
        """get gemm per split k
        first we need to get iterations by tile shape k,
        
        """
        code = pccm.FunctionCode()
        code.arg("k, split_k", "int")
        code.raw(f"""
        int total_gemm_k_iterations = tv::div_up(k, {self.tile_shape[2]});
        int gemm_k_iterations_per_split =
            tv::div_up(total_gemm_k_iterations, split_k);
        auto gemm_k_size_per_split = gemm_k_iterations_per_split * {self.tile_shape[2]}; 
        return gemm_k_size_per_split;
        """)
        return code.ret("int")

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def get_gemm_k_bound(self):
        code = pccm.FunctionCode()
        code.arg("k, gemm_k_size_per_split, tile_offset_k", "int")
        code.raw(f"""
        int k_bound = min(k, (tile_offset_k + 1) * gemm_k_size_per_split);
        return k_bound;
        """)
        return code.ret("int")

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def get_gemm_iterations(self):
        code = pccm.FunctionCode()
        code.arg("k_bound, gemm_k_size_per_split, tile_offset_k", "int")
        code.raw(f"""
        int gemm_k_iterations =
            tv::div_up(k_bound - tile_offset_k * gemm_k_size_per_split, {self.tile_shape[2]});
        return gemm_k_iterations;
        """)
        return code.ret("int")
