from typing import Dict, List, Optional, Tuple, Type, Union

from cumm.gemm.algospec import GemmAlgo

from .simt import AlgoSpecificSimt

# from .volta import AlgoSpecificVolta
# from .turing import AlgoSpecificTuring

ALGO_TO_SPEC = {
    GemmAlgo.Simt: AlgoSpecificSimt,
    GemmAlgo.SimtDP4A: AlgoSpecificSimt,
    # GemmAlgo.Volta: AlgoSpecificVolta,
    # GemmAlgo.Turing: AlgoSpecificTuring,
}  # type: Dict[GemmAlgo, Union[Type[AlgoSpecificSimt]]]


def get_algo_spec(algo: GemmAlgo):
    return ALGO_TO_SPEC[algo]
