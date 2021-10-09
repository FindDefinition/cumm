from typing import Dict, List, Union, Optional, Type, Tuple
from .simt import AlgoSpecificSimt
from .core import GemmAlgo
from .volta import AlgoSpecificVolta
from .turing import AlgoSpecificTuring

ALGO_SPEC_TYPES = Union[AlgoSpecificSimt, AlgoSpecificVolta, AlgoSpecificTuring]

ALGO_TO_SPEC = {
    GemmAlgo.Simt: AlgoSpecificSimt,
    GemmAlgo.SimtDP4A: AlgoSpecificSimt,
    GemmAlgo.Volta: AlgoSpecificVolta,
    GemmAlgo.Turing: AlgoSpecificTuring,
}  # type: Dict[GemmAlgo, Union[Type[AlgoSpecificSimt], Type[AlgoSpecificVolta], Type[AlgoSpecificTuring]]]

def get_algo_spec(algo: GemmAlgo):
    return ALGO_TO_SPEC[algo]
