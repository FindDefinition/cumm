from typing import Dict, List, Optional, Tuple, Type, Union

from .core import GemmAlgo
from .simt import AlgoSpecificSimt
from .turing import AlgoSpecificTuring
from .volta import AlgoSpecificVolta

ALGO_SPEC_TYPES = Union[AlgoSpecificSimt, AlgoSpecificVolta,
                        AlgoSpecificTuring]

ALGO_TO_SPEC = {
    GemmAlgo.Simt: AlgoSpecificSimt,
    GemmAlgo.SimtDP4A: AlgoSpecificSimt,
    GemmAlgo.Volta: AlgoSpecificVolta,
    GemmAlgo.Turing: AlgoSpecificTuring,
}  # type: Dict[GemmAlgo, Union[Type[AlgoSpecificSimt], Type[AlgoSpecificVolta], Type[AlgoSpecificTuring]]]


def get_algo_spec(algo: GemmAlgo):
    return ALGO_TO_SPEC[algo]
