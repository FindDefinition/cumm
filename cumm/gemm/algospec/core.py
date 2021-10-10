import enum
from typing import Tuple 
from cumm.gemm.core import metaseq, seq, MetaArray

class GemmAlgo(enum.Enum):
    Simt = "Simt"
    SimtDP4A = "SimtDP4A"
    SimtDP2A = "SimtDP2A"
    Volta = "Volta"
    Turing = "Turing"
    Ampere = "Ampere"

class ShuffleStrideType(enum.Enum):
    NoShuffle = "NS"
    # A and C have indices, for spatial spconv forward and backward input
    ShuffleAC = "SAC" 
    # A and B have indices, for spatial spconv backward weight
    ShuffleAB = "SAB" 

class TensorOpParams(object):
    def __init__(self, shape: Tuple[int, int, int]):
        self.shape = seq(*shape)

    def to_string(self):
        return f"{self.shape[0]}{self.shape[1]}{self.shape[2]}"


    def __getitem__(self, val: int):
        return self.shape[val]
