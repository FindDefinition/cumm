import os
from pathlib import Path
from typing import List

PACKAGE_NAME = "cumm"
PACKAGE_ROOT = Path(__file__).parent.resolve()

_TENSORVIEW_INCLUDE_PATHS: List[Path] = [
    PACKAGE_ROOT.parent / "include",  # pip dev install
    PACKAGE_ROOT / "include",  # pip package
]

TENSORVIEW_INCLUDE_PATH = _TENSORVIEW_INCLUDE_PATHS[0]
if not TENSORVIEW_INCLUDE_PATH.exists():
    for p in _TENSORVIEW_INCLUDE_PATHS[1:]:
        if p.exists():
            TENSORVIEW_INCLUDE_PATH = p

assert TENSORVIEW_INCLUDE_PATH.exists()

CUTLASS_MODE = False
CUTLASS_INPUT_ITER = CUTLASS_MODE and True
CUTLASS_SMEM_WARP_ITER = CUTLASS_MODE and True
CUTLASS_OUTPUT_ITER = CUTLASS_MODE and True
CUTLASS_DEBUG = False
