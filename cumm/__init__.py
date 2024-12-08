# Copyright 2024 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pccm
from pccm.utils import project_is_editable, project_is_installed
import os 
from .__version__ import __version__
from .constants import CUMM_CPU_ONLY_BUILD, CUMM_DISABLE_JIT, PACKAGE_NAME
from cumm.constants import PACKAGE_ROOT
from ccimport import compat
from pathlib import Path 
import importlib.util 

if project_is_installed(PACKAGE_NAME) and project_is_editable(
        PACKAGE_NAME) and not CUMM_DISABLE_JIT:
    from cumm.csrc.arrayref import ArrayPtr
    from cumm.tensorview_bind import TensorViewBind, AppleMetalImpl
    pccm.builder.build_pybind([ArrayPtr(), TensorViewBind(), AppleMetalImpl()],
                              PACKAGE_ROOT / "core_cc",
                              namespace_root=PACKAGE_ROOT,
                              load_library=False,
                              std="c++17" if compat.InMacOS else "c++14",
                              verbose=False)


if compat.InWindows:
    try:
        from cumm import core_cc
    except:
        cuda_path = os.getenv("CUDA_PATH", None)
        spec = importlib.util.find_spec("torch")
        cuda_dll_path = None
        if cuda_path is None:
            # try find torch cuda path without import torch
            spec = importlib.util.find_spec("torch")
            if spec is not None and spec.origin is not None:
                torch_path = Path(spec.origin).parent
                cuda_dll_path = str(torch_path / "lib")
        else:
            cuda_dll_path = cuda_path + "/bin"
        if cuda_dll_path is None:
            raise RuntimeError("Can't find cuda libraries, install cuda first.")
        os.add_dll_directory(cuda_dll_path)
        from cumm import core_cc
