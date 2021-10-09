from pathlib import Path

import pccm

from .build_meta import ENABLE_JIT

if ENABLE_JIT:
    from cumm.constants import PACKAGE_ROOT
    from cumm.csrc.arrayref import ArrayPtr
    from cumm.tensorview_bind import TensorViewBind

    pccm.builder.build_pybind([ArrayPtr(), TensorViewBind()],
                              "cumm/core_cc",
                              namespace_root=PACKAGE_ROOT,
                              load_library=False)
