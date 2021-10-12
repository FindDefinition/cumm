from pathlib import Path

import pccm
from pccm.utils import project_is_editable, project_is_installed

from .constants import PACKAGE_NAME

if project_is_installed(PACKAGE_NAME) and project_is_editable(PACKAGE_NAME):
    from cumm.constants import PACKAGE_ROOT
    from cumm.csrc.arrayref import ArrayPtr
    from cumm.tensorview_bind import TensorViewBind

    pccm.builder.build_pybind([ArrayPtr(), TensorViewBind()],
                              "cumm/core_cc",
                              namespace_root=PACKAGE_ROOT,
                              load_library=False)
