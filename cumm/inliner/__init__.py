# Copyright 2022 Yan Yan
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

import pccm
from pccm.builder.inliner import InlineBuilder
from cumm.nvrtc import CummNVRTCModule
from pathlib import Path
from cumm.common import TensorViewKernel


class NVRTCInlineBuilder(InlineBuilder):
    def builder(self, pccm_cls: pccm.Class, mod_root: Path, prev_mod_name: str,
                unique_key, timeout: float):
        mod = CummNVRTCModule([pccm_cls])
        return mod

    def runner(self, func: CummNVRTCModule, *args):
        return func.run_kernel(func.kernels[0], *args)

    def inline_cuda(self, ):
        pass

if __name__ == "__main__":
    INLINE = NVRTCInlineBuilder([TensorViewKernel])
    from cumm import tensorview as tv 
    a = tv.zeros([2], tv.float32, 0)

    # INLINE.inline("hahaha", f"""
    # for ()
    # """)
    