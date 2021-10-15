# Copyright 2021 Yan Yan
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

from typing import List, Tuple

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.common import GemmBasic, TensorView, TensorViewKernel
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import constants
from cumm.gemm.core import MetaArray, array_type, metaseq, seq


class MmaSync(pccm.ParameterizedClass):
    """
    Volta: [8, 8, 4]
    Turing: tnt, [16, 8, 8] f16f16[f16|f32]
    [8, 8, 16] [u8|s8][u8|s8]s32
    [8, 8, 32] [u4|s4][u4|s4]s32

    Requires sm_70 or higher.

    Note:mma.sync.m8n8k4 is optimized for target architecture sm_70 and may have substantially reduced performance on other target architectures.
    Shapes .m16n8k8, .m16n8k16, .m8n8k128, .m8n8k16 and .m8n8k32 require sm_75 or higher.

    Alternate floating point types .bf16 and .tf32 on shape .m16n8k8 require sm_80 or higher.

    Shapes .m16n8k4, .m16n8k32, .m16n8k64, .m16n8k128 and .m16n8k256 require sm_80 or higher.

    Shapes .m8n8k4 with .f64 floating point type require sm_80 or higher.

    .and operation in single-bit mma requires sm_80 or higher.

    m16n8k16 tf32 only available in sparse kernel
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-shape
    """
    def __init__(self,
                 shape: MetaArray[int],
                 num_threads: int,
                 dtype_a: dtypes.DType,
                 dtype_b: dtypes.DType,
                 dtype_c: dtypes.DType,
                 trans_a: bool,
                 trans_b: bool,
                 trans_c: bool,
                 satfinite: bool = False):
        super().__init__()
        self.add_dependency(TensorView, TensorViewKernel, GemmBasic)
        self.shape = shape
        self.num_threads = num_threads
        self.dtype_a = dtype_a
        self.dtype_b = dtype_b
        self.dtype_c = dtype_c
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.satfinite = satfinite
        self.mn = shape[0] * shape[1]
        self.km = shape[2] * shape[0]
        self.kn = shape[2] * shape[1]
        if num_threads == 8:
            assert shape[0] == 8 and shape[1] == 8 and shape[2] == 4
            assert self.trans_c is False

        self.fragment_c_count = self.mn // self.num_threads
        self.fragment_a_count = self.km // self.num_threads
        self.fragment_b_count = self.kn // self.num_threads
        self.fragment_a_t = array_type(str(dtype_a), self.fragment_a_count)
        self.fragment_b_t = array_type(str(dtype_b), self.fragment_b_count)
        self.fragment_c_t = array_type(str(dtype_c), self.fragment_c_count)

    def python_ctor(self):
        return self

    @pccm.cuda.member_function(name="operator()",
                               device=True,
                               forceinline=True)
    def call_operator(self):
        mma_stmt = f"mma.sync.aligned.m{self.shape[0]}n{self.shape[1]}k{self.shape[2]}"
        layout_a = "col" if self.trans_a else "row"
        layout_b = "col" if self.trans_b else "row"

        mma_stmt += f".{layout_a}.{layout_b}"
        if self.satfinite:
            mma_stmt += ".satfinite"
        mma_stmt += f".{self.dtype_c.shortcut()}.{self.dtype_a.shortcut()}.{self.dtype_b.shortcut()}.{self.dtype_c.shortcut()}"

        # compute number of inputs
        num_32_registers_out = self.dtype_c.bitsize(
        ) * self.mn // self.num_threads // 32
        num_32_registers_a = self.dtype_a.bitsize(
        ) * self.km // self.num_threads // 32
        num_32_registers_b = self.dtype_b.bitsize(
        ) * self.kn // self.num_threads // 32
        out_label = "f" if self.dtype_c == dtypes.float32 else "r"
        cnt = 0
        # register names
        out_register_names = ",".join(
            [f"%{i + cnt}" for i in range(num_32_registers_out)])
        cnt += num_32_registers_out
        a_register_names = ",".join(
            [f"%{i + cnt}" for i in range(num_32_registers_a)])
        cnt += num_32_registers_a
        b_register_names = ",".join(
            [f"%{i + cnt}" for i in range(num_32_registers_b)])
        cnt += num_32_registers_b
        c_register_names = ",".join(
            [f"%{i + cnt}" for i in range(num_32_registers_out)])
        cnt += num_32_registers_out
        code = pccm.FunctionCode()

        # output operands
        out_operands = ", ".join(f"\"={out_label}\"(D[{i}])"
                                 for i in range(num_32_registers_out))
        # input operands
        if self.shape[0] == 8 and self.shape[1] == 8 and self.shape[2] == 4:
            code.raw("""
            #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
            """)
        else:
            code.raw("""
            #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
            """)
        if num_32_registers_a == 1:
            a_operands = "\"r\"(A)"
            code.raw(
                "unsigned const & A = reinterpret_cast<unsigned const &>(a);")
        else:
            a_operands = ", ".join(f"\"r\"(A[{i}])"
                                   for i in range(num_32_registers_a))
            code.raw(
                "unsigned const *A = reinterpret_cast<unsigned const *>(&a);")
        if num_32_registers_b == 1:
            b_operands = "\"r\"(B)"
            code.raw(
                "unsigned const & B = reinterpret_cast<unsigned const &>(b);")
        else:
            b_operands = ", ".join(f"\"r\"(B[{i}])"
                                   for i in range(num_32_registers_b))
            code.raw(
                "unsigned const *B = reinterpret_cast<unsigned const *>(&b);")

        if self.dtype_c == dtypes.float32:
            code.raw("float const *C = reinterpret_cast<float const *>(&c);")
            code.raw("float *D = reinterpret_cast<float *>(&d);")
        else:
            code.raw(
                "unsigned const *C = reinterpret_cast<unsigned const *>(&c);")
            code.raw("unsigned *D = reinterpret_cast<unsigned *>(&d);")
        c_operands = ", ".join(f"\"{out_label}\"(C[{i}])"
                               for i in range(num_32_registers_out))
        code.raw(f"""
        asm volatile("{mma_stmt} {{{out_register_names}}}, {{{a_register_names}}}, {{{b_register_names}}}, {{{c_register_names}}};\\n"
            : {out_operands}
            : {a_operands}, {b_operands}, {c_operands});
        """)
        code.raw("#endif")
        code.arg("d", f"{self.fragment_c_t}&")
        code.arg("a", f"{self.fragment_a_t} const &")
        code.arg("b", f"{self.fragment_b_t} const &")
        code.arg("c", f"{self.fragment_c_t} const &")
        return code

    async def __call__(self, d: ArrayPtr, a: ArrayPtr, b: ArrayPtr,
                       c: ArrayPtr):

        lane_id = cudasim.get_lane_id()
        warp_id = cudasim.get_warp_id()
        resource = cudasim.get_warp_resource()
        warp_data = await resource.gather(
            lane_id, (d, a, b, c),
            0)  # type: List[Tuple[ArrayPtr, ArrayPtr, ArrayPtr,ArrayPtr]]
        dabc = (self.dtype_a, self.dtype_b, self.dtype_c)
        dab = (self.dtype_a, self.dtype_b)

        if lane_id == 0:
            Ds = [x[0].data.numpy_view() for x in warp_data]
            As = [x[1].data.numpy_view() for x in warp_data]
            Bs = [x[2].data.numpy_view() for x in warp_data]
            Cs = [x[3].data.numpy_view() for x in warp_data]
            if self.shape == (8, 8, 4):
                w_off = [0, 4, 8, 12]
                # Ds_ten = np.stack(Ds)
                # Ds_ten[:] = 0
                # As_ten = np.stack(As)
                # Bs_ten = np.stack(Bs)
                # Cs_ten = np.stack(Cs)
                # MmaSyncTester.mma_test(tv.from_numpy(As_ten), tv.from_numpy(Bs_ten),
                #     tv.from_numpy(Cs_ten), tv.from_numpy(Ds_ten))
                # for i in range(len(Ds)):
                #     Ds[i][:] = Ds_ten[i]
                # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma
                if not self.trans_a:
                    A_qp0 = np.stack(As[w_off[0]:w_off[0] + 4] +
                                     As[w_off[0] + 16:w_off[0] + 16 + 4])
                    A_qp1 = np.stack(As[w_off[1]:w_off[1] + 4] +
                                     As[w_off[1] + 16:w_off[1] + 16 + 4])
                    A_qp2 = np.stack(As[w_off[2]:w_off[2] + 4] +
                                     As[w_off[2] + 16:w_off[2] + 16 + 4])
                    A_qp3 = np.stack(As[w_off[3]:w_off[3] + 4] +
                                     As[w_off[3] + 16:w_off[3] + 16 + 4])
                else:
                    A_qp0 = np.concatenate([
                        np.stack(As[w_off[0]:w_off[0] + 4]).T,
                        np.stack(As[w_off[0] + 16:w_off[0] + 16 + 4]).T
                    ])
                    A_qp1 = np.concatenate([
                        np.stack(As[w_off[1]:w_off[1] + 4]).T,
                        np.stack(As[w_off[1] + 16:w_off[1] + 16 + 4]).T
                    ])
                    A_qp2 = np.concatenate([
                        np.stack(As[w_off[2]:w_off[2] + 4]).T,
                        np.stack(As[w_off[2] + 16:w_off[2] + 16 + 4]).T
                    ])
                    A_qp3 = np.concatenate([
                        np.stack(As[w_off[3]:w_off[3] + 4]).T,
                        np.stack(As[w_off[3] + 16:w_off[3] + 16 + 4]).T
                    ])
                if not self.trans_b:
                    B_qp0 = np.concatenate([
                        np.stack(Bs[w_off[0]:w_off[0] + 4]),
                        np.stack(Bs[w_off[0] + 16:w_off[0] + 16 + 4])
                    ],
                                           axis=1)
                    B_qp1 = np.concatenate([
                        np.stack(Bs[w_off[1]:w_off[1] + 4]),
                        np.stack(Bs[w_off[1] + 16:w_off[1] + 16 + 4])
                    ],
                                           axis=1)
                    B_qp2 = np.concatenate([
                        np.stack(Bs[w_off[2]:w_off[2] + 4]),
                        np.stack(Bs[w_off[2] + 16:w_off[2] + 16 + 4])
                    ],
                                           axis=1)
                    B_qp3 = np.concatenate([
                        np.stack(Bs[w_off[3]:w_off[3] + 4]),
                        np.stack(Bs[w_off[3] + 16:w_off[3] + 16 + 4])
                    ],
                                           axis=1)
                else:
                    B_qp0 = np.stack(Bs[w_off[0]:w_off[0] + 4] +
                                     Bs[w_off[0] + 16:w_off[0] + 16 + 4]).T
                    B_qp1 = np.stack(Bs[w_off[1]:w_off[1] + 4] +
                                     Bs[w_off[1] + 16:w_off[1] + 16 + 4]).T
                    B_qp2 = np.stack(Bs[w_off[2]:w_off[2] + 4] +
                                     Bs[w_off[2] + 16:w_off[2] + 16 + 4]).T
                    B_qp3 = np.stack(Bs[w_off[3]:w_off[3] + 4] +
                                     Bs[w_off[3] + 16:w_off[3] + 16 + 4]).T
                # print(A_qp0.shape, B_qp0.shape)

                # assume A always 8x4, B always 4x8
                D_qp0_res = (A_qp0 @ B_qp0)
                D_qp1_res = (A_qp1 @ B_qp1)
                D_qp2_res = (A_qp2 @ B_qp2)
                D_qp3_res = (A_qp3 @ B_qp3)
                C_res = [np.zeros_like(D_qp0_res) for _ in range(4)]
                if dabc == (dtypes.float16, dtypes.float16, dtypes.float32):
                    for qp_idx in range(4):
                        C_qp_res = C_res[qp_idx]
                        for lane_id_mma in range(w_off[qp_idx],
                                                 w_off[qp_idx] + 4):
                            for i in range(8):
                                row = (lane_id_mma & 0b1) + (i & 0b10)
                                col = (i & 0b100) + (lane_id_mma
                                                     & 0b10) + (i & 0b1)
                                C_qp_res[row, col] = Cs[lane_id_mma][i]
                        for lane_id_mma in range(w_off[qp_idx] + 16,
                                                 w_off[qp_idx] + 16 + 4):
                            for i in range(8):
                                row = (lane_id_mma & 0b1) + (i & 0b10) + 4
                                col = (i & 0b100) + (lane_id_mma
                                                     & 0b10) + (i & 0b1)
                                C_qp_res[row, col] = Cs[lane_id_mma][i]
                    C_qp0, C_qp1, C_qp2, C_qp3 = C_res
                else:
                    C_qp0 = np.stack(Cs[w_off[0]:w_off[0] + 4] +
                                     Cs[w_off[0] + 16:w_off[0] + 16 + 4])
                    C_qp1 = np.stack(Cs[w_off[1]:w_off[1] + 4] +
                                     Cs[w_off[1] + 16:w_off[1] + 16 + 4])
                    C_qp2 = np.stack(Cs[w_off[2]:w_off[2] + 4] +
                                     Cs[w_off[2] + 16:w_off[2] + 16 + 4])
                    C_qp3 = np.stack(Cs[w_off[3]:w_off[3] + 4] +
                                     Cs[w_off[3] + 16:w_off[3] + 16 + 4])
                D_qp0_res += C_qp0
                D_qp1_res += C_qp1
                D_qp2_res += C_qp2
                D_qp3_res += C_qp3

                D_res = [D_qp0_res, D_qp1_res, D_qp2_res, D_qp3_res]
                D_res = [ddd.astype(self.dtype_c.npdtype()) for ddd in D_res]
                if dabc == (dtypes.float16, dtypes.float16, dtypes.float32):
                    for qp_idx in range(4):
                        D_qp_res = D_res[qp_idx]
                        for lane_id_mma in range(w_off[qp_idx],
                                                 w_off[qp_idx] + 4):
                            for i in range(8):
                                row = (lane_id_mma & 0b1) + (i & 0b10)
                                col = (i & 0b100) + (lane_id_mma
                                                     & 0b10) + (i & 0b1)
                                Ds[lane_id_mma][i] = D_qp_res[row, col]
                        for lane_id_mma in range(w_off[qp_idx] + 16,
                                                 w_off[qp_idx] + 16 + 4):
                            for i in range(8):
                                row = (lane_id_mma & 0b1) + (i & 0b10) + 4
                                col = (i & 0b100) + (lane_id_mma
                                                     & 0b10) + (i & 0b1)
                                Ds[lane_id_mma][i] = D_qp_res[row, col]

                else:
                    for qp_idx in range(4):
                        D_qp_res = D_res[qp_idx]
                        for lane_id_mma in range(w_off[qp_idx],
                                                 w_off[qp_idx] + 4):
                            for i in range(8):
                                row = lane_id_mma % 4
                                col = i
                                Ds[lane_id_mma][i] = D_qp_res[row, col]
                        for lane_id_mma in range(w_off[qp_idx] + 16,
                                                 w_off[qp_idx] + 16 + 4):
                            for i in range(8):
                                row = lane_id_mma % 4 + 4
                                col = i
                                Ds[lane_id_mma][i] = D_qp_res[row, col]
            elif self.shape == (16, 8, 8):
                if dab == (dtypes.float16, dtypes.float16):
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
                    mma_A = np.zeros((16, 8), dtype=np.float16)
                    mma_B = np.zeros((8, 8), dtype=np.float16)
                    mma_C = np.zeros((16, 8), dtype=self.dtype_c.npdtype())

                    for aid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if aid < 2:
                                row = group_id
                            else:
                                row = group_id + 8
                            col = tid_in_group * 2 + (aid & 0x1)
                            mma_A[row, col] = As[lane_id_mma][aid]
                            mma_C[row, col] = Cs[lane_id_mma][aid]

                    for bid in range(2):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            row = tid_in_group * 2 + bid
                            col = group_id
                            mma_B[row, col] = Bs[lane_id_mma][bid]

                    mma_D = mma_A @ mma_B + mma_C
                    for aid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if aid < 2:
                                row = group_id
                            else:
                                row = group_id + 8
                            col = tid_in_group * 2 + (aid & 0x1)
                            Ds[lane_id_mma][aid] = mma_D[row, col]
                elif dab == (dtypes.tf32, dtypes.tf32):
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
                    mma_A = np.zeros((16, 8), dtype=np.float32)
                    mma_B = np.zeros((8, 8), dtype=np.float32)
                    mma_C = np.zeros((16, 8), dtype=np.float32)

                    for aid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if aid < 2:
                                row = group_id
                                col = tid_in_group
                            else:
                                row = group_id + 8
                                col = tid_in_group + 4
                            mma_A[row, col] = As[lane_id_mma][aid]

                    for bid in range(2):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if bid == 0:
                                row = tid_in_group
                            else:
                                row = tid_in_group + 4
                            col = group_id
                            mma_B[row, col] = Bs[lane_id_mma][bid]

                    for cid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if cid < 2:
                                row = group_id
                            else:
                                row = group_id + 8
                            col = tid_in_group * 2 + (cid & 0x1)
                            mma_C[row, col] = Cs[lane_id_mma][cid]

                    mma_D = mma_A @ mma_B + mma_C

                    for cid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if cid < 2:
                                row = group_id
                            else:
                                row = group_id + 8
                            col = tid_in_group * 2 + (cid & 0x1)
                            Ds[lane_id_mma][cid] = mma_D[row, col]

                else:
                    raise NotImplementedError
            elif self.shape == (16, 8, 16):
                if dab == (dtypes.float16, dtypes.float16):
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-f16
                    mma_A = np.zeros((16, 16), dtype=np.float16)
                    mma_B = np.zeros((16, 8), dtype=np.float16)
                    mma_C = np.zeros((16, 8), dtype=self.dtype_c.npdtype())

                    for aid in range(8):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if aid < 2 or (4 <= aid and aid < 6):
                                row = group_id
                            else:
                                row = group_id + 8
                            if aid < 4:
                                col = (tid_in_group * 2) + (aid & 0x1)

                            else:
                                col = (tid_in_group * 2) + (aid & 0x1) + 8
                            mma_A[row, col] = As[lane_id_mma][aid]

                    for bid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if bid < 2:
                                row = (tid_in_group * 2) + (bid & 0x1)
                            else:
                                row = (tid_in_group * 2) + (bid & 0x1) + 8
                            col = group_id
                            mma_B[row, col] = Bs[lane_id_mma][bid]

                    for cid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if cid < 2:
                                row = group_id
                            else:
                                row = group_id + 8
                            col = tid_in_group * 2 + (cid & 0x1)
                            mma_C[row, col] = Cs[lane_id_mma][cid]

                    mma_D = mma_A @ mma_B + mma_C

                    for cid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if cid < 2:
                                row = group_id
                            else:
                                row = group_id + 8
                            col = tid_in_group * 2 + (cid & 0x1)
                            Ds[lane_id_mma][cid] = mma_D[row, col]
                elif dab == (dtypes.int8,
                             dtypes.int8) or dab == (dtypes.uint8,
                                                     dtypes.uint8):
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-i8
                    mma_A = np.zeros((16, 16), dtype=self.dtype_a.npdtype())
                    mma_B = np.zeros((16, 8), dtype=self.dtype_b.npdtype())
                    mma_C = np.zeros((16, 8), dtype=self.dtype_c.npdtype())

                    for aid in range(8):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if aid < 4:
                                row = group_id
                            else:
                                row = group_id + 8
                            col = (tid_in_group * 4) + (aid & 0x3)
                            mma_A[row, col] = As[lane_id_mma][aid]

                    for bid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            row = (tid_in_group * 4) + bid
                            col = group_id
                            mma_B[row, col] = Bs[lane_id_mma][bid]

                    for cid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if cid < 2:
                                row = group_id
                            else:
                                row = group_id + 8
                            col = tid_in_group * 2 + (cid & 0x1)
                            mma_C[row, col] = Cs[lane_id_mma][cid]

                    mma_D = mma_A @ mma_B + mma_C

                    for cid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if cid < 2:
                                row = group_id
                            else:
                                row = group_id + 8
                            col = tid_in_group * 2 + (cid & 0x1)
                            Ds[lane_id_mma][cid] = mma_D[row, col]
                else:
                    raise NotImplementedError
            elif self.shape == (16, 8, 32):
                if dab == (dtypes.int8, dtypes.int8) or dab == (dtypes.uint8,
                                                                dtypes.uint8):
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16832
                    mma_A = np.zeros((16, 32), dtype=self.dtype_a.npdtype())
                    mma_B = np.zeros((32, 8), dtype=self.dtype_b.npdtype())
                    mma_C = np.zeros((16, 8), dtype=self.dtype_c.npdtype())
                    for aid in range(16):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if aid < 4 or 8 <= aid < 12:
                                row = group_id
                            else:
                                row = group_id + 8
                            if aid < 8:
                                col = (tid_in_group * 4) + (aid & 0x3)
                            else:
                                col = (tid_in_group * 4) + (aid & 0x3) + 16
                            mma_A[row, col] = As[lane_id_mma][aid]

                    for bid in range(8):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if bid < 4:
                                row = (tid_in_group * 4) + (bid & 0x3)
                            else:
                                row = (tid_in_group * 4) + (bid & 0x3) + 16
                            col = group_id
                            mma_B[row, col] = Bs[lane_id_mma][bid]

                    for cid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if cid < 2:
                                row = group_id
                            else:
                                row = group_id + 8
                            col = tid_in_group * 2 + (cid & 0x1)
                            mma_C[row, col] = Cs[lane_id_mma][cid]

                    mma_D: np.ndarray = mma_A @ mma_B + mma_C

                    for cid in range(4):
                        for lane_id_mma in range(32):
                            group_id = lane_id_mma >> 2
                            tid_in_group = lane_id_mma % 4
                            if cid < 2:
                                row = group_id
                            else:
                                row = group_id + 8
                            col = tid_in_group * 2 + (cid & 0x1)
                            Ds[lane_id_mma][cid] = mma_D[row, col]
                else:
                    raise NotImplementedError

            elif self.shape == (8, 8, 16):
                # s8/u8 -> s32, RCR
                # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-8816
                mma_A = np.zeros((8, 16), dtype=self.dtype_a.npdtype())
                mma_B = np.zeros((16, 8), dtype=self.dtype_b.npdtype())
                mma_C = np.zeros((8, 8), dtype=self.dtype_c.npdtype())
                for aid in range(4):
                    for lane_id_mma in range(32):
                        group_id = lane_id_mma >> 2
                        tid_in_group = lane_id_mma % 4
                        row = group_id
                        col = (tid_in_group * 4) + aid
                        mma_A[row, col] = As[lane_id_mma][aid]
                for bid in range(4):
                    for lane_id_mma in range(32):
                        group_id = lane_id_mma >> 2
                        tid_in_group = lane_id_mma % 4
                        col = group_id
                        row = (tid_in_group * 4) + bid
                        mma_B[row, col] = Bs[lane_id_mma][bid]
                for cid in range(2):
                    for lane_id_mma in range(32):
                        group_id = lane_id_mma >> 2
                        tid_in_group = lane_id_mma % 4
                        row = group_id
                        col = (tid_in_group * 2) + cid
                        mma_C[row, col] = Cs[lane_id_mma][cid]
                npdtype_c = self.dtype_c.npdtype()
                mma_D = (mma_A.astype(npdtype_c) @ mma_B.astype(npdtype_c) +
                         mma_C)
                for cid in range(2):
                    for lane_id_mma in range(32):
                        group_id = lane_id_mma >> 2
                        tid_in_group = lane_id_mma % 4
                        row = group_id
                        col = (tid_in_group * 2) + cid
                        Ds[lane_id_mma][cid] = mma_D[row, col]
            else:
                raise NotImplementedError
        # wait for compute finish
        await resource.wait()
        return
