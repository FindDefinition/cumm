import pccm 

from cumm.nvrtc import CummNVRTCModule
from cumm.common import TensorViewNVRTCKernel
from cumm import tensorview as tv 

class SimpleKernel(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewNVRTCKernel)

    @pccm.cuda.cuda_global_function
    def add(self):
        code = pccm.code()
        code.arg("x", "float*")
        code.arg("n", "int")
        code.arg("arr", "tv::array<tv::array<float, 3>, 4>")

        code.raw(f"""
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;  
        if (tid < n) {{
            x[tid] += 1;
        }}
        """)
        return code 


if __name__ == "__main__":
    import numpy as np 
    m = CummNVRTCModule([SimpleKernel()])
    # print(m.program.ptx())
    print(m.name_to_meta)
    a = tv.zeros([2], tv.float32, 0)
    arr = np.zeros((4, 3), dtype=np.float32)
    m.prepare_launch([1, 1, 1], [128, 1, 1])
    m.run_kernel("cumm::nvrtc::dev::add", a, 2, arr)
    print(a.cpu().numpy())

