from cumm import tensorview as tv 
import numpy as np 

def test_tensor_basic():
    a = np.random.uniform(0, 1, size=[5, 5]).astype(np.float32)
    a_tv = tv.from_numpy(a)
    a_tv2 = a_tv[0, :3]
    print(np.linalg.norm(a[0, :3] - a_tv2.numpy()))

    b = np.random.uniform(0, 1, size=[128, 27, 64]).astype(np.float32)
    b_tv = tv.from_numpy(b)
    b_tv2 = b_tv[:, 3]
    b_tv3 = b_tv.slice_axis(1, 3, 4)
    print(b_tv3.stride)
    print(b[:, 3].strides, b_tv.stride, b_tv2.byte_offset(), b_tv2.shape, b_tv2.stride)
    print(np.linalg.norm(b[:, 3] - b_tv2.numpy()))

if __name__ == "__main__":
    test_tensor_basic()