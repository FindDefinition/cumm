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

from typing import List, Optional
import os 
import pccm
from pccm.utils import project_is_editable, project_is_installed

from .constants import PACKAGE_NAME
from cumm.common import PyBind11, TensorView, TensorViewCPU
from cumm.constants import PACKAGE_ROOT

_TENSORVIEW_BIND_CODE_ANNO_PATH = PACKAGE_ROOT / "tensorview_bind_anno.pyi"
with _TENSORVIEW_BIND_CODE_ANNO_PATH.open("r") as f:
    _TENSORVIEW_BIND_CODE_ANNO = f.read()


class TensorViewBind(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        cumm_cuda_ver = os.getenv("CUMM_CUDA_VERSION", "")
        if cumm_cuda_ver or (project_is_installed(PACKAGE_NAME) and project_is_editable(PACKAGE_NAME)):
            self.add_dependency(TensorView, PyBind11)
        else:
            self.add_dependency(TensorViewCPU, PyBind11)
        self.add_include("tensorview/pybind_utils.h")

    @pccm.pybind.mark
    @pccm.static_function
    def hello(self):
        code = pccm.FunctionCode()
        return code

    @pccm.pybind.mark_bind_raw(raw_bind_anno=_TENSORVIEW_BIND_CODE_ANNO)
    @pccm.static_function
    def bind_tensorview(self):
        code = pccm.FunctionCode()
        code.arg("m", "pybind11::module_")
        # we remove fill in below code because it depends on libcuda.so.
        code.raw("""
  py::class_<tv::Context, std::shared_ptr<tv::Context>>(m, "Context")
    .def(py::init<>())
#ifdef TV_CUDA
    .def("create_cuda_stream", &tv::Context::create_cuda_stream)
    .def("has_cuda_stream", &tv::Context::has_cuda_stream)
#endif
    ;

  py::class_<tv::Tensor, std::shared_ptr<tv::Tensor>>(m, "Tensor")
    .def(py::init([](std::vector<int64_t> shape, int dtype, int device, bool pinned, bool managed) {
        return tv::Tensor(tv::TensorShape(shape), tv::DType(dtype), device, pinned, managed);
    }), py::arg("shape"), py::arg("dtype") = 0, py::arg("device") = -1, py::arg("pinned") = false, py::arg("managed") = false)
    .def(py::init([]() {
        return tv::Tensor();
    }))
    .def("clone", [](const tv::Tensor& ten, bool pinned, bool use_cpu_copy){
      return ten.clone(pinned, use_cpu_copy);
    }, py::arg("pinned") = false, py::arg("use_cpu_copy") = false)
    .def("view", [](const tv::Tensor& ten, std::vector<int64_t> shape){
      return ten.view(tv::TensorShape(shape));
    })
    .def("__getitem__", &tv::Tensor::operator[])
    .def("slice_first_axis", &tv::Tensor::slice_first_axis)
    .def("dim", &tv::Tensor::dim)
    .def("squeeze", py::overload_cast<>(&tv::Tensor::squeeze, py::const_))
    .def("squeeze", py::overload_cast<int>(&tv::Tensor::squeeze, py::const_))
    .def("zero_", &tv::Tensor::zero_, py::arg("ctx") = tv::Context())
    // .def("fill_int_", py::overload_cast<int, tv::Context>(&tv::Tensor::fill_), py::arg("val"), py::arg("ctx") = tv::Context())
    // .def("fill_float_", py::overload_cast<float, tv::Context>(&tv::Tensor::fill_), py::arg("val"), py::arg("ctx") = tv::Context())
    .def("copy_", [](tv::Tensor& t, const tv::Tensor& other, tv::Context ctx) -> void{
      t.copy_(other, ctx);
    }, py::arg("other"), py::arg("ctx") = tv::Context())
    .def("cpu", py::overload_cast<tv::Context>(&tv::Tensor::cpu, py::const_), py::arg("ctx") = tv::Context())
    .def("numpy", [](const tv::Tensor& ten){
      TV_ASSERT_RT_ERR(ten.device() == -1 || (ten.device() == 0 && ten.managed()), "you need to call .cpu() before convert cuda tensor to numpy");
      return tv::tensor2array(ten);
    })
    .def_property_readonly("size", py::overload_cast<>(&tv::Tensor::size, py::const_))
    .def_property_readonly("itemsize", &tv::Tensor::itemsize)
    .def_property_readonly("ndim", &tv::Tensor::ndim)
    .def_property_readonly("device", &tv::Tensor::device)
    .def("pinned", &tv::Tensor::pinned)
    .def("unsqueeze", &tv::Tensor::unsqueeze)
    .def("empty", &tv::Tensor::empty)
    .def("byte_pointer", [](const tv::Tensor& ten){
      return reinterpret_cast<std::uintptr_t>(ten.raw_data());
    })
#ifdef TV_CUDA
    .def("cuda", py::overload_cast<tv::Context>(&tv::Tensor::cuda, py::const_), py::arg("ctx") = tv::Context())
#endif
#if (PYBIND11_VERSION_MAJOR > 2 || (PYBIND11_VERSION_MAJOR == 2 && PYBIND11_VERSION_MINOR >= 6))
    .def("get_memoryview", [](tv::Tensor& ten) {
        TV_ASSERT_RT_ERR(ten.device() == -1 || (ten.device() == 0 && ten.managed()), "you need to call .cpu() before convert cuda tensor to numpy");
        return py::memoryview::from_memory(
            ten.raw_data(),
            ten.raw_size()
        );
    })
#endif
    .def_property_readonly("dtype", [](const tv::Tensor& t){
      return int(t.dtype());
    })
    .def_property_readonly("shape", [](const tv::Tensor& t){
      auto& shape = t.shape();
      return std::vector<int64_t>(shape.begin(), shape.end());
    })
    .def_property_readonly("stride", [](const tv::Tensor& t){
      auto& shape = t.strides();
      return std::vector<int64_t>(shape.begin(), shape.end());
    });
  // from_blob is used for pytorch.
  m.def("from_blob", [](std::uintptr_t ptr_uint, std::vector<int64_t> shape, int dtype, int device){
      return tv::from_blob(reinterpret_cast<void*>(ptr_uint), shape, tv::DType(dtype), device);
  }, py::arg("ptr"), py::arg("shape"), py::arg("dtype"), py::arg("device")); 
  m.def("from_const_blob", [](std::uintptr_t ptr_uint, std::vector<int64_t> shape, int dtype, int device){
      return tv::from_blob(reinterpret_cast<const void*>(ptr_uint), shape, tv::DType(dtype), device);
  }, py::arg("ptr"), py::arg("shape"), py::arg("dtype"), py::arg("device")); 
  m.def("zeros", [](std::vector<int64_t> shape, int dtype, int device, bool pinned, bool managed){
    return tv::zeros(shape, tv::DType(dtype), device, pinned, managed);
  }, py::arg("shape"), py::arg("dtype") = 0, py::arg("device") = -1, py::arg("pinned") = false, py::arg("managed") = false); 
  m.def("empty", [](std::vector<int64_t> shape, int dtype, int device, bool pinned, bool managed){
    return tv::empty(shape, tv::DType(dtype), device, pinned, managed);
  }, py::arg("shape"), py::arg("dtype") = 0, py::arg("device") = -1, py::arg("pinned") = false, py::arg("managed") = false); 
  // m.def("full_int", [](std::vector<int64_t> shape, int val, int dtype, int device, bool pinned, bool managed){
  //   return tv::full(shape, val, tv::DType(dtype), device, pinned, managed);
  // }, py::arg("shape"), py::arg("value"), py::arg("dtype") = 0, py::arg("device") = -1, py::arg("pinned") = false, py::arg("managed") = false); 
  // m.def("full_float", [](std::vector<int64_t> shape, float val, int dtype, int device, bool pinned, bool managed){
  //   return tv::full(shape, val, tv::DType(dtype), device, pinned, managed);
  // }, py::arg("shape"), py::arg("value"), py::arg("dtype") = 0, py::arg("device") = -1, py::arg("pinned") = false, py::arg("managed") = false); 
#ifdef TV_CUDA
  m.def("zeros_managed", [](std::vector<int64_t> shape, int dtype){
    return tv::zeros(shape, tv::DType(dtype), 0, false, true);
  }, py::arg("shape"), py::arg("dtype") = 0); 
  // m.def("full_int_managed", [](std::vector<int64_t> shape, int val, int dtype){
  //   return tv::full(shape, val, tv::DType(dtype), 0, false, true);
  // }, py::arg("shape"), py::arg("value"), py::arg("dtype") = 0); 
  // m.def("full_float_managed", [](std::vector<int64_t> shape, float val, int dtype){
  //   return tv::full(shape, val, tv::DType(dtype), 0, false, true);
  // }, py::arg("shape"), py::arg("value"), py::arg("dtype") = 0); 
#endif
  m.def("from_numpy", [](py::array arr){
    return tv::array2tensor(arr);
  }); 
  m.def("cat_first_axis", &tv::cat_first_axis);
        """)
        return code
