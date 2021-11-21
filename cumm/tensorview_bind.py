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
from cumm.constants import PACKAGE_ROOT, CUMM_CPU_ONLY_BUILD

_TENSORVIEW_BIND_CODE_ANNO_PATH = PACKAGE_ROOT / "tensorview_bind_anno.pyi"
with _TENSORVIEW_BIND_CODE_ANNO_PATH.open("r") as f:
    _TENSORVIEW_BIND_CODE_ANNO = f.read()


class TensorViewBind(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, PyBind11)
        self.add_include("tensorview/pybind_utils.h")
        self.add_include("tensorview/profile/all.h")

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
  py::class_<tv::CUDAKernelTimer, std::shared_ptr<tv::CUDAKernelTimer>>(m, "CUDAKernelTimer")
    .def(py::init<bool>(), py::arg("enable"))
    .def("push", &tv::CUDAKernelTimer::push, py::arg("name"))
    .def("pop", &tv::CUDAKernelTimer::pop)
    .def("record", &tv::CUDAKernelTimer::record, py::arg("name"), py::arg("stream") = 0)
    .def("insert_pair", &tv::CUDAKernelTimer::insert_pair, py::arg("name"), py::arg("start"), py::arg("stop"))
    .def("has_pair", &tv::CUDAKernelTimer::has_pair, py::arg("name"))
    .def("sync_all_event", &tv::CUDAKernelTimer::sync_all_event)
    .def_property_readonly("enable", &tv::CUDAKernelTimer::enable)
    .def("get_all_pair_duration", &tv::CUDAKernelTimer::get_all_pair_duration);
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
    .def("clone_whole_storage", &tv::Tensor::clone_whole_storage)
    .def("zero_whole_storage_", &tv::Tensor::zero_whole_storage_)
    .def("view", [](const tv::Tensor& ten, std::vector<int64_t> shape){
      return ten.view(tv::TensorShape(shape));
    })
    .def("__getitem__", &tv::Tensor::operator[])
    .def("__getitem__", [](const tv::Tensor& src, const pybind11::slice& key){
      namespace py = pybind11;
      Py_ssize_t start;
      Py_ssize_t stop;
      Py_ssize_t step;
      PySlice_Unpack(key.ptr(), &start, &stop, &step);
      PySliceObject* slice_key = reinterpret_cast<PySliceObject*>(key.ptr());
      bool start_is_none = py::detail::PyNone_Check(slice_key->start);
      bool end_is_none = py::detail::PyNone_Check(slice_key->stop);
      bool step_is_none = py::detail::PyNone_Check(slice_key->step);
      if (step_is_none){
        step = 1;
      }
      return src.slice(0, start, stop, step, start_is_none, end_is_none);
    })
    .def("__getitem__", [](const tv::Tensor& src, const pybind11::tuple& key_list){

      namespace py = pybind11;
      int64_t dim = 0;
      // tv::Tensor prev_dim_result;
      int64_t specified_dims = 0;

      for (auto& key : key_list){
        if (py::detail::PyEllipsis_Check(key.ptr()) || key.is_none()){
          ++specified_dims;
        }
      }
      tv::Tensor result = src;
      for (auto& key : key_list){
        if (py::detail::PyEllipsis_Check(key.ptr())){
          dim += src.ndim() - specified_dims;
        }else if (py::isinstance<py::slice>(key)){
          Py_ssize_t start;
          Py_ssize_t stop;
          Py_ssize_t step;
          PySlice_Unpack(key.ptr(), &start, &stop, &step);
          PySliceObject* slice_key = reinterpret_cast<PySliceObject*>(key.ptr());
          bool start_is_none = py::detail::PyNone_Check(slice_key->start);
          bool end_is_none = py::detail::PyNone_Check(slice_key->stop);
          bool step_is_none = py::detail::PyNone_Check(slice_key->step);
          if (step_is_none){
            step = 1;
          }
          result = result.slice(dim, start, stop, step, start_is_none, end_is_none);
          ++dim;
        }else if (py::isinstance<py::int_>(key)){
          int select_idx = py::cast<int64_t>(key);
          result = result.select(dim, select_idx);
          // result = result.slice(dim, select_idx, select_idx + 1, 1, false, false).squeeze(dim);
        }else if (key.is_none()){
          result = result.unsqueeze(dim);
          dim++;
        }else{
          TV_THROW_INVALID_ARG("tv::Tensor only support .../None/int/slice slicing");
        }
      }
      return result;
    })

    .def("as_strided", [](const tv::Tensor& ten, std::vector<int64_t> shape, std::vector<int64_t> stride, int64_t storage_byte_offset){
      return ten.as_strided(shape, stride, storage_byte_offset);
    }, py::arg("shape"), py::arg("stride"), py::arg("storage_byte_offset") = 0)
    .def("slice_first_axis", &tv::Tensor::slice_first_axis)
    .def("slice_axis", [](const tv::Tensor& ten, int dim, py::object start, py::object stop, py::object step){
      bool start_is_none = start.is_none();
      bool stop_is_none = stop.is_none();
      bool step_is_none = step.is_none();
      int64_t start_val = start_is_none ? 0 : py::cast<int64_t>(start);
      int64_t stop_val = stop_is_none ? 0 : py::cast<int64_t>(stop);
      int64_t step_val = step_is_none ? 1 : py::cast<int64_t>(step); 
      return ten.slice(dim, start_val, stop_val, step_val, start_is_none, stop_is_none);
    }, py::arg("dim"), py::arg("start"), py::arg("stop"), py::arg("step") = py::none())
    .def("select", &tv::Tensor::select)

    // .def("slice_axis", &tv::Tensor::slice)
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
    .def("is_contiguous", &tv::Tensor::is_contiguous)
    .def("byte_offset", &tv::Tensor::byte_offset)

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
  m.def("from_blob", [](std::uintptr_t ptr_uint, std::vector<int64_t> shape, std::vector<int64_t> stride, int dtype, int device){
      return tv::from_blob(reinterpret_cast<void*>(ptr_uint), shape, stride, tv::DType(dtype), device);
  }, py::arg("ptr"), py::arg("shape"), py::arg("stride"), py::arg("dtype"), py::arg("device")); 
  m.def("from_const_blob", [](std::uintptr_t ptr_uint, std::vector<int64_t> shape, std::vector<int64_t> stride, int dtype, int device){
      return tv::from_blob(reinterpret_cast<const void*>(ptr_uint), shape, stride, tv::DType(dtype), device);
  }, py::arg("ptr"), py::arg("shape"), py::arg("stride"), py::arg("dtype"), py::arg("device")); 
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
