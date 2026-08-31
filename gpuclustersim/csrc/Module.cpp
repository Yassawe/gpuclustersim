#include <ATen/Context.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/python_numbers.h>
#include <utils/Macros.h>
#include <sim_engine.h>
// shamelessly copied and adapted from openreg

static PyObject* _initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _getDeviceCount(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
  return THPUtils_packUInt64(c10::gpuclustersim::gcsDeviceCount());
  END_HANDLE_TH_ERRORS
}

PyObject* _getDevice(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto device = static_cast<int32_t>(  c10::gpuclustersim::gcsCurrentDevice());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* _setDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  auto device = THPUtils_unpackDeviceIndex(arg);
  torch::utils::device_lazy_init(at::kPrivateUse1);
    c10::gpuclustersim::gcsSetDevice(device);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _exchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
  return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto current_device =   c10::gpuclustersim::gcsExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

static PyObject* _getDefaultGenerator(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
    THPUtils_checkLong(arg),
    "_get_default_generator expects an int, but got ",
    THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));

  torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
  return THPGenerator_initDefaultGenerator(at::globalContext().defaultGenerator(c10::Device(c10::DeviceType::PrivateUse1, idx)));
  END_HANDLE_TH_ERRORS
}

//add methods here
static PyMethodDef methods[] = {
  {"_init", _initExtension, METH_NOARGS, nullptr},
  {"_get_device_count", _getDeviceCount, METH_NOARGS, nullptr},
  {nullptr, nullptr, 0, nullptr},
  {"_get_device", _getDevice, METH_NOARGS, nullptr},
  {"_set_device", _setDevice, METH_O, nullptr},
  {"_exchangeDevice", _exchangeDevice, METH_O, nullptr},
  {"_get_default_generator", _getDefaultGenerator, METH_O, nullptr},
  {nullptr, nullptr, 0, nullptr}
};


extern "C" ENABLE_EXPORT PyObject* initGCSModule(void) {
  static struct PyModuleDef gpuclustersim_C_module = {
    PyModuleDef_HEAD_INIT, "gpuclustersim._C", nullptr, -1, methods
  };
  PyObject* mod = PyModule_Create(&gpuclustersim_C_module);
  return mod;
}