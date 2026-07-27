#include <ATen/Context.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/python_numbers.h>

#ifdef _WIN32
#define ENABLE_EXPORT __declspec(dllexport)
#else
#define ENABLE_EXPORT __attribute__((visibility("default")))
#endif


static PyObject* _initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);
  Py_RETURN_NONE;
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
  return THPGenerator_initDefaultGenerator(

      // at::detail::getPrivateUse1Hooks().getDefaultGenerator(idx));

      at::globalContext().defaultGenerator(
          c10::Device(c10::DeviceType::PrivateUse1, idx))
      
    );

  END_HANDLE_TH_ERRORS
}

//add methods here
static PyMethodDef methods[] = {
    {"_init", _initExtension, METH_NOARGS, nullptr},
    {"_get_default_generator", _getDefaultGenerator, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}
};


extern "C" ENABLE_EXPORT PyObject* initGPUClusterSimModule(void) {
  static struct PyModuleDef gpuclustersim_C_module = {
      PyModuleDef_HEAD_INIT, "gpuclustersim._C", nullptr, -1, methods
  };
  PyObject* mod = PyModule_Create(&gpuclustersim_C_module);
  return mod;
}