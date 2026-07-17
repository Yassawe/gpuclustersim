#include <ATen/Context.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <pybind11/pybind11.h>

// Include the macro header for visibility (you have it in include/Macros.h)
// If you renamed the macro to GPUCLUSTERSIM_EXPORT, change it accordingly.
#ifdef _WIN32
#define GPUCLUSTERSIM_EXPORT __declspec(dllexport)
#else
#define GPUCLUSTERSIM_EXPORT __attribute__((visibility("default")))
#endif

static PyObject* _initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // This registers a fork handler and lazy-initializes the PrivateUse1 device.
  torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// Minimal method table – we removed all device-management bindings
// because the Python 'sim' module now handles them.
static PyMethodDef methods[] = {
    {"_init", _initExtension, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};

// This is the function that stub.c calls to initialize the module.
extern "C" GPUCLUSTERSIM_EXPORT PyObject* initGPUClusterSimModule(void) {
  static struct PyModuleDef gpuclustersim_C_module = {
      PyModuleDef_HEAD_INIT, "gpuclustersim._C", nullptr, -1, methods
  };
  PyObject* mod = PyModule_Create(&gpuclustersim_C_module);

  // Optional: bind a tiny test function to verify the module loads
  namespace py = pybind11;
  py::module m = py::reinterpret_borrow<py::module>(mod);
  m.def("_test", [](){ return "GPUClusterSim C++ backend loaded!"; });

  return mod;
}