#include <Python.h>

// stub.c is compiled by setup.py separately from cmake, so needs to include all it needs explicitly.

#ifdef _WIN32
#define ENABLE_EXPORT __declspec(dllexport)
#else
#define ENABLE_EXPORT __attribute__((visibility("default")))
#endif

extern ENABLE_EXPORT PyObject* initGPUClusterSimModule(void);

#ifdef __cplusplus
extern "C"
#endif

ENABLE_EXPORT PyObject* PyInit__C(void);

PyMODINIT_FUNC PyInit__C(void) {
  return initGPUClusterSimModule();
}