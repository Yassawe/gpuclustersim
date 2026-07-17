#include <Python.h>

#ifdef _WIN32
#define GPUCLUSTERSIM_EXPORT __declspec(dllexport)
#else
#define GPUCLUSTERSIM_EXPORT __attribute__((visibility("default")))
#endif

extern GPUCLUSTERSIM_EXPORT PyObject* initGPUClusterSimModule(void);

#ifdef __cplusplus
extern "C"
#endif

GPUCLUSTERSIM_EXPORT PyObject* PyInit__C(void);

PyMODINIT_FUNC PyInit__C(void) {
  return initGPUClusterSimModule();
}