import sys

import torch


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries

import gpuclustersim._C  # type: ignore[misc]
import gpuclustersim.sim

_C.init()


torch.utils.rename_privateuse1_backend("gpuclustersim")
torch._register_device_module("gpuclustersim", gpuclustersim.sim)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)



