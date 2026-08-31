import torch
import gpuclustersim._C

_initialized = False

class device:
    def __init__(self, device):
        self.idx = torch.accelerator._get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = gpuclustersim._C._exchangeDevice(self.idx)

    def __exit__(self, type, value, traceback):
        self.idx = gpuclustersim._C._set_device(self.prev_idx)
        return False

def is_available():
    return True

def device_count() -> int:
    return gpuclustersim._C._get_device_count()


def current_device():
    return gpuclustersim._C._get_device()

def set_device(device) -> None:
    if device >= 0:
        gpuclustersim._C._set_device(device)

def is_initialized():
    return _initialized


def _lazy_init():
    global _initialized
    if is_initialized():
        return
    gpuclustersim._C._init()
    _initialized = True

def init():
    _lazy_init()

__all__ = [
    "device",
    "device_count",
    "current_device",
    "set_device",
    "is_available",
    "init",
    "is_initialized"
]