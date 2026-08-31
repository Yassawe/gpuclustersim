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

def device_count():
  return gpuclustersim._C._get_device_count()

def current_device():
  return gpuclustersim._C._get_device()

def set_device(device):
  if device >= 0:
    gpuclustersim._C._set_device(device)

def current_stream():
  return gpuclustersim._C._get_current_stream()

def default_stream():
  return gpuclustersim._C._get_default_stream()

def new_stream():
  return gpuclustersim._C._new_stream()

def set_stream(stream):
  return gpuclustersim._C._exchange_stream(stream)

class stream:
  def __init__(self, stream_id):
    self.stream_id = stream_id
    self.prev = None

  def __enter__(self):
    self.prev = gpuclustersim._C._exchange_stream(self.stream_id)

  def __exit__(self, type, value, traceback):
    gpuclustersim._C._exchange_stream(self.prev)
    return False

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
  "is_initialized",
  "current_stream",
  "default_stream",
  "new_stream",
  "set_stream",
  "stream"
]