import torch
import gpuclustersim._C

_initialized = False

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

def is_available():
  return True


def _device_index(device=None):
  if device is None:
    return gpuclustersim._C._get_device()
  return torch.accelerator._get_device_index(device, optional=True)


class device:
  def __init__(self, device):
    self.idx = torch.accelerator._get_device_index(device, optional=True)
    self.prev_idx = -1

  def __enter__(self):
    self.prev_idx = gpuclustersim._C._exchange_device(self.idx)

  def __exit__(self, type, value, traceback):
    self.idx = gpuclustersim._C._set_device(self.prev_idx)
    return False


class stream:
  def __init__(self, stream):
    self.stream = stream
    self.prev_stream = None

  def __enter__(self):
    if self.stream is not None:
      self.prev_stream = gpuclustersim._C._get_stream(self.stream.device_index)
      gpuclustersim._C._exchange_stream(self.stream.device_index, self.stream.stream_id)

  def __exit__(self, type, value, traceback):
    if self.stream is not None and self.prev_stream is not None:
      gpuclustersim._C._exchange_stream(self.stream.device_index, self.prev_stream)
    return False


def current_stream(device=None):
  device = _device_index(device)
  return torch.Stream(
    stream_id=gpuclustersim._C._get_stream(device),
    device_index=device,
    device_type=gpuclustersim._C._device_type(),
  )


def default_stream(device=None):
  device = _device_index(device)
  return torch.Stream(
    stream_id=gpuclustersim._C._get_default_stream(device),
    device_index=device,
    device_type=gpuclustersim._C._device_type(),
  )


def Stream(device=None, priority=0):
  if device is None:
    device = torch.device("gpuclustersim", gpuclustersim._C._get_device())
  return torch.Stream(device=device, priority=priority)


def Event(*args, **kwargs):
  return torch.Event(*args, **kwargs)


def synchronize(device=None):
  pass


def device_count():
  return gpuclustersim._C._get_device_count()


def current_device():
  return gpuclustersim._C._get_device()


def set_device(device):
  if device >= 0:
    gpuclustersim._C._set_device(device)


__all__ = [
  "is_available",
  "is_initialized",
  "init",
  "device",
  "stream",
  "current_stream",
  "default_stream",
  "Stream",
  "Event",
  "synchronize",
  "memory_summary",
  "device_count",
  "current_device",
  "set_device"
]