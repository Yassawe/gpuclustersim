import os
import socket as _socket
import sys

def _valid_master_addr(addr):
  if not addr:
    return False
  if ":" in addr or addr.endswith(".ip6.arpa"):
    return False
  try:
    _socket.inet_aton(addr)
    return True
  except OSError:
    return False

if not _valid_master_addr(os.environ.get("MASTER_ADDR")):
  os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ.setdefault("MASTER_PORT", "29500")

if sys.platform == "win32":
  from ._utils import _load_dll_libraries
  _load_dll_libraries()
  del _load_dll_libraries

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d
import gpuclustersim._C as _C
import gpuclustersim.module
import json

# dummy allocator allocates fake tensors (torch_backend/runtime/DeviceAllocator.cpp), this verify_params reads the config tensor, which causes OOB and segfault whenever DDP is used. For some reason DDP routes its internal config files through the same paths as the normal tensors, which was perhaps code reuse by devs? idk. monkeypatching with noop solves this. downside: no other distributed backend can work within the same runtime, which is ok since it is rare to switch c10d backends mid run

def _noop_verify(*args, **kwargs):
    pass

dist._verify_params_across_processes = _noop_verify
distributed_c10d._verify_params_across_processes = _noop_verify

# backend registration
torch.utils.rename_privateuse1_backend("gpuclustersim")
torch._register_device_module("gpuclustersim", gpuclustersim.module)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

_C._init()

# distributed backend registration
def _simccl_creator(dist_backend_opts, backend_options):
  return _C._create_simccl_backend(
    dist_backend_opts.group_rank,
    dist_backend_opts.group_size,
    list(dist_backend_opts.global_ranks_in_group),
    dist_backend_opts.store,
  )

dist.Backend.register_backend("simccl", _simccl_creator, extended_api=True, devices=["gpuclustersim"])

# user facing functions
def init_device_group(path, n=None):
  with open(path, "r") as f:
    spec = json.load(f)
  if n is None:
    n = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
  _C._init_device_group(spec, n)
  _C._set_device(int(os.environ.get("LOCAL_RANK", 0)))

def save_timeline(path):
  raw = _C._get_timeline()
  rank = dist.get_rank() if dist.is_initialized() else 0

  trace_events = []
  for streams in raw:
    for stream_id, stream in enumerate(streams):
      for op in stream["ops"]:
        trace_events.append(
          {
            "name": op["name"],
            "ph": "X",
            "ts": op["start_time"],
            "dur": op["end_time"] - op["start_time"],
            "pid": rank,
            "tid": stream_id,
          }
        )

  output = {
    "displayTimeUnit": "ns",
    "traceEvents": trace_events
  }

  if dist.is_initialized():
    root, ext = os.path.splitext(path)
    path = f"{root}.rank{rank}{ext}"

  with open(path, "w") as f:
    json.dump(output, f, indent=2)

def reset_timeline():
  _C._reset_timeline()

__all__ = [
  "init_device_group",
  "save_timeline",
  "reset_timeline"
]