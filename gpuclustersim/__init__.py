import sys
import torch
import os
import json

if sys.platform == "win32":
  from ._utils import _load_dll_libraries
  _load_dll_libraries()
  del _load_dll_libraries

import gpuclustersim._C as _C
import gpuclustersim.module

torch.utils.rename_privateuse1_backend("gpuclustersim")
torch._register_device_module("gpuclustersim", gpuclustersim.module)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

def init_devices(path, n):
  with open(path, "r") as f:
    spec = json.load(f)
  _C._init_devices(spec, n)

def dump_timeline(path):
  raw = _C._get_timeline()

  trace_events = []
  for device_id, streams in enumerate(raw):
    for stream_id, stream in enumerate(streams):
      for op in stream["ops"]:
        trace_events.append(
          {
            "name": op["name"],
            "ph": "X",
            "ts": int(op["start_time"]),
            "dur": int(op["end_time"] - op["start_time"]),
            "pid": device_id,
            "tid": stream_id,
          }
        )

  output = {"traceEvents": trace_events}

  with open(path, "w") as f:
    json.dump(output, f, indent=2)

