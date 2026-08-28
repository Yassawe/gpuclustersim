# Dummy utilities for torch.gpuclustersim.*
# These are required for torch._register_device_module to work.

def device_count():
    """Return the number of available simulated devices."""
    return 1 

def current_device():
    """Return the index of the currently selected device."""
    return 0

def set_device(index):
    """Set the current device index (MVP: no-op)."""
    pass

def get_device_properties(device):
    """Return dummy device properties (optional but safe)."""
    return None

def synchronize(device=None):
    """Synchronize the device (MVP: no-op)."""
    pass
