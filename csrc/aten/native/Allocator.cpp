#include <c10/core/Allocator.h>
#include <ATen/Context.h>

// This static initializer runs when the library is loaded.
static int register_allocator = []() {
    // Register the CPU allocator for the PrivateUse1 device.
    // This lets PyTorch allocate "host" memory for tensors on our "device".
    at::SetAllocator(at::kPrivateUse1, at::getCPUAllocator());
    return 0;
}();