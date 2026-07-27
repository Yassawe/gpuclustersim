#include <c10/core/Allocator.h>


//TODO: track allocated and peak memory for each device to output memory usage statistics

namespace c10::gpuclustersim {

// cost models infer time from shape, the actual data can be dummy 1 byte, as long as nothing dereferences it. 
// extreme caution so that nothing dereferences it, otherwise segfault and crash

// must be per device

struct DummyAllocator : at::Allocator {
  at::DataPtr allocate(size_t nbytes) override {
    void* ptr = std::malloc(1); //change here
    return at::DataPtr(ptr, ptr, &raw_delete, 
      at::Device(at::kPrivateUse1, 0)); // HERE ALLOCATED TO A DEVICE
  }

  static void raw_delete(void* ptr) {
    free(ptr);
  }
  
  void copy_data(void* dest, const void* src, std::size_t count) const override {
    memcpy(dest, src, count);
  }

};

static DummyAllocator g_allocator;

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

}