#include <c10/core/Allocator.h>


//TODO: track allocated and peak memory for each device to output memory usage statistics

namespace gpuclustersim {

// cost models infer time from shape, the actual data can be dummy 1 byte, as long as nothing dereferences it

struct DummyAllocator : at::Allocator {
  at::DataPtr allocate(size_t nbytes) const override {
    void* ptr = std::malloc(1);
    return at::DataPtr(ptr, ptr, &raw_deallocate, 
      at::Device(at::kPrivateUse1));
  }

private:
  static void raw_deallocate(void* ptr) {
    std::free(ptr);
  }
};


static DummyAllocator g_allocator;

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

}