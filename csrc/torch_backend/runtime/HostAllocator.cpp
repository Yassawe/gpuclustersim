#include "HostAllocator.h"
#include <cstdlib>
#include <cstring>

namespace c10::gpuclustersim {

at::DataPtr DummyHostAllocator::allocate(size_t nbytes) {
  void* ptr = malloc(nbytes); 
  return at::DataPtr(ptr, ptr, &DummyHostAllocator::deallocate, at::Device(at::kCPU));
}

void DummyHostAllocator::deallocate(void* ptr){
  free(ptr);
};

void DummyHostAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
  memcpy(dest, src, count);
}

} //namespace c10::gpuclustersim

namespace {

static c10::gpuclustersim::DummyHostAllocator g_host_allocator;
REGISTER_HOST_ALLOCATOR(at::kPrivateUse1, &g_host_allocator);

}
