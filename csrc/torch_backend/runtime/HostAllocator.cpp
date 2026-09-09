#include "HostAllocator.h"
#include <cstdlib>
#include <cstring>

namespace {

static c10::gpuclustersim::GCSHostAllocator g_host_allocator;
REGISTER_HOST_ALLOCATOR(at::kPrivateUse1, &g_host_allocator);

}

namespace c10::gpuclustersim {

at::DataPtr GCSHostAllocator::allocate(size_t nbytes) {
  (void) nbytes;
  void* ptr = malloc(1); 
  return at::DataPtr(ptr, ptr, &GCSHostAllocator::deallocate, at::Device(at::kCPU));
}

void GCSHostAllocator::deallocate(void* ptr){
  free(ptr);
};

void GCSHostAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
  (void) count;
  memcpy(dest, src, 1);
}

} //namespace c10::gpuclustersim

