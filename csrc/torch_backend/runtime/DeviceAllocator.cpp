#include "DeviceAllocator.h"
#include "DeviceFunctions.h"
#include <cstdlib>
#include <cstring>

namespace {
  static c10::gpuclustersim::DummyAllocator g_allocator;
  REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);
}


namespace c10::gpuclustersim {

DummyAllocator::DummyAllocator(): per_device_stats(device_count()) {}

at::DataPtr DummyAllocator::allocate(size_t nbytes) {
  std::lock_guard<std::mutex> lock(mutex_); 

  void* ptr = malloc(1); 
  DeviceIndex device = current_device();  
  auto& stats = per_device_stats[device];
  stats.current_allocated+=nbytes;
  if (stats.current_allocated>stats.peak_allocated) stats.peak_allocated=stats.current_allocated;
  stats.n_allocations++;
  allocation_sizes[ptr] = {nbytes, device};

  return at::DataPtr(ptr, ptr, &DummyAllocator::deallocate, at::Device(at::kPrivateUse1, device));
}

void DummyAllocator::deallocate(void* ptr) {
  std::lock_guard<std::mutex> lock(g_allocator.mutex_); 
  auto it = g_allocator.allocation_sizes.find(ptr);
  size_t nbytes = it->second.first;
  DeviceIndex device = it->second.second;
  auto& stats = g_allocator.per_device_stats[device];
  stats.current_allocated-=nbytes;
  stats.n_deallocations++;
  g_allocator.allocation_sizes.erase(it);
  free(ptr);
}

void DummyAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
  (void) count;
  memcpy(dest, src, 1);
}

gcs::sim::MemStats DummyAllocator::getStats(DeviceIndex device){
  std::lock_guard<std::mutex> lock(mutex_); 
  return per_device_stats[device];
}

void DummyAllocator::resetStats(DeviceIndex device){
  per_device_stats[device] = gcs::sim::MemStats{};
}

}