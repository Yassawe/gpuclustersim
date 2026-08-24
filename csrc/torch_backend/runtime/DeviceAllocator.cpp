#include "DeviceAllocator.h"
#include "DeviceFunctions.h"
#include <cstdlib>
#include <cstring>

namespace {
  static c10::gpuclustersim::GCSDeviceAllocator g_allocator;
  REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);
}


namespace c10::gpuclustersim {

GCSDeviceAllocator::GCSDeviceAllocator(): per_device_stats(gcsDeviceCount()) {}

at::DataPtr GCSDeviceAllocator::allocate(size_t nbytes) {
  std::lock_guard<std::mutex> lock(mutex_); 

  void* ptr = malloc(1); 
  DeviceIndex device_id = gcsCurrentDevice();  
  auto& stats = per_device_stats[device_id];
  stats.current_allocated+=nbytes;
  if (stats.current_allocated>stats.peak_allocated) stats.peak_allocated=stats.current_allocated;
  stats.n_allocations++;
  allocation_info[ptr] = {device_id, nbytes};

  return at::DataPtr(ptr, ptr, &GCSDeviceAllocator::deallocate, at::Device(at::kPrivateUse1, device_id));
}

void GCSDeviceAllocator::deallocate(void* ptr) {
  // very ugly to access a global instance, but because this function is static it can't access the instance members. and 
  // the allocate above requires it to be static, otherwise compile error.
  // this works because the class is a singleton, only one allocator thoughout all torch program lifetime

  std::lock_guard<std::mutex> lock(g_allocator.mutex_); 
  auto it = g_allocator.allocation_info.find(ptr);
  DeviceIndex device_id = it->second.first;
  size_t nbytes = it->second.second;
  auto& stats = g_allocator.per_device_stats[device_id];
  stats.current_allocated-=nbytes;
  stats.n_deallocations++;
  g_allocator.allocation_info.erase(it);
  free(ptr);
}

void GCSDeviceAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
  (void) count;
  memcpy(dest, src, 1);
}

gcs::sim::MemStats GCSDeviceAllocator::getStats(DeviceIndex device_id){
  std::lock_guard<std::mutex> lock(mutex_); 
  return per_device_stats[device_id];
}

void GCSDeviceAllocator::resetStats(DeviceIndex device_id){
  per_device_stats[device_id] = gcs::sim::MemStats{};
}

DeviceIndex GCSDeviceAllocator::PtrToDevice(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_); 
  auto it = allocation_info.find(ptr);
  if (it != allocation_info.end()) return it->second.first;
  return -1;
}

}