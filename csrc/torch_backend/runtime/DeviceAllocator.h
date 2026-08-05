#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <sim_engine.h>

// cost models infer time from shape or from lookup tables, pytorch manages shape on its own, the actual data can be dummy 1 byte, as long as nothing dereferences it. 
// this allows to simulate huge tensors accross many fake devices, without running out of real memory by allocating only 1 byte for each supposed tensor
// extreme caution must be taken so that nothing dereferences it, otherwise segfault and undefined behavior


namespace c10::gpuclustersim {


class DummyAllocator : public c10::Allocator{

public:
  DummyAllocator();

  at::DataPtr allocate(size_t nbytes) override;
  static void deallocate(void* ptr);
  void copy_data(void* dest, const void* src, std::size_t count) const override;
  gcs::sim::MemStats getStats(DeviceIndex device); 
  void resetStats(DeviceIndex device);
  DeviceIndex PtrToDevice(void* ptr);

private: 
  mutable std::mutex mutex_;
  std::vector<gcs::sim::MemStats> per_device_stats;
  std::unordered_map<void*, std::pair<DeviceIndex, size_t>> allocation_info;
};


}