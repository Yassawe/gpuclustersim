#pragma once
#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>

// providing hostalloc for pinned memory just in case, so that it doesn't raise exception in some training scripts that use it
// this is just a stub it does nothing, fancy malloc wrapper
// allocating real memory should be fine here since pinned host memory is short lived and shouldn't take much space, it is batch data not params

namespace c10::gpuclustersim {

class DummyHostAllocator : public at::HostAllocator {
  at::DataPtr allocate(size_t nbytes) override;
  static void deallocate(void* ptr);
  void copy_data(void* dest, const void* src, std::size_t count) const override;

  //noop stubs
  bool record_event(void* ptr, void* ctx, c10::Stream stream) override {return true;}
  void empty_cache() override {}
  at::HostStats get_stats() override {
    return at::HostStats();
  }
  void reset_accumulated_stats() override {}
  void reset_peak_stats() override {}  
};

} // namespace c10::gpuclustersim

