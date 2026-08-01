#pragma once

#include <utils/Macros.h>


namespace gcs::sim {

// memory

struct ENABLE_EXPORT MemStats {
  int current_allocated = 0;
  int peak_allocated = 0;
  int n_allocations = 0;
  int n_deallocations = 0;
};


// device

ENABLE_EXPORT int device_count();

ENABLE_EXPORT int current_device();

ENABLE_EXPORT void set_device(int device);

ENABLE_EXPORT int exchange_device(int device);

}