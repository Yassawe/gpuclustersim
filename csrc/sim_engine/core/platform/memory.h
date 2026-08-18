#pragma once
#include <utils/Macros.h>

namespace gcs::sim{

struct ENABLE_EXPORT MemStats {
  int current_allocated = 0;
  int peak_allocated = 0;
  int n_allocations = 0;
  int n_deallocations = 0;
};

//todo: memory bookkeeping

}