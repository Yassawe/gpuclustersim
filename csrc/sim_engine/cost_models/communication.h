#pragma once
#include <utils/Macros.h>
#include <string>
#include <vector>
#include "common.h"
#include "core/platform/topology.h"

namespace gcs::sim::cost_models {
  
struct ENABLE_EXPORT CommSpec{
  std::string name;
  int rank;
  int world_size;
  std::vector<TensorSpec> payload;
  int root = 0;
  int peer = -1; // for direct send/recv
  std::vector<int64_t> input_counts; // for all-to-all
  std::vector<int64_t> output_counts;
};


}