#pragma once
#include <utils/Macros.h>
#include <string>
#include <vector>
#include "common.h"
#include "core/platform/topology.h"

namespace gcs::sim::cost_models {
  
struct ENABLE_EXPORT CommSpec{
  std::string name;
  
  int rank; // in process-group rank
  int world_size; // process group world size, i.e. DP/TP/MP
  std::vector<TensorSpec> payload;

  std::vector<int64_t> participants; // map from process group rank -> global rank, mirror of torch global_rank_per_group
  int root = 0;
  int peer = -1; // for direct send/recv
  std::vector<int64_t> input_counts; // for all-to-all
  std::vector<int64_t> output_counts;
};


}