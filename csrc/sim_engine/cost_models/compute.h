#pragma once
#include <utils/Macros.h>
#include "common.h"
#include "core/platform/device.h"
#include <string>

namespace gcs::sim::cost_models {
  
struct ENABLE_EXPORT ArgSpec {
  std::string name;
  enum class Type {Tensor, TensorList, Scalar, ScalarList, Other};
  Type type;
  TensorSpec tensor;
  std::vector<TensorSpec> tensor_list;
  ScalarSpec scalar;
  std::vector<ScalarSpec> scalar_list;
};
 
struct ENABLE_EXPORT OpSpec {
  std::string name;
  std::vector<ArgSpec> inputs;
  std::vector<ArgSpec> outputs;
};

struct OpCost{
  int64_t flops; // total flops of the op
  int64_t bytes; // memory bandwidth cost, reads and writes
  DataType dominant_dtype; 
};

double estimate_compute_duration(OpSpec& op_spec, DeviceSpec& device_spec);

}

