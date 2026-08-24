#pragma once
#include <utils/Macros.h>
#include <string>
#include <vector>
#include "common.h"
#include "core/platform/device.h" // for DeviceSpec type

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
  int64_t bytes; // memory bandwidth cost, it is not 1 to 1 to just size, e.g. matmul reads twice writes once
  DataType dominant_dtype; 
};

double estimate_compute_duration(OpSpec& op_spec, DeviceSpec& device_spec);

}

