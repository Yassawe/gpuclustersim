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
  std::string overload_name;
  std::vector<ArgSpec> inputs;
  std::vector<ArgSpec> outputs;
};

double estimate_compute_duration(OpSpec& op_spec, DeviceSpec& device_spec);

}

