#pragma once
#include <utils/Macros.h>
#include <string>
#include <vector>
#include "common.h"
#include "core/platform/device.h" // for DeviceSpec type

namespace gcs::sim::cost_models {

struct ENABLE_EXPORT TensorSpec {
  std::vector<int64_t> dims; 
  DataType dtype;
  int dtype_size;
  int64_t numel; // redundant technically, but easier to store than to recompute
  bool defined = true;
};

struct ENABLE_EXPORT ScalarSpec {
  DataType dtype;
  double value = 0; // scalars in torch stack are things like stride, kernel_size, etc, so i do actually need to know them
};
  
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
  int64_t bytes; // memory bandwidth cost, it is not 1 to 1 to just tensor size, e.g. matmul reads twice writes once
  DataType dominant_dtype; 
};

double estimate_compute_duration(OpSpec& op_spec, DeviceSpec& device_spec);

}

