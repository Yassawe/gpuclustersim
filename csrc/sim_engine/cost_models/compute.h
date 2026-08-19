#pragma once
#include <utils/Macros.h>
#include <string>
#include <vector>

namespace gcs::sim::cost_models {

struct ENABLE_EXPORT TensorSpec {
  std::vector<int64_t> sizes; 
  int dtype_size;
  bool defined = true;
};

struct ENABLE_EXPORT ScalarSpec {
  enum class Type {Int, Float, Bool};
  Type type;
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
  std::string overload_name;
  std::vector<ArgSpec> inputs;
  std::vector<ArgSpec> outputs;
};

}

