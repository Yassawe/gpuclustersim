#pragma once
#include <utils/Macros.h>

namespace gcs::sim::cost_models{
  
enum class ENABLE_EXPORT DataType {
    FP64, 
    FP32, 
    FP16, 
    FP8, 
    INT64, 
    INT32, 
    INT16,
    INT8, 
    BOOL,
    Other
};

struct ENABLE_EXPORT TensorSpec {
  std::vector<int64_t> dims; 
  DataType dtype;
  int dtype_size;
  bool defined = true;
};

struct ENABLE_EXPORT ScalarSpec {
  DataType dtype;
  double value = 0; // scalars in torch stack are things like stride, kernel_size, etc, so i do actually need to know them
};

}