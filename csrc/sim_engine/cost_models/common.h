#pragma once
#include <utils/Macros.h>
#include <vector>

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
  int64_t numel; // redundant technically, but easier to store than to recompute
  bool defined = true;
  // for some compute ops torch incldues preallocated write buffers tensors in inputs, this messes up my calculations since they are counted as real tensors, so outputs are counted twice (in inputs and in outputs), so i need a way to mark them clearly
  bool is_input_write_buffer = false; 
};

struct ENABLE_EXPORT ScalarSpec {
  DataType dtype;
  double value = 0; // scalars in torch stack are things like stride, kernel_size, mask, etc, so i do actually need to know them
};

}