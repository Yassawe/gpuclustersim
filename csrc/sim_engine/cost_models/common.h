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

}