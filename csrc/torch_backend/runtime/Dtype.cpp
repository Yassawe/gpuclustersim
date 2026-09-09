#include "Dtype.h"

namespace c10::gpuclustersim {

gcs::sim::cost_models::DataType map_dtype(at::ScalarType st){
  switch (st) {
    case at::ScalarType::Double: return gcs::sim::cost_models::DataType::FP64;
    case at::ScalarType::Float: return gcs::sim::cost_models::DataType::FP32;
    case at::ScalarType::Half: return gcs::sim::cost_models::DataType::FP16;
    case at::ScalarType::BFloat16: return gcs::sim::cost_models::DataType::FP16; 
    case at::ScalarType::Float8_e5m2: return gcs::sim::cost_models::DataType::FP8; 
    case at::ScalarType::Float8_e4m3fn: return gcs::sim::cost_models::DataType::FP8;
    case at::ScalarType::Float8_e5m2fnuz: return gcs::sim::cost_models::DataType::FP8;
    case at::ScalarType::Float8_e4m3fnuz: return gcs::sim::cost_models::DataType::FP8;
    case at::ScalarType::Float8_e8m0fnu: return gcs::sim::cost_models::DataType::FP8;
                                       
    case at::ScalarType::Byte: return gcs::sim::cost_models::DataType::INT8;
    case at::ScalarType::Char: return gcs::sim::cost_models::DataType::INT8;
    case at::ScalarType::Short: return gcs::sim::cost_models::DataType::INT16;
    case at::ScalarType::Int: return gcs::sim::cost_models::DataType::INT32;
    case at::ScalarType::Long: return gcs::sim::cost_models::DataType::INT64;
    case at::ScalarType::UInt16: return gcs::sim::cost_models::DataType::INT16; //unsigned are same costwise&memorywise
    case at::ScalarType::UInt32: return gcs::sim::cost_models::DataType::INT32;
    case at::ScalarType::UInt64: return gcs::sim::cost_models::DataType::INT64;

    case at::ScalarType::Bool: return gcs::sim::cost_models::DataType::BOOL;

    default: return gcs::sim::cost_models::DataType::Other;
  }
}

}
