#include "compute.h"


namespace gcs::sim::cost_models {

enum class OpFamily {
  Unknown, GEMM, Conv, Elementwise, Reduction, Memory, Embedding, Attention
};

double estimate_compute_duration(OpSpec& op_spec, DeviceSpec& device_spec){
  (void) op_spec;
  (void) device_spec;
}

}