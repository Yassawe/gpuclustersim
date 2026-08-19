#include "compute.h"


namespace gcs::sim::cost_models {

enum class OpFamily {
  Unknown, GEMM, Conv, Elementwise, Reduction, Memory, Embedding, Attention
};


}