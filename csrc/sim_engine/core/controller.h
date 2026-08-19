#pragma once
#include <utils/Macros.h>
#include "cost_models/compute.h" //for types

namespace gcs::sim{


ENABLE_EXPORT void submit_compute_op(int device, int stream, cost_models::OpSpec& op_spec);


}