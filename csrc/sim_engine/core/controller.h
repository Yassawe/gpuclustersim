#pragma once
#include <utils/Macros.h>
#include "cost_models/compute.h"
#include "cost_models/communication.h"

namespace gcs::sim{

ENABLE_EXPORT void submit_compute_op(int device, int stream, cost_models::OpSpec& op_spec);
ENABLE_EXPORT void submit_communication_op(int device, int stream, cost_models::CommSpec& comm_spec);

}