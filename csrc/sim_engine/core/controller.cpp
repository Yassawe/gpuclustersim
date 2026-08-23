#include "controller.h"
#include "core/platform/device.h"
#include "cost_models/compute.h"
#include "core/timeline/scheduler.h"
#include <string>

namespace gcs::sim {

ENABLE_EXPORT void submit_compute_op(int device, int stream, cost_models::OpSpec& op_spec) {
  DeviceSpec device_spec = get_device_spec();
  double duration = cost_models::estimate_compute_duration(op_spec, device_spec);
  schedule_op_on_timeline(device, stream, op_spec.name, duration);
}



}