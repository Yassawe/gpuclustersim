#include "controller.h"
#include "core/platform/device.h"
#include "core/platform/topology.h"
#include "cost_models/compute.h"
#include "cost_models/communication.h"
#include "core/timeline/scheduler.h"
#include <string>

namespace gcs::sim {

void submit_compute_op(int device, int stream, cost_models::OpSpec& op_spec) {
  DeviceSpec device_spec = get_device_spec();
  double duration = cost_models::estimate_compute_duration(op_spec, device_spec);
  schedule_op_on_timeline(device, stream, op_spec.name, duration);
}

void submit_communication_op(int device, int stream, cost_models::CommSpec& comm_spec){
  // TopologySpec top_spec = get_topology();
  // double duration = cost_models::estimate_communication_duration(comm_spec, topology_spec);
  double duration = 100000; //debug
  schedule_op_on_timeline(device, stream, comm_spec.name, duration);
}


}