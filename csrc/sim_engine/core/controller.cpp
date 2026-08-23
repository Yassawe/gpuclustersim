// this is the file that will have submit_comp_op. submit_memory_event, submit_comm_op, etc and call all various other parts, including placing on a scheduler
#include "controller.h"
#include "cost_models/compute.h"

namespace gcs::sim {

ENABLE_EXPORT void submit_compute_op(int device, int stream, cost_models::OpSpec& op_spec) {
  (void) device;
  (void) stream;
  (void) op_spec;

  

  //todo
}

}