#include "compute.h"
#include <vector>
#include <unordered_map>
#include <functional>


namespace gcs::sim::cost_models {


OpCost gemm_cost(OpSpec& op_spec){
  return OpCost{0, 0};
}

OpCost default_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}



std::function<OpCost(OpSpec&)> get_cost_fn(OpSpec& op_spec){

  static const std::unordered_map<std::string, std::function<OpCost(OpSpec&)>> registry = {
    {},
    {},
  };

  auto it = registry.find(op_spec.name);
  if (it!=registry.end()){
    return it->second;
  }
  return default_cost;
}

double duration_fn(OpCost& op_cost, DeviceSpec& device_spec){
  return 0;
}


double estimate_compute_duration(OpSpec& op_spec, DeviceSpec& device_spec){
  std::function<OpCost(OpSpec&)> cost_fn = get_cost_fn(op_spec);
  OpCost op_cost = cost_fn(op_spec);
  return duration_fn(op_cost, device_spec);
}

}