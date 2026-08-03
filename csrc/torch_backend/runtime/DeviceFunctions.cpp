#include "DeviceFunctions.h"
#include <sim_engine.h>

namespace c10::gpuclustersim {

ENABLE_EXPORT DeviceIndex device_count(){
  return static_cast<DeviceIndex>(gcs::sim::device_count());
}

ENABLE_EXPORT DeviceIndex current_device(){
  return static_cast<DeviceIndex>(gcs::sim::current_device());
}

ENABLE_EXPORT void set_device(DeviceIndex device){
  gcs::sim::set_device(static_cast<int>(device));
}

ENABLE_EXPORT DeviceIndex exchange_device(DeviceIndex device){
  return static_cast<DeviceIndex>(gcs::sim::exchange_device(static_cast<int>(device)));
}

ENABLE_EXPORT DeviceIndex maybe_exchange_device(DeviceIndex device){
  return static_cast<DeviceIndex>(gcs::sim::exchange_device(static_cast<int>(device))); // no cost to changing so doesn't matter, cold path anyways.
}

}