#include "DeviceFunctions.h"
#include <sim_engine.h>

namespace c10::gpuclustersim {

ENABLE_EXPORT DeviceIndex gcsDeviceCount(){
  return static_cast<DeviceIndex>(gcs::sim::device_count());
}

ENABLE_EXPORT DeviceIndex gcsCurrentDevice(){
  return static_cast<DeviceIndex>(gcs::sim::current_device());
}

ENABLE_EXPORT void gcsSetDevice(DeviceIndex device_id){
  gcs::sim::set_device(static_cast<int>(device_id));
}

ENABLE_EXPORT DeviceIndex gcsExchangeDevice(DeviceIndex device_id){
  return static_cast<DeviceIndex>(gcs::sim::exchange_device(static_cast<int>(device_id)));
}

ENABLE_EXPORT DeviceIndex gcsMaybeExchangeDevice(DeviceIndex device_id){
  return static_cast<DeviceIndex>(gcs::sim::exchange_device(static_cast<int>(device_id))); // no cost to changing so doesn't matter, cold path anyways.
}

}