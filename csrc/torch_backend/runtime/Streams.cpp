#include "Streams.h"
#include <sim_engine.h>

namespace c10::gpuclustersim {


StreamId getSimStream(DeviceIndex device) {
  int device_idx = static_cast<int>(device);
  int stream_idx = gcs::sim::current_stream(device_idx);
  return static_cast<StreamId>(stream_idx);
}


StreamId getDefaultSimStream(DeviceIndex device){
  int device_idx = static_cast<int>(device);
  int stream_idx = gcs::sim::default_stream(device_idx);
  return static_cast<StreamId>(stream_idx);
}


StreamId getNewSimStream(DeviceIndex device){
  int device_idx = static_cast<int>(device);
  int stream_idx = gcs::sim::create_stream(device_idx);
  return static_cast<StreamId>(stream_idx);
}


StreamId exchangeSimStream(DeviceIndex device, StreamId s){
  int device_idx = static_cast<int>(device);
  int stream_idx = static_cast<int>(s);
  int old_stream_idx = gcs::sim::exchange_stream(device_idx, stream_idx);
  return static_cast<StreamId>(old_stream_idx);
}


} //namespace