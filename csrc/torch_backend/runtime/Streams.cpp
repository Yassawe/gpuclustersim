#include "Streams.h"
#include <sim_engine.h>

namespace c10::gpuclustersim {


StreamId getSimStream(DeviceIndex device_id) {
  int stream_idx = gcs::sim::current_stream(static_cast<int>(device_id));
  return static_cast<StreamId>(stream_idx);
}


StreamId getDefaultSimStream(DeviceIndex device_id){
  int stream_idx = gcs::sim::default_stream(static_cast<int>(device_id));
  return static_cast<StreamId>(stream_idx);
}


StreamId getNewSimStream(DeviceIndex device_id){
  int stream_idx = gcs::sim::create_stream(static_cast<int>(device_id));
  return static_cast<StreamId>(stream_idx);
}


StreamId exchangeSimStream(DeviceIndex device_id, StreamId stream_id){
  int old_stream_idx = gcs::sim::exchange_stream(static_cast<int>(device_id), static_cast<int>(stream_id));
  return static_cast<StreamId>(old_stream_idx);
}


} //namespace