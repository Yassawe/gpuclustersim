#include "sim_engine.h"

#include <vector>
#include <mutex>

namespace gcs::sim {

static std::mutex mutex_; 

std::vector<int> device_next_stream_id;
thread_local std::vector<int> device_current_stream;

static void ensure_size(int device) {
  if (d>=device_next_stream_id.size() || d>=device_current_stream.size()){
    device_next_stream_id.resize(d+1, 1);
    device_current_stream.resize(d+1, 0);
  }
}

ENABLE_EXPORT int create_stream(int device) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_size(device);
  int stream_id = device_next_stream_id[device];
  device_next_stream_id[device]++;
  return stream_id;
}

ENABLE_EXPORT int default_stream(int device) {
  int stream_id = 0; 
  device_current_stream[device]=stream_id;
  return stream_id;
}

ENABLE_EXPORT int current_stream(int device) {
  ensure_size(device);
  return device_current_stream[device];
}

ENABLE_EXPORT void set_stream(int device, int stream_id) {;
  ensure_size(device);
  device_current_stream[device] = stream_id;
}

ENABLE_EXPORT int exchange_stream(int device, int stream_id) {
  ensure_size(device);
  int old_stream_id = device_current_stream[device];
  device_current_stream[device] = stream_id;
  return old;
}

} // namespace gcs::sim