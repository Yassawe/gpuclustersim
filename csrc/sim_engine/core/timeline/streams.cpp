#include "streams.h"

#include <vector>
#include <mutex>

namespace gcs::sim {

static std::mutex mutex_; 

static std::vector<int> device_next_stream_id;
static thread_local std::vector<int> device_current_stream;

static void ensure_size(int device) {
  if (device < 0) device = 0;
  if (device>=device_current_stream.size()){
    device_current_stream.resize(device+1, 0); // by default stream 0 is default
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if(device>=device_next_stream_id.size()){
    device_next_stream_id.resize(device+1, 1); // 0 is presumed as default stream, so next is init as 1
  }
}

int create_stream(int device) {
  ensure_size(device);

  std::lock_guard<std::mutex> lock(mutex_);
  int stream_id = device_next_stream_id[device];
  device_next_stream_id[device]++;

  return stream_id;
}

int default_stream(int device) {
  return 0;
}

int current_stream(int device) {
  ensure_size(device);
  return device_current_stream[device];
}

int exchange_stream(int device, int stream_id) {
  ensure_size(device);
  int old_stream_id = device_current_stream[device];
  device_current_stream[device] = stream_id;
  return old_stream_id;
}

} // namespace gcs::sim