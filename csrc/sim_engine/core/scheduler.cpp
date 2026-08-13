#include "sim_engine.h"
#include "core/scheduler.h"
#include <mutex>


namespace gcs::sim {

static std::mutex mutex_;

static std::vector<std::vector<StreamTimeline>> timeline;

static void ensure_size(int device, int stream) {
  if (device < 0) device = 0;
  if (stream < 0) stream = 0;

  if (device >= timeline.size()) {
    timeline.resize(device + 1);
  }

  auto& d = timeline[device];
  if (stream >= d.size()) {
    d.resize(stream + 1);
  }
}

double get_current_stream_time(int device, int stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_size(device, stream);
  return timeline[device][stream].current_time;
}

void advance_current_stream_time(int device, int stream, double time) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_size(device, stream);
  auto& t = timeline[device][stream];
  if (time > t.current_time) t.current_time = time;
}


void schedule_op(int device, int stream, std::string name, double duration) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_size(device, stream);
  auto& t = timeline[device][stream];
  double start = t.current_time;
  double end = start+duration;
  t.ops.push_back(ScheduledOp{name, start, end});
  t.current_time = end;
}

std::vector<std::vector<StreamTimeline>> get_timeline() {
  std::lock_guard<std::mutex> lock(mutex_);
  return timeline;
}

} 
