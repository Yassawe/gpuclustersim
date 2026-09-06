#include "scheduler.h"
#include <mutex>
#include <string>
#include <algorithm>


namespace gcs::sim {

static std::mutex mutex_;

static std::vector<std::vector<StreamTimeline>> timeline; // [device][stream] -> StreamTimeline


double max_stream_time(int device){
  double t = 0;
  if (device<timeline.size()){
    auto& d = timeline[device];
    for(auto& s : d){
      t = std::max(t, s.current_time);
    }
  }
  return t;
}

static void ensure_size(int device, int stream) {
  if (device < 0) device = 0;
  if (stream < 0) stream = 0;

  if (device >= timeline.size()) {
    timeline.resize(device + 1);
  }
  auto& d = timeline[device];
  double stream_init_time = max_stream_time(device); //start new streams from the current device time
  while(d.size()<=stream){
    d.push_back(StreamTimeline{stream_init_time, {}});
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

void schedule_op_on_timeline(int device, int stream, std::string name, double duration) {
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

void reset_timeline(){
  timeline.clear();
}


} 
