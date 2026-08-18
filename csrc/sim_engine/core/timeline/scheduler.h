#pragma once
#include <string>
#include <vector>

namespace gcs::sim{

struct ScheduledOp{
  std::string name;
  double start_time;
  double end_time;
};

struct StreamTimeline{
  double current_time = 0;
  std::vector<ScheduledOp> ops;
};

double get_current_stream_time(int device, int stream);
void advance_current_stream_time(int device, int stream, double time);
void schedule_op_on_timeline(int device, int stream, std::string name, double duration); 
std::vector<std::vector<StreamTimeline>> get_timeline();

}
