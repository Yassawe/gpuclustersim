#pragma once
#include <string>
#include <vector>

namespace gcs::sim{

struct ENABLE_EXPORT ScheduledOp{
  std::string name;
  double start_time; //us
  double end_time; //us
};

struct ENABLE_EXPORT StreamTimeline{
  double current_time = 0; //us
  std::vector<ScheduledOp> ops;
};

double get_current_stream_time(int device, int stream);
void advance_current_stream_time(int device, int stream, double time);
void schedule_op_on_timeline(int device, int stream, std::string name, double duration); 

ENABLE_EXPORT std::vector<std::vector<StreamTimeline>> get_timeline();

}
