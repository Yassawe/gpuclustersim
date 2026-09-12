#pragma once
#include <string>
#include <vector>
#include <utils/Macros.h>

namespace gcs::sim{

struct ENABLE_EXPORT ScheduledOp{
  std::string name;
  double start_time; //ns
  double end_time; //ns
};

struct ENABLE_EXPORT StreamTimeline{
  double current_time = 0; //us
  std::vector<ScheduledOp> ops;
};

ENABLE_EXPORT double get_current_stream_time(int device, int stream);
ENABLE_EXPORT void advance_current_stream_time(int device, int stream, double time);

void schedule_op_on_timeline(int device, int stream, std::string name, double duration); 

ENABLE_EXPORT std::vector<std::vector<StreamTimeline>> get_timeline();
ENABLE_EXPORT void reset_timeline();

}
