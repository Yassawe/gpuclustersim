#include "sim_engine.h"
#include <vector>


namespace gcs::sim {

struct TraceEvent{
  char* name;
  double start_time;
  double end_time;
};

struct StreamTimeline{
  double current_time = 0;
  std::vector<TraceEvent> events;
}


} //namespace gcs::sim