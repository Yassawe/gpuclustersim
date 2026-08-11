#include <vector>
#include <string>

namespace gcs::sim{

struct Scheduled_Op{
  std::string name;
  double start_time;
  double end_time;
};

struct StreamTimeline{
  double current_time = 0;
  std::vector<Scheduled_Op> ops;
};

get_current_stream_time(int device, int stream); 
advance_current_stream_time(int device, int stream, double time);

}