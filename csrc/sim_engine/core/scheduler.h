#include <vector>
#include <string>

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

std::vector<std::vector<StreamTimeline>> get_timeline();

}
