#include "events.h"
#include "scheduler.h"

#include <vector>
#include <algorithm>

namespace gcs::sim{

struct SimEvent{
  double timestamp = -1;
  bool recorded = false;
}; 


void event_record(void** event, int device, int stream){
  if (*event==nullptr){
    *event = new SimEvent();
  }
  auto e = static_cast<SimEvent*>(*event);
  double stream_time = get_current_stream_time(device, stream);
  e->timestamp = stream_time;
  e->recorded = true;
} 


void event_block(void* event, int device, int stream){
  if (event==nullptr) return;
  auto e = static_cast<SimEvent*>(event);
  advance_current_stream_time(device, stream, e->timestamp);
} 


bool event_query(void* event){
  if (event==nullptr){
    return true;
  }
  auto e = static_cast<SimEvent*>(event);
  return e->recorded;
}


double event_elapsed_time(void* event1, void* event2){
  auto e1 = static_cast<SimEvent*>(event1);
  auto e2 = static_cast<SimEvent*>(event2);
  if (e1->recorded && e2->recorded) return e2->timestamp - e1->timestamp;
  return 0;
}

void event_destroy(void* event){
  delete static_cast<SimEvent*>(event);
}

}
