#include "Events.h"
#include <sim_engine.h>

namespace c10::gpuclustersim {

void gcsRecordEvent(void** event, DeviceIndex device_id, StreamId stream_id){
  gcs::sim::event_record(event, static_cast<int>(device_id), static_cast<int>(stream_id));
}

void gcsBlockEvent(void* event, DeviceIndex device_id, StreamId stream_id){
  gcs::sim::event_block(event, static_cast<int>(device_id), static_cast<int>(stream_id));
}

bool gcsQueryEvent(void* event){
  return gcs::sim::event_query(event);
}

double gcsEventElapsedTime(void* event1, void* event2){
  return gcs::sim::event_elapsed_time(event1, event2);
}

void gcsDestroyEvent(void* event){
  gcs::sim::event_destroy(event);
}

}