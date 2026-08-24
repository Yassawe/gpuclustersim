#pragma once
#include <c10/core/Device.h>
#include <c10/core/Stream.h>

namespace c10::gpuclustersim {

void gcsRecordEvent(void** event, DeviceIndex device_id, StreamId stream_id);
void gcsBlockEvent(void* event, DeviceIndex device_id, StreamId stream_id);
bool gcsQueryEvent(void* event); 
double gcsEventElapsedTime(void* event1, void* event2);
void gcsDestroyEvent(void* event);

} 