#include <c10/core/Device.h>
#include <c10/core/Stream.h>

namespace c10::gpuclustersim {

void recordSimEvent(void** event, DeviceIndex device_id, StreamId stream_id);
void blockSimEvent(void* event, DeviceIndex device_id, StreamId stream_id);
bool querySimEvent(void* event); 
double simEventElapsedTime(void* event1, void* event2);
void destroySimEvent(void* event);

} 