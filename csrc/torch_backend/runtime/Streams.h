#pragma once
#include <c10/core/Stream.h>


namespace c10::gpuclustersim{

StreamId getSimStream(DeviceIndex device_id);

StreamId getDefaultSimStream(DeviceIndex device_id); 

StreamId getNewSimStream(DeviceIndex device_id);

StreamId exchangeSimStream(DeviceIndex device_id, StreamId stream_id);


} //namespace c10::gpuclustersim