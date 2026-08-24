#pragma once
#include <c10/core/Stream.h>


namespace c10::gpuclustersim{

StreamId gcsGetStream(DeviceIndex device_id);

StreamId gcsGetDefaultStream(DeviceIndex device_id); 

StreamId gcsGetNewStream(DeviceIndex device_id);

StreamId gcsExchangeStream(DeviceIndex device_id, StreamId stream_id);


} //namespace c10::gpuclustersim