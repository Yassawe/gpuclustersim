#pragma once
#include <c10/core/Stream.h>
#include <utils/Macros.h>


namespace c10::gpuclustersim{

ENABLE_EXPORT StreamId gcsGetStream(DeviceIndex device_id);

ENABLE_EXPORT StreamId gcsGetDefaultStream(DeviceIndex device_id); 

ENABLE_EXPORT StreamId gcsGetNewStream(DeviceIndex device_id);

ENABLE_EXPORT StreamId gcsExchangeStream(DeviceIndex device_id, StreamId stream_id);


} //namespace c10::gpuclustersim