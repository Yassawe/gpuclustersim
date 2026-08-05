#include <c10/core/Stream.h>


namespace c10::gpuclustersim{

StreamId getSimStream(DeviceIndex device);

StreamId getDefaultSimStream(DeviceIndex device); 

StreamId getNewSimStream(DeviceIndex device);

StreamId exchangeSimStream(DeviceIndex device, StreamId s);

void synchronizeSimStream(DeviceIndex device, StreamId s); // for later

} //namespace c10::gpuclustersim