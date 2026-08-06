#include <c10/core/Device.h>
#include <utils/Macros.h>


namespace c10::gpuclustersim {

// device
ENABLE_EXPORT DeviceIndex device_count();
ENABLE_EXPORT DeviceIndex current_device();
ENABLE_EXPORT void set_device(DeviceIndex device_id);
ENABLE_EXPORT DeviceIndex exchange_device(DeviceIndex device_id);
ENABLE_EXPORT DeviceIndex maybe_exchange_device(DeviceIndex device_id);

// streams





}