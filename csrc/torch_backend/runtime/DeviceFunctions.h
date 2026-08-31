#pragma once
#include <c10/core/Device.h>
#include <utils/Macros.h>


namespace c10::gpuclustersim {

ENABLE_EXPORT DeviceIndex gcsDeviceCount();
ENABLE_EXPORT DeviceIndex gcsCurrentDevice();
ENABLE_EXPORT void gcsSetDevice(DeviceIndex device_id);
ENABLE_EXPORT DeviceIndex gcsExchangeDevice(DeviceIndex device_id);
ENABLE_EXPORT DeviceIndex gcsMaybeExchangeDevice(DeviceIndex device_id);

}