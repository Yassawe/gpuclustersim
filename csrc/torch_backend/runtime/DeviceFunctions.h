#pragma once
#include <c10/core/Device.h>
#include <utils/Macros.h>


namespace c10::gpuclustersim {

DeviceIndex gcsDeviceCount();
DeviceIndex gcsCurrentDevice();
void gcsSetDevice(DeviceIndex device_id);
DeviceIndex gcsExchangeDevice(DeviceIndex device_id);
DeviceIndex gcsMaybeExchangeDevice(DeviceIndex device_id);

}