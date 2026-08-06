#include "Hooks.h"
#include "Generator.h"
#include "DeviceAllocator.h"
#include "DeviceFunctions.h"


namespace c10::gpuclustersim {

DeviceIndex SimHooksInterface::deviceCount() const {
  return device_count();
}

void SimHooksInterface::setCurrentDevice(DeviceIndex device_id) const {
  set_device(device_id);
}

DeviceIndex SimHooksInterface::getCurrentDevice() const {
  return current_device();
}

DeviceIndex SimHooksInterface::exchangeDevice(DeviceIndex device_id) const {
  return exchange_device(device_id);
}

DeviceIndex SimHooksInterface::maybeExchangeDevice(DeviceIndex device_id) const {
  return exchange_device(device_id);
}

at::Allocator* SimHooksInterface::getPinnedMemoryAllocator() const {
  return at::getHostAllocator(at::kPrivateUse1);
}

at::Device SimHooksInterface::getDeviceFromPtr(void* data) const {
  auto* allocator = static_cast<c10::gpuclustersim::DummyAllocator*>(c10::GetAllocator(at::kPrivateUse1));
  DeviceIndex device_id = allocator->PtrToDevice(data);
  if (device_id == -1) return at::Device(at::kPrivateUse1, 0);
  return Device(c10::DeviceType::PrivateUse1, device_id);
}

const at::Generator& SimHooksInterface::getDefaultGenerator(DeviceIndex device_id) const {
  return getGenerator(device_id);
}

at::Generator SimHooksInterface::getNewGenerator(DeviceIndex device_id) const {
  return at::make_generator<DummyGenerator>(device_id);
}


static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new SimHooksInterface());
  return true;
}();


} // namespace
