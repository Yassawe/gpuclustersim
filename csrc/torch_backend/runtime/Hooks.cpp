#include "Hooks.h"
#include "Generator.h"
#include "DeviceAllocator.h"
#include "DeviceFunctions.h"
#include <c10/core/StorageImpl.h>


namespace c10::gpuclustersim {

DeviceIndex GCSHooksInterface::deviceCount() const {
  return gcsDeviceCount();
}

void GCSHooksInterface::setCurrentDevice(DeviceIndex device_id) const {
  gcsSetDevice(device_id);
}

DeviceIndex GCSHooksInterface::getCurrentDevice() const {
  return gcsCurrentDevice();
}

DeviceIndex GCSHooksInterface::exchangeDevice(DeviceIndex device_id) const {
  return gcsExchangeDevice(device_id);
}

DeviceIndex GCSHooksInterface::maybeExchangeDevice(DeviceIndex device_id) const {
  return gcsMaybeExchangeDevice(device_id);
}

at::Allocator* GCSHooksInterface::getPinnedMemoryAllocator() const {
  return at::getHostAllocator(at::kPrivateUse1);
}

at::Device GCSHooksInterface::getDeviceFromPtr(void* data) const {
  auto* allocator = static_cast<c10::gpuclustersim::GCSDeviceAllocator*>(c10::GetAllocator(at::kPrivateUse1));
  DeviceIndex device_id = allocator->PtrToDevice(data);
  if (device_id == -1) return at::Device(at::kPrivateUse1, 0);
  return Device(c10::DeviceType::PrivateUse1, device_id);
}

const at::Generator& GCSHooksInterface::getDefaultGenerator(DeviceIndex device_id) const {
  return getGenerator(device_id);
}

at::Generator GCSHooksInterface::getNewGenerator(DeviceIndex device_id) const {
  return at::make_generator<GCSGenerator>(device_id);
}

void GCSHooksInterface::resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const {
  auto* impl = storage.unsafeGetStorageImpl(); // minimal stub required by FSDP to issue allgather
  impl->set_nbytes(newsize);
}


static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new GCSHooksInterface());
  return true;
}();


} // namespace
