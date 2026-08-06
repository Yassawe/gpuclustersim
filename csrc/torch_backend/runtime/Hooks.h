#include <c10/core/Allocator.h>
#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Device.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace c10::gpuclustersim {

class SimHooksInterface : public at::PrivateUse1HooksInterface {

  bool hasPrimaryContext(DeviceIndex device_id) const override {return true;}
  bool isAvailable() const override {return true;}

  DeviceIndex deviceCount() const override;
  void setCurrentDevice(DeviceIndex device_id) const override;
  DeviceIndex getCurrentDevice() const override;
  DeviceIndex exchangeDevice(DeviceIndex device_id) const override;
  DeviceIndex maybeExchangeDevice(DeviceIndex device_id) const override;
  at::Allocator* getPinnedMemoryAllocator() const override;
  at::Device getDeviceFromPtr(void* data) const override;
  const at::Generator& getDefaultGenerator(DeviceIndex device_id) const override;
  at::Generator getNewGenerator(DeviceIndex device_id) const override;
  virtual void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const override {};

};


}