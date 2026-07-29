#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include "Generator.h"


namespace c10::gpuclustersim {
struct DummyHooksInterface : public at::PrivateUse1HooksInterface {
  DummyHooksInterface() {};
  ~DummyHooksInterface() override = default;

  void init() const override {
    // This is called when PyTorch first accesses the device
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    return true;
  }

  bool isBuilt() const override {
    return true;
  }

  bool isAvailable() const override {
    return true; //implement
  }

  DeviceIndex deviceCount() const override {
    return 1; //implement
  }

  void setCurrentDevice(DeviceIndex device) const override {
    // noop
  }

  DeviceIndex getCurrentDevice() const override {
    return 0; //implement
  }

  DeviceIndex exchangeDevice(DeviceIndex device) const override {
    return device; //noop
  }

  DeviceIndex maybeExchangeDevice(DeviceIndex device) const override {

    return device; //noop
  }

  at::Allocator* getPinnedMemoryAllocator() const override {
    return at::getHostAllocator(at::kPrivateUse1);
  }

  bool isPinnedPtr(const void* data) const override {
    return false;
  }

  at::Device getDeviceFromPtr(void* data) const override {
        return at::Device(at::DeviceType::PrivateUse1, 0);
  }

  const at::Generator& getDefaultGenerator(DeviceIndex device_index) const override {
    return getDefaultGenerator(device_index);
  }
  
  at::Generator getNewGenerator(DeviceIndex device_index) const override {
    return at::make_generator<DummyGenerator>(device_index);
  }
};


static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new DummyHooksInterface());
  return true;
}();


} // namespace c10::openreg
