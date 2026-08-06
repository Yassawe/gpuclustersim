#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/DeviceCapability.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/macros/Macros.h>

namespace c10::gpuclustersim {

class SimGuard : public c10::impl::DeviceGuardImplInterface { 

  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;
  
  DeviceType type() const override {
    return static_type;
  };

  // device
  DeviceIndex deviceCount() const noexcept override;

  Device exchangeDevice(Device device) const override;

  Device getDevice() const override;

  void setDevice(Device) const override;

  void uncheckedSetDevice(Device) const noexcept override;

  DeviceCapability getDeviceCapability(Device device) const override;
  
  void synchronizeDevice(const DeviceIndex device_id) const override;

  // streams

  Stream getStream(Device) const override; 

  Stream getDefaultStream(Device device) const override;

  Stream getStreamFromGlobalPool(Device device, bool isHighPriority = false) const override;

  Stream getNewStream(Device device, int priority = 0) const override;

  Stream exchangeStream(Stream stream) const override;

  void* getStreamNativeHandle(const Stream stream) const override;

  bool queryStream(const Stream& stream) const override;

  void synchronizeStream(const Stream& stream) const override; 

  // events

  void record(void** event, const Stream& stream, const DeviceIndex device_id, const c10::EventFlag flag) const override;

  void block(void* event, const Stream& stream) const override;

  bool queryEvent(void* event) const override;

  void synchronizeEvent(void* event) const override;

  double elapsedTime(
      void* event1,
      void* event2,
      const DeviceIndex device_id) const override;

};


}