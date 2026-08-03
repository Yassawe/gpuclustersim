#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/Device.h>
#include <c10/macros/Macros.h>




namespace c10::gpuclustersim {

class SimGuard : public c10::impl::DeviceGuardImplInterface { 

  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;
  
  DeviceType type() const override {
    return static_type;
  };

  // device
  DeviceIndex deviceCount() const noexcept override;

  Device exchangeDevice(Device d) const override;

  Device getDevice() const override;

  void setDevice(Device) const override;

  void uncheckedSetDevice(Device) const noexcept override;

  DeviceCapability getDeviceCapability(Device /*unused*/) const override;
  
  void synchronizeDevice(const DeviceIndex /*device_index*/) const override;

  // stream

  Stream getStream(Device) const override; 

  Stream getDefaultStream(Device /*unused*/) const override;

  Stream getStreamFromGlobalPool(Device /*unused*/, bool isHighPriority = false) const override;

  Stream getNewStream(Device /*unused*/, int priority = 0) const override;

  Stream exchangeStream(Stream) const override;

  void* getStreamNativeHandle(const Stream) const override;

  bool queryStream(const Stream& /*stream*/) const override;

  void synchronizeStream(const Stream& /*stream*/) const override; 

  // events


  void record(void** /*event*/, const Stream& /*stream*/, const DeviceIndex /*device_index*/, const c10::EventFlag /*flag*/) const override;

  void block(void* /*event*/, const Stream& /*stream*/) const override;

  bool queryEvent(void* /*event*/) const override;

  void synchronizeEvent(void* /*event*/) const override;

  double elapsedTime(
      void* /*event1*/,
      void* /*event2*/,
      const DeviceIndex /*device_index*/) const override;

};


}