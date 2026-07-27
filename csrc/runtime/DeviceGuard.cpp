#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

namespace c10::gpuclustersim {

c10::DeviceIndex current_device_ = 0;

struct SimulatorDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  c10::DeviceType type() const override {
    return static_type;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    auto old = current_device_;
    current_device_ = d.index();
    return c10::Device(static_type, old);
  }

  c10::Device getDevice() const override {
    return c10::Device(static_type, current_device_);
  }

  void setDevice(c10::Device d) const override {
    current_device_ = d.index();
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    current_device_ = d.index();
  }

  c10::Stream getStream(c10::Device d) const override {
    auto s = c10::Stream(c10::Stream::DEFAULT, d);
    return s;
  }
  c10::Stream exchangeStream(c10::Stream s) const override {
    // Minimal: just return default stream for previous
    return c10::Stream(c10::Stream::DEFAULT, s.device());
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return 1; //TODO: this will be managed by simulator engine
  }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, SimulatorDeviceGuardImpl);

} // namespace c10::gpuclustersim