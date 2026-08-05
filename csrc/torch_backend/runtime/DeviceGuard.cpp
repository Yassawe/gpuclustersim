#include "DeviceGuard.h"
#include "DeviceFunctions.h"

#include <c10/core/DeviceCapability.h>
#include <c10/core/Stream.h>

namespace c10::gpuclustersim {

// device

DeviceIndex SimGuard::deviceCount() const noexcept {
  return device_count();
}

Device SimGuard::exchangeDevice(Device d) const {
  auto old = exchange_device(d.index());
  return Device(static_type, old);
}

Device SimGuard::getDevice() const {
  return Device(static_type, current_device());
}

void SimGuard::setDevice(Device d) const {
  set_device(d.index());
}

void SimGuard::uncheckedSetDevice(Device d) const noexcept {
  set_device(d.index());
}

DeviceCapability SimGuard::getDeviceCapability(Device /*unused*/) const {
  return DeviceCapability{};
}

void SimGuard::synchronizeDevice(const DeviceIndex /*device_index*/) const {
  //TODO: implement? sync point for sim
}

// streams
// stubs for now

Stream SimGuard::getStream(Device device) const {
  return Stream(Stream::DEFAULT, device);
}

Stream SimGuard::getDefaultStream(Device device) const {
  return Stream(Stream::DEFAULT, device);
}

Stream SimGuard::getStreamFromGlobalPool(Device device, bool /*isHighPriority*/) const {
  return Stream(Stream::DEFAULT, device);
}

Stream SimGuard::getNewStream(Device device, int /*priority*/) const {
  return Stream(Stream::DEFAULT, device);
}

Stream SimGuard::exchangeStream(Stream s) const {
  // TODO
  return s;
}

void* SimGuard::getStreamNativeHandle(const Stream /*stream*/) const {
  return nullptr;
}

bool SimGuard::queryStream(const Stream& /*stream*/) const {
  return true;
}

void SimGuard::synchronizeStream(const Stream& /*stream*/) const {
  // TODO: implement
}

// events

void SimGuard::record(
    void** /*event*/,
    const Stream& /*stream*/,
    const DeviceIndex /*device_index*/,
    const c10::EventFlag /*flag*/) const {
  // TODO: record on the sim stream timeline
}

void SimGuard::block(void* /*event*/, const Stream& /*stream*/) const {
  // TODO: insert a cross-stream wait edge in the sim scheduler
}

bool SimGuard::queryEvent(void* /*event*/) const {
  return true;
}

void SimGuard::synchronizeEvent(void* /*event*/) const {
  // TODO: implement
}

double SimGuard::elapsedTime(
    void* /*event1*/,
    void* /*event2*/,
    const DeviceIndex /*device_index*/) const {
  return 0.0;
}


C10_REGISTER_GUARD_IMPL(PrivateUse1, SimGuard);

} // namespace c10::gpuclustersim