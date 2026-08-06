#include "DeviceGuard.h"
#include "DeviceFunctions.h"
#include "Streams.h"

namespace c10::gpuclustersim {

// device

DeviceIndex SimGuard::deviceCount() const noexcept {
  return device_count();
}

Device SimGuard::exchangeDevice(Device device) const {
  auto old = exchange_device(device.index());
  return Device(static_type, old);
}

Device SimGuard::getDevice() const {
  return Device(static_type, current_device());
}

void SimGuard::setDevice(Device device) const {
  set_device(device.index());
}

void SimGuard::uncheckedSetDevice(Device device) const noexcept {
  set_device(device.index());
}

DeviceCapability SimGuard::getDeviceCapability(Device /*unused*/) const {
  return DeviceCapability{};
}

void SimGuard::synchronizeDevice(const DeviceIndex /*device_index*/) const {
  //TODO: implement? sync point for sim
}

// streams

Stream SimGuard::getStream(Device device) const {
  return Stream(Stream::UNSAFE, device, getSimStream(device.index()));
}

Stream SimGuard::getDefaultStream(Device device) const {
  return Stream(Stream::UNSAFE, device, getDefaultSimStream(device.index()));
}

Stream SimGuard::getStreamFromGlobalPool(Device device, bool /*isHighPriority*/) const {
  // ignore priority just return the current stream
  return Stream(Stream::UNSAFE, device, getNewSimStream(device.index()));
}

Stream SimGuard::getNewStream(Device device, int /*priority*/) const {
  return Stream(Stream::UNSAFE, device, getNewSimStream(device.index()));
}

Stream SimGuard::exchangeStream(Stream stream) const {
  StreamId old_stream_id = exchangeSimStream(stream.device_index(), stream.id()); 
  return Stream(Stream::UNSAFE, Device(kPrivateUse1, stream.device_index()), old_stream_id);
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