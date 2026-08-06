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

DeviceCapability SimGuard::getDeviceCapability(Device device) const {
  return DeviceCapability{};
}

void SimGuard::synchronizeDevice(const DeviceIndex device_id) const {
  //TODO: this should wait until all streams on a device are synced, set_stream_timelines to a max value among them
}

// streams

Stream SimGuard::getStream(Device device) const {
  return Stream(Stream::UNSAFE, device, getSimStream(device.index()));
}

Stream SimGuard::getDefaultStream(Device device) const {
  return Stream(Stream::UNSAFE, device, getDefaultSimStream(device.index()));
}

Stream SimGuard::getStreamFromGlobalPool(Device device, bool isHighPriority) const {
  // ignore priority just return the current stream
  return Stream(Stream::UNSAFE, device, getNewSimStream(device.index()));
}

Stream SimGuard::getNewStream(Device device, int priority) const {
  return Stream(Stream::UNSAFE, device, getNewSimStream(device.index()));
}

Stream SimGuard::exchangeStream(Stream stream) const {
  StreamId old_stream_id = exchangeSimStream(stream.device_index(), stream.id()); 
  return Stream(Stream::UNSAFE, Device(kPrivateUse1, stream.device_index()), old_stream_id);
}

void* SimGuard::getStreamNativeHandle(const Stream stream) const {
  return nullptr;
}

bool SimGuard::queryStream(const Stream& stream) const {
  return true;
}

void SimGuard::synchronizeStream(const Stream& stream) const {
  // TODO: this waits until all work on the stream is done, effectively noop in my case, should advance global time to end of stream, but it already is there
}

// events

void SimGuard::record(
    void** event,
    const Stream& stream,
    const DeviceIndex device_id,
    const c10::EventFlag flag) const {
  // TODO: record on the sim stream timeline
}

void SimGuard::block(void* event, const Stream& stream) const {
  // TODO: insert a cross-stream wait edge in the sim scheduler
}

bool SimGuard::queryEvent(void* event) const {
  return true;
}

void SimGuard::synchronizeEvent(void* event) const {
  // TODO: should advance global time to the event timestamp, but it will always be satisfied so noop
}

double SimGuard::elapsedTime(
    void* event1,
    void* event2,
    const DeviceIndex device_id) const {
  return 0.0;
}


C10_REGISTER_GUARD_IMPL(PrivateUse1, SimGuard);

} // namespace c10::gpuclustersim