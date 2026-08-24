#include "DeviceGuard.h"
#include "DeviceFunctions.h"
#include "Streams.h"
#include "Events.h"

namespace c10::gpuclustersim {

// device

DeviceIndex GCSDeviceGuard::deviceCount() const noexcept {
  return gcsDeviceCount();
}

Device GCSDeviceGuard::exchangeDevice(Device device) const {
  auto old = gcsExchangeDevice(device.index());
  return Device(static_type, old);
}

Device GCSDeviceGuard::getDevice() const {
  return Device(static_type, gcsCurrentDevice());
}

void GCSDeviceGuard::setDevice(Device device) const {
  gcsSetDevice(device.index());
}

void GCSDeviceGuard::uncheckedSetDevice(Device device) const noexcept {
  gcsSetDevice(device.index());
}

DeviceCapability GCSDeviceGuard::getDeviceCapability(Device device) const {
  return DeviceCapability{};
}

void GCSDeviceGuard::synchronizeDevice(const DeviceIndex device_id) const {
  //this should make host wait until all streams on a device are finished, but i have no host_time now so no-op
}

// streams

Stream GCSDeviceGuard::getStream(Device device) const {
  return Stream(Stream::UNSAFE, device, gcsGetStream(device.index()));
}

Stream GCSDeviceGuard::getDefaultStream(Device device) const {
  return Stream(Stream::UNSAFE, device, gcsGetDefaultStream(device.index()));
}

Stream GCSDeviceGuard::getStreamFromGlobalPool(Device device, bool isHighPriority) const {
  // ignore priority just return the current stream
  return Stream(Stream::UNSAFE, device, gcsGetStream(device.index()));
}

Stream GCSDeviceGuard::getNewStream(Device device, int priority) const {
  return Stream(Stream::UNSAFE, device, gcsGetNewStream(device.index()));
}

Stream GCSDeviceGuard::exchangeStream(Stream stream) const {
  StreamId old_stream_id = gcsExchangeStream(stream.device_index(), stream.id()); 
  return Stream(Stream::UNSAFE, Device(kPrivateUse1, stream.device_index()), old_stream_id);
}

void* GCSDeviceGuard::getStreamNativeHandle(const Stream stream) const {
  return nullptr;
}

bool GCSDeviceGuard::queryStream(const Stream& stream) const {
  return true; //always ready
}

void GCSDeviceGuard::synchronizeStream(const Stream& stream) const {
  // this waits until all work on the stream is done, effectively noop in my case, should advance global time to end of stream, but it already is there regardless
}

// events

void GCSDeviceGuard::record(void** event, const Stream& stream, const DeviceIndex device_id, const c10::EventFlag flag) const {
  gcsRecordEvent(event, stream.device_index(), stream.id());
}

void GCSDeviceGuard::block(void* event, const Stream& stream) const {
  gcsBlockEvent(event, stream.device_index(), stream.id());
}

bool GCSDeviceGuard::queryEvent(void* event) const {
  return gcsQueryEvent(event);
}

void GCSDeviceGuard::synchronizeEvent(void* event) const {
  // should advance global time to the event timestamp, but it will always be satisfied so noop
}

double GCSDeviceGuard::elapsedTime(void* event1, void* event2, const DeviceIndex device_id) const {
  return gcsEventElapsedTime(event1, event2);
}

void GCSDeviceGuard::destroyEvent(void* event, const DeviceIndex device_id) const noexcept {
  gcsDestroyEvent(event);
}


C10_REGISTER_GUARD_IMPL(PrivateUse1, GCSDeviceGuard);

} // namespace c10::gpuclustersim