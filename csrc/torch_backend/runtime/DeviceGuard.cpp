#include "DeviceGuard.h"
#include "DeviceFunctions.h"

namespace c10::gpuclustersim {

DeviceIndex SimGuard::deviceCount() const noexcept{
  return device_count();
}

Device SimGuard::exchangeDevice(Device d) const{
  
}


C10_REGISTER_GUARD_IMPL(PrivateUse1, SimGuard);

} // namespace c10::gpuclustersim