#pragma once
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <c10/core/Device.h>

namespace c10::gpuclustersim {
  
class GCSGenerator : public at::CPUGeneratorImpl {
  public:
    GCSGenerator(DeviceIndex device_id) {
      device_ = Device(c10::DeviceType::PrivateUse1, device_id);
      key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
    }
    ~GCSGenerator() override = default;
  };

const at::Generator& getGenerator(DeviceIndex device_id);

}