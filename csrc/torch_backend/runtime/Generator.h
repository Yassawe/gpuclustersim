#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <c10/core/Device.h>

namespace c10::gpuclustersim {
  
  class DummyGenerator : public at::CPUGeneratorImpl {
    public:
        DummyGenerator(c10::DeviceIndex device_index) {
            device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
            key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
        }
        ~DummyGenerator() override = default;
    };
  
  const at::Generator& getGenerator(c10::DeviceIndex device_index);

}