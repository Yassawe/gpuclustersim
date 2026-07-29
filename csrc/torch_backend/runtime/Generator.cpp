#include "Generator.h"

namespace {
  constexpr int kDeviceCount = 1; //placeholder
}

std::vector<at::Generator> default_generators;

namespace c10::gpuclustersim {
  
  const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) {
    static bool initialized [[maybe_unused]] = []() {
      default_generators.resize(kDeviceCount);
      for (int i = 0; i < kDeviceCount; i++) {
        default_generators[i] = at::make_generator<DummyGenerator>(i);
        default_generators[i].seed();
      }
      return true;
    }();

    c10::DeviceIndex idx = device_index;

    if (idx == -1) {
      idx = 0; 
    }

    TORCH_CHECK(idx >= 0 && idx < kDeviceCount,
          "Invalid device index ", idx, ". Expected 0 to ", kDeviceCount - 1);

    return default_generators[idx];
  }

} // namespace c10::gpuclustersim