#include "Generator.h"
#include "DeviceFunctions.h"

#include <vector>

namespace c10::gpuclustersim {

static std::vector<at::Generator> default_generators;
  
const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) {
  static bool flag [[maybe_unused]] = []() {
  auto device_nums = device_count();
  default_generators.resize(device_nums);
  for (auto i = 0; i < device_nums; i++) {
    default_generators[i] = at::make_generator<DummyGenerator>(i);
    default_generators[i].seed();
  }
  return true;
  }(); // lazy init woodo magic

  DeviceIndex idx = device_index;
  
  if (idx == -1) {
    idx = current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < device_count());
  }
  return default_generators[idx];
}

} // namespace c10::gpuclustersim