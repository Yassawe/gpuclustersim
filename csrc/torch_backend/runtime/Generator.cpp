#include "Generator.h"
#include "DeviceFunctions.h"

#include <vector>

namespace c10::gpuclustersim {

static std::vector<at::Generator> generators;
  
const at::Generator& getGenerator(DeviceIndex device_id) {
  static bool flag [[maybe_unused]] = []() {
  auto device_nums = device_count();
  generators.resize(device_nums);
  for (auto i = 0; i < device_nums; i++) {
    generators[i] = at::make_generator<DummyGenerator>(i);
    generators[i].seed();
  }
  return true;
  }(); // lazy init woodo magic

  int idx = static_cast<int>((device_id == -1) ? current_device() : device_id);

  return generators[idx];
}

} // namespace c10::gpuclustersim