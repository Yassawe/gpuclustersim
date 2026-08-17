#include <utils/Macros.h>

namespace gcs::sim{

ENABLE_EXPORT int device_count();
ENABLE_EXPORT int current_device();
ENABLE_EXPORT void set_device(int device);
ENABLE_EXPORT int exchange_device(int device);

struct DeviceProps {
  double peak_tflops;
  double mem_bw_gbps;
  double kernel_overhead_us;
};


}