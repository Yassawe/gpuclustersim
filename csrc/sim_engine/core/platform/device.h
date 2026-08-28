#pragma once
#include <utils/Macros.h>

namespace gcs::sim{

struct DeviceSpec {
  double fp64_tflops;
  double fp32_tflops;
  double fp16_tflops;
  double fp8_tflops;

  double mem_size; //gb
  double mem_bandwidth; //gb/s
};


void init_devices(DeviceSpec device_spec, int n);
DeviceSpec get_device_spec(); 

ENABLE_EXPORT int device_count();
ENABLE_EXPORT int current_device();
ENABLE_EXPORT void set_device(int device);
ENABLE_EXPORT int exchange_device(int device);




}