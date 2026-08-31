#pragma once
#include <utils/Macros.h>

namespace gcs::sim{

struct ENABLE_EXPORT DeviceSpec {
  double fp64_tflops;
  double fp32_tflops;
  double fp16_tflops;
  double fp8_tflops;

  double mem_size; //gb
  double mem_bandwidth; //gb/s
};


ENABLE_EXPORT void init_devices(DeviceSpec device_spec, int n);
ENABLE_EXPORT DeviceSpec get_device_spec(); 

ENABLE_EXPORT int device_count();
ENABLE_EXPORT int current_device();
ENABLE_EXPORT void set_device(int device);
ENABLE_EXPORT int exchange_device(int device);




}