#include "device.h"

namespace gcs::sim {

static int g_num_devices = 1; // default
static DeviceSpec g_device_spec;
static thread_local int g_current_device = 0;


void init_devices(DeviceSpec device_spec, int n){
  g_device_spec = device_spec;
  g_num_devices = n;
}

DeviceSpec get_device_spec(){
  return g_device_spec;
}

int device_count() {
    return g_num_devices;
}

int current_device() {
    return g_current_device;
}

void set_device(int device) {
    g_current_device = device;
}

int exchange_device(int device) {
    int old = g_current_device;
    g_current_device = device;
    return old;
}

}