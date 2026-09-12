#include "device.h"

namespace gcs::sim {

static int g_num_devices = 1; // devices in group, local max 128
static DeviceSpec g_device_spec; // homogenious, all devices are assumbed to be the same
static thread_local int g_current_device = 0; // local device in-group index


void init_device_group(DeviceSpec device_spec, int G){
  g_device_spec = device_spec;
  g_num_devices = G;
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