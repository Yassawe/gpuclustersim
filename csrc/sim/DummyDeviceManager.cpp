#include "DummyDeviceManager.h"

namespace gpuclustersim {

static int num_devices = 8;
static thread_local int current_device = 0;


int device_count() noexcept {
    return num_devices;
}

int get_current_device() {
    return current_device;
}

void set_current_device(int device) {
    current_device = device;
}

int exchange_device(int device) {
    int old = current_device;
    current_device = device;
    return old;
}

}