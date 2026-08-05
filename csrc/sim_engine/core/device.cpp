#include "sim_engine.h"

namespace gcs::sim {

static int g_num_devices = 8;
static thread_local int g_current_device = 0;


ENABLE_EXPORT int device_count() {
    return g_num_devices;
}

ENABLE_EXPORT int current_device() {
    return g_current_device;
}

ENABLE_EXPORT void set_device(int device) {
    g_current_device = device;
}

ENABLE_EXPORT int exchange_device(int device) {
    int old = g_current_device;
    g_current_device = device;
    return old;
}

}