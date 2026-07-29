#pragma once

namespace gpuclustersim {

int device_count() noexcept;

int get_current_device();

void set_current_device(int device);

int exchange_device(int device);

}