#pragma once
#include <utils/Macros.h>

namespace gcs::sim{

void submit_compute_op();
void submit_communication_op();
void submit_memory_event();
void sumbit_memcpy_event();


}