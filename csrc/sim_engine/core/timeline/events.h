#pragma once
#include <utils/Macros.h>

namespace gcs::sim{

ENABLE_EXPORT void event_record(void** event, int device, int stream);
ENABLE_EXPORT void event_block(void* event, int device, int stream);
ENABLE_EXPORT bool event_query(void* event);
ENABLE_EXPORT double event_elapsed_time(void* event1, void* event2);
ENABLE_EXPORT void event_destroy(void* event);

}