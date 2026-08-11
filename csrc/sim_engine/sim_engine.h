#include <utils/Macros.h>
#include <vector>

namespace gcs::sim {

// devices

ENABLE_EXPORT int device_count();

ENABLE_EXPORT int current_device();

ENABLE_EXPORT void set_device(int device);

ENABLE_EXPORT int exchange_device(int device);


// memory

struct ENABLE_EXPORT MemStats {
  int current_allocated = 0;
  int peak_allocated = 0;
  int n_allocations = 0;
  int n_deallocations = 0;
};

// torch streams

ENABLE_EXPORT int create_stream(int device);
ENABLE_EXPORT int default_stream(int device);
ENABLE_EXPORT int current_stream(int device);
ENABLE_EXPORT int exchange_stream(int device, int stream_id);


// torch events

ENABLE_EXPORT void event_record(void** event, int device, int stream);
ENABLE_EXPORT void event_block(void* event, int device, int stream);
ENABLE_EXPORT bool event_query(void* event);
ENABLE_EXPORT double event_elapsed_time(void* event1, void* event2);
ENABLE_EXPORT void event_destroy(void* event);



}