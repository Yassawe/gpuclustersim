#include <utils/Macros.h>

namespace gcs::sim{

ENABLE_EXPORT int create_stream(int device);
ENABLE_EXPORT int default_stream(int device);
ENABLE_EXPORT int current_stream(int device);
ENABLE_EXPORT int exchange_stream(int device, int stream_id);

}