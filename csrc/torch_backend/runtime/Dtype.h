#include <c10/core/ScalarType.h>
#include <sim_engine.h>

namespace c10::gpuclustersim {

gcs::sim::cost_models::DataType map_dtype(at::ScalarType st);

}
