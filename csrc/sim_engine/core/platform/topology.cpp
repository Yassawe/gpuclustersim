#include "topology.h"

namespace gcs::sim{

// global rank in the cluster. Physical rank, not per process group rank (TP/DP/MP), i.e. the actual global index of a physical device on a physical cluster, the ultimate source of truth used to label traces at the end
static int g_rank = 0; 

void set_rank(int r){
  g_rank = r;
}

int get_rank(){
  return g_rank;
}

}