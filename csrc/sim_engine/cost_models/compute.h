namespace gcs::sim::cost_models{

enum class OpFamily {
  Unknown, GEMM, Conv, Elementwise, Reduction, Memory, Embedding, Attention
};

struct TensorSpec {
  std::vector<int64_t> sizes; 
  int dtype_size;
};

struct OpSpec {
  std::string name;
  OpFamily family;
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
};


}

