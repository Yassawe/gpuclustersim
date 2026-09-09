#include "ProcessGroupGCS.h"
#include "runtime/Streams.h"
#include "runtime/Dtype.h"
#include <sim_engine.h>

namespace c10d::gpuclustersim{

DummyWork::DummyWork() : Work(-1, OpType::UNKNOWN), future_(c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get())) {
  if (!future_->completed()) {
    future_->markCompleted(c10::IValue());
  }
  finish();
}

DummyWork::~DummyWork() = default;

bool DummyWork::isCompleted() {
  return future_->completed();
}

bool DummyWork::isSuccess() const {
  return !future_->hasError();
}

bool DummyWork::wait(std::chrono::milliseconds timeout) {
  return true;
}

void DummyWork::synchronize() {}

void DummyWork::abort() {}

c10::intrusive_ptr<c10::ivalue::Future> DummyWork::getFuture() {
  return future_;
}

ProcessGroupGCS::ProcessGroupGCS(int rank, int size) : Backend(rank, size), options_(c10::make_intrusive<Options>()) {}

ProcessGroupGCS::~ProcessGroupGCS() = default;


void ProcessGroupGCS::submit_comm_op_helper(std::string name, std::vector<at::Tensor>& tensors, int root, int peer, std::vector<int64_t> input_counts, std::vector<int64_t> output_counts){

  int rank = getRank();
  int world_size = getSize();

  std::vector<gcs::sim::cost_models::TensorSpec> payload;
  payload.reserve(tensors.size());
  for(int i=0; i<tensors.size(); i++){
    at::Tensor& t = tensors[i];
    gcs::sim::cost_models::TensorSpec t_spec;
    t_spec.defined = t.defined();
    if(t.defined()){
      t_spec.dims = t.sizes().vec();
      t_spec.dtype = c10::gpuclustersim::map_dtype(t.scalar_type());
      t_spec.dtype_size = static_cast<int>(t.element_size());
      t_spec.numel = t.numel();
    }
    payload.push_back(t_spec);
  }

  gcs::sim::cost_models::CommSpec comm_spec = {name, rank, world_size, payload, root, peer, input_counts, output_counts};
  int stream = static_cast<int>(c10::gpuclustersim::gcsGetStream(static_cast<c10::DeviceIndex>(rank)));
  gcs::sim::submit_communication_op(rank, stream, comm_spec);

}

c10::intrusive_ptr<Work> ProcessGroupGCS::broadcast(std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  submit_comm_op_helper("broadcast", tensors, opts.rootRank);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allreduce(std::vector<at::Tensor>& tensors, const AllreduceOptions& opts){
  submit_comm_op_helper("allreduce", tensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allreduce_sparse(std::vector<at::Tensor>& tensors, const AllreduceOptions& opts){
  submit_comm_op_helper("allreduce_sparse", tensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allreduce_coalesced(std::vector<at::Tensor>& tensors, const AllreduceCoalescedOptions& opts) {
  submit_comm_op_helper("allreduce_coalesced", tensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::reduce(std::vector<at::Tensor>& tensors, const ReduceOptions& opts){
  submit_comm_op_helper("reduce", tensors, opts.rootRank);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::_allgather_base(at::Tensor& output_tensor, at::Tensor& input_tensor, const AllgatherOptions& opts){
  std::vector<at::Tensor> tensors = {input_tensor};
  submit_comm_op_helper("allgather_base", tensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allgather(std::vector<std::vector<at::Tensor>>& outputs, std::vector<at::Tensor>& inputs, const AllgatherOptions& opts){
  submit_comm_op_helper("allgather", inputs);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allgather_coalesced(std::vector<std::vector<at::Tensor>>& output_lists, std::vector<at::Tensor>& input_list, const AllgatherOptions& opts) {
  submit_comm_op_helper("allgather_coalesced", input_list);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allgather_into_tensor_coalesced(std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs, const AllgatherOptions& opts) {
  submit_comm_op_helper("allgather_into_tensor_coalesced", inputs);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::gather(std::vector<std::vector<at::Tensor>>& outputs, std::vector<at::Tensor>& inputs, const GatherOptions& opts){
  submit_comm_op_helper("gather", inputs, opts.rootRank);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::scatter(std::vector<at::Tensor>& outputs, std::vector<std::vector<at::Tensor>>& inputs, const ScatterOptions& opts){
  submit_comm_op_helper("scatter", outputs, opts.rootRank);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::_reduce_scatter_base(at::Tensor& outputTensor, at::Tensor& inputTensor, const ReduceScatterOptions& opts){
  std::vector<at::Tensor> tensors = {inputTensor};
  submit_comm_op_helper("reduce_scatter_base", tensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::reduce_scatter(std::vector<at::Tensor>& outputs, std::vector<std::vector<at::Tensor>>& inputs, const ReduceScatterOptions& opts){
  std::vector<at::Tensor> flattened;
  for(auto& rank_tensors : inputs){
    flattened.insert(flattened.end(), rank_tensors.begin(), rank_tensors.end());
  }
  submit_comm_op_helper("reduce_scatter", flattened);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::reduce_scatter_tensor_coalesced(std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors, const ReduceScatterOptions& opts){
  submit_comm_op_helper("reduce_scatter_tensor_coalesced", inputTensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::alltoall_base(at::Tensor& outputTensor, at::Tensor& inputTensor, std::vector<int64_t>& outputCounts, std::vector<int64_t>& inputCounts, const AllToAllOptions& opts){
  std::vector<at::Tensor> tensors = {inputTensor};
  submit_comm_op_helper("alltoall_base", tensors, 0, -1, inputCounts, outputCounts);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::alltoall(std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors, const AllToAllOptions& opts){
  submit_comm_op_helper("alltoall", inputTensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::send(std::vector<at::Tensor>& tensors, int dstRank, int tag){
  submit_comm_op_helper("send", tensors, 0, dstRank);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::recv(std::vector<at::Tensor>& tensors, int srcRank, int tag){
  submit_comm_op_helper("recv", tensors, 0, srcRank);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::recvAnysource(std::vector<at::Tensor>& tensors, int tag){
  submit_comm_op_helper("recvAnysource", tensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::barrier(const BarrierOptions& opts){
  std::vector<at::Tensor> empty;
  submit_comm_op_helper("barrier", empty);
  return c10::make_intrusive<DummyWork>();
}

}