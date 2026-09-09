#include "ProcessGroupGCS.h"
#include "runtime/Streams.h"
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

ProcessGroupGCS::ProcessGroupGCS(int rank, int size) : Backend(rank, size), options_(Options::create()) {}

ProcessGroupGCS::~ProcessGroupGCS() = default;


void ProcessGroupGCS::submit_comm_op_helper(std::string name, std::vector<at::Tensor>& tensors, int root=0, int peer=-1, std::vector<int64_t> input_counts = {}, std::vector<int64_t> output_counts = {}){

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
      t_spec.dtype = map_dtype(t.scalar_type());
      t_spec.dtype_size = static_cast<int>(t.element_size());
      t_spec.numel = t.numel();
    }
    payload.push_back(t_spec);
  }

  gcs::sim::cost_models::CommSpec comm_spec = CommSpec{name, rank, world_size, payload, root, peer, input_counts, output_counts};
  int stream = static_cast<int>(c10::gpuclustersim::gcsGetStream(static_cast<c10::DeviceIndex>(rank)));
  gcs::sim::submit_communication_op(rank, stream, comm_spec);

}

c10::intrusive_ptr<Work> ProcessGroupGCS::broadcast(std::vector<at::Tensor>& tensors, const BroadcastOptions& opts = BroadcastOptions()) {
  submit_comm_op_helper("broadcast", tensors, opts.rootRank);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allreduce(std::vector<at::Tensor>& tensors, const AllreduceOptions& opts = AllreduceOptions()){
  submit_comm_op_helper("allreduce", tensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allreduce_sparse(std::vector<at::Tensor>& tensors, const AllreduceOptions& opts = AllreduceOptions()){
  submit_comm_op_helper("allreduce_sparse", tensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allreduce_coalesced(std::vector<at::Tensor>& tensors, const AllreduceCoalescedOptions& opts = AllreduceCoalescedOptions()) {
  submit_comm_op_helper("allreduce_coalesced", tensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> reduce(std::vector<at::Tensor>& tensors, const ReduceOptions& opts = ReduceOptions()){
  submit_comm_op_helper("broadcast", tensors, opts.rootRank);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> _reduce_scatter_base(at::Tensor& outputTensor, at::Tensor& inputTensor, const ReduceScatterOptions& opts = ReduceScatterOptions()){
  std::vector<at::Tensor> tensors = {inputTensor};
  submit_comm_op_helper("reduce_scatter", tensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> _allgather_base(at::Tensor& output_tensor, at::Tensor& input_tensor, const AllgatherOptions& opts = AllgatherOptions()){
  std::vector<at::Tensor> tensors = {input_tensor};
  submit_comm_op_helper("allgather", tensors);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> allgather(std::vector<std::vector<at::Tensor>>& outputs, std::vector<at::Tensor>& inputs, const AllgatherOptions& opts = AllgatherOptions()){
  submit_comm_op_helper("allgather", inputs);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> allgather_coalesced(std::vector<std::vector<at::Tensor>>& output_lists, std::vector<at::Tensor>& input_list, const AllgatherOptions& opts = AllgatherOptions()) {
  submit_comm_op_helper("allgather_coalesced", input_list);
  return c10::make_intrusive<DummyWork>();
}

c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs, const AllgatherOptions& opts = AllgatherOptions()) 


}