#include "ProcessGroupGCS.h"
#include "runtime/DeviceFunctions.h"
#include "runtime/Streams.h"
#include "runtime/Events.h"
#include "runtime/Dtype.h"
#include <sim_engine.h>
#include <mutex>
#include <numeric>
#include <algorithm>


namespace c10d::gpuclustersim{

static std::mutex mutex_; 

c10::intrusive_ptr<Backend> create_simccl_backend(int rank, int size, std::vector<int64_t> global_ranks_in_group, c10::intrusive_ptr<Store> store){
  return c10::make_intrusive<ProcessGroupGCS>(rank, size, std::move(global_ranks_in_group), std::move(store));
}

// GCSWork 
GCSWork::GCSWork(c10::DeviceIndex device_id, c10::StreamId comm_stream, double end_time) : Work(-1, OpType::UNKNOWN), future_(c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get())) {

  device_id_ = device_id;
  comm_stream_ = comm_stream;
  end_time_ = end_time;

  if (!future_->completed()) {
    future_->markCompleted(c10::IValue());
  }
  finish();
}

GCSWork::~GCSWork() = default;

bool GCSWork::isCompleted() {
  double current_time = gcs::sim::get_current_stream_time(static_cast<int>(device_id_), static_cast<int>(comm_stream_));
  return current_time>=end_time_; 
}

bool GCSWork::isSuccess() const {
  return !future_->hasError();
}

bool GCSWork::wait(std::chrono::milliseconds timeout) {
  synchronize();
  return true;
}

void GCSWork::synchronize() {
  c10::StreamId caller_stream = c10::gpuclustersim::gcsGetStream(device_id_);
  gcs::sim::advance_current_stream_time(static_cast<int>(device_id_), static_cast<int>(caller), end_time_);
}

void GCSWork::blockCurrentStream(){
  synchronize();
}

void GCSWork::abort() {}

c10::intrusive_ptr<c10::ivalue::Future> GCSWork::getFuture() {
  return future_;
}

// ProcessGroupGCS

ProcessGroupGCS::ProcessGroupGCS(int rank, int size, std::vector<int64_t> global_ranks_in_group, c10::intrusive_ptr<Store> store) : Backend(rank, size) {
  if (global_ranks_in_group.empty()) {
    participants_.resize(std::max(0, size));
    std::iota(participants_.begin(), participants_.end(), 0);
  } else {
    participants_ = global_ranks_in_group;
  }
  options_ = c10::make_intrusive<Options>();
  store_ = std::move(store); 
}

ProcessGroupGCS::~ProcessGroupGCS() = default;


c10::StreamId ProcessGroupGCS::create_or_get_comm_stream(c10::DeviceIndex device_id){
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = comm_streams_.find(device_id);
  if (it!=comm_streams_.end()) return it->second;

  c10::StreamId comm_stream = c10::gpuclustersim::gcsGetNewStream(device_id);
  comm_streams_[device_id] = comm_stream;
  return comm_stream;
}

void sync_streams(c10::DeviceIndex device_id, c10::StreamId compute_stream, c10::StreamId comm_stream){
  // makes communication stream wait until compute stream current end. Prevents "going back in time" bug on simulated timeline
  // в случае если комм стрим заканчивает работу раньше чем комп стрим перед тем как вызвать еще один коллектив колл, 
  // в реальной жизни он бы вызвал его в настоящем времени, т.е. сейчас, но тк стримы разные если сделать по тупому без синха, он аппендится в комм стрим на точку раньше, т.е. в прошлое, что естественно не правильно
  // релевантно только когда asyncOp=true, потому что иначе ивенты и стримы хендлятся апстрим кодом (FSDP и тд, не доходят до сюда крч)
  void* ev = nullptr;
  c10::gpuclustersim::gcsRecordEvent(&ev, device_id, compute_stream);
  c10::gpuclustersim::gcsBlockEvent(ev, device_id, comm_stream); 
  c10::gpuclustersim::gcsDestroyEvent(ev); 
}


double ProcessGroupGCS::rendezvous(uint64_t seq, double ready) {
  const std::string base = BACKEND_NAME + "/" + std::to_string(seq) + "/";
  store_->set(base + std::to_string(getRank()), std::to_string(ready));

  std::vector<std::string> keys;
  for (int r = 0; r < getSize(); r++) keys.push_back(base + std::to_string(r));

  store_->wait(keys);

  double T = 0;
  for (const auto& k : keys) T = std::max(T, std::stod(store_->get_to_str(k)));
  return T;
}


c10::intrusive_ptr<Work> ProcessGroupGCS::submit_comm_op_helper(std::string name, bool asyncOp, std::vector<at::Tensor>& tensors, int root, int peer, std::vector<int64_t> input_counts, std::vector<int64_t> output_counts){

  int rank = getRank();
  int world_size = getSize();

  c10::DeviceIndex device_id = c10::gpuclustersim::gcsCurrentDevice();
  c10::StreamId caller_stream = c10::gpuclustersim::gcsGetStream(device_id); // compute stream if asyncOp, and comm stream if not

  c10::StreamId current_comm_stream;

  if (asyncOp){
    current_comm_stream = create_or_get_comm_stream(device_id);
    sync_streams(device_id, caller_stream, current_comm_stream);
  }
  else {
    current_comm_stream = caller_stream;
  }


  // ensuring every rank agrees when to start the collective
  double ready = gcs::sim::get_current_stream_time(static_cast<int>(device_id), static_cast<int>(current_comm_stream));
  uint64_t seq = seq_.fetch_add(1); 
  double T_max = rendezvous(seq, ready);
  gcs::sim::advance_current_stream_time(static_cast<int>(device_id), static_cast<int>(current_comm_stream), T_max);

  // payload
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

  gcs::sim::cost_models::CommSpec comm_spec{
    name,
    rank,
    world_size,
    payload,
    participants_, // class member, created at construction of process group
    root,
    peer,
    input_counts,
    output_counts
  };

  gcs::sim::submit_communication_op(static_cast<int>(device_id), static_cast<int>(current_comm_stream), comm_spec);

  double end = c10::gpuclustersim::get_current_stream_time(static_cast<int>(device_id), static_cast<int>(current_comm_stream));
  return c10::make_intrusive<GCSWork>(device, current_comm_stream, end);
}

c10::intrusive_ptr<Work> ProcessGroupGCS::broadcast(std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  return submit_comm_op_helper("broadcast", opts.asyncOp, tensors, opts.rootRank);
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allreduce(std::vector<at::Tensor>& tensors, const AllreduceOptions& opts){
  submit_comm_op_helper("allreduce", tensors);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allreduce_sparse(std::vector<at::Tensor>& tensors, const AllreduceOptions& opts){
  submit_comm_op_helper("allreduce_sparse", tensors);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allreduce_coalesced(std::vector<at::Tensor>& tensors, const AllreduceCoalescedOptions& opts) {
  submit_comm_op_helper("allreduce_coalesced", tensors);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::reduce(std::vector<at::Tensor>& tensors, const ReduceOptions& opts){
  submit_comm_op_helper("reduce", tensors, opts.rootRank);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::_allgather_base(at::Tensor& output_tensor, at::Tensor& input_tensor, const AllgatherOptions& opts){
  std::vector<at::Tensor> tensors = {input_tensor};
  submit_comm_op_helper("allgather_base", tensors);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allgather(std::vector<std::vector<at::Tensor>>& outputs, std::vector<at::Tensor>& inputs, const AllgatherOptions& opts){
  submit_comm_op_helper("allgather", inputs);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allgather_coalesced(std::vector<std::vector<at::Tensor>>& output_lists, std::vector<at::Tensor>& input_list, const AllgatherOptions& opts) {
  submit_comm_op_helper("allgather_coalesced", input_list);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::allgather_into_tensor_coalesced(std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs, const AllgatherOptions& opts) {
  submit_comm_op_helper("allgather_into_tensor_coalesced", inputs);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::gather(std::vector<std::vector<at::Tensor>>& outputs, std::vector<at::Tensor>& inputs, const GatherOptions& opts){
  submit_comm_op_helper("gather", inputs, opts.rootRank);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::scatter(std::vector<at::Tensor>& outputs, std::vector<std::vector<at::Tensor>>& inputs, const ScatterOptions& opts){
  submit_comm_op_helper("scatter", outputs, opts.rootRank);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::_reduce_scatter_base(at::Tensor& outputTensor, at::Tensor& inputTensor, const ReduceScatterOptions& opts){
  std::vector<at::Tensor> tensors = {inputTensor};
  submit_comm_op_helper("reduce_scatter_base", tensors);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::reduce_scatter(std::vector<at::Tensor>& outputs, std::vector<std::vector<at::Tensor>>& inputs, const ReduceScatterOptions& opts){
  std::vector<at::Tensor> flattened;
  for(auto& rank_tensors : inputs){
    flattened.insert(flattened.end(), rank_tensors.begin(), rank_tensors.end());
  }
  submit_comm_op_helper("reduce_scatter", flattened);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::reduce_scatter_tensor_coalesced(std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors, const ReduceScatterOptions& opts){
  submit_comm_op_helper("reduce_scatter_tensor_coalesced", inputTensors);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::alltoall_base(at::Tensor& outputTensor, at::Tensor& inputTensor, std::vector<int64_t>& outputCounts, std::vector<int64_t>& inputCounts, const AllToAllOptions& opts){
  std::vector<at::Tensor> tensors = {inputTensor};
  submit_comm_op_helper("alltoall_base", tensors, 0, -1, inputCounts, outputCounts);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::alltoall(std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors, const AllToAllOptions& opts){
  submit_comm_op_helper("alltoall", inputTensors);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::send(std::vector<at::Tensor>& tensors, int dstRank, int tag){
  submit_comm_op_helper("send", tensors, 0, dstRank);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::recv(std::vector<at::Tensor>& tensors, int srcRank, int tag){
  submit_comm_op_helper("recv", tensors, 0, srcRank);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::recvAnysource(std::vector<at::Tensor>& tensors, int tag){
  submit_comm_op_helper("recvAnysource", tensors);
  return c10::make_intrusive<GCSWork>();
}

c10::intrusive_ptr<Work> ProcessGroupGCS::barrier(const BarrierOptions& opts){
  std::vector<at::Tensor> empty;
  submit_comm_op_helper("barrier", empty);
  return c10::make_intrusive<GCSWork>();
}

}