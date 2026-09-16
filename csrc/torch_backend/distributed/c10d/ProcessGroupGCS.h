#pragma once
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <string>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <sim_engine.h>
#include <utils/Macros.h>


namespace c10d::gpuclustersim{

const std::string BACKEND_NAME = "simccl";

// a simple factory function, because exposing class constructor to pybind directly looks ugly and boilerplaity, while function is one line
ENABLE_EXPORT c10::intrusive_ptr<Backend> create_simccl_backend(int rank, int size, std::vector<int64_t>& global_ranks_in_group, c10::intrusive_ptr<Store>& store);

class GCSWork : public Work {
  public:
    GCSWork(c10::DeviceIndex device_id, c10::StreamId comm_stream, double end_time, std::vector<at::Tensor>& tensors);
    virtual ~GCSWork();
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout) override;
    void synchronize() override;
    void blockCurrentStream() override;
    void abort() override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

  private:
    c10::intrusive_ptr<c10::ivalue::Future> future_;
    std::vector<at::Tensor> outputs_;
    c10::DeviceIndex device_id_;
    c10::StreamId comm_stream_;
    double end_time_;
};

class ProcessGroupGCS : public Backend{
  public:
    struct Options : public Backend::Options {
      explicit Options() : Backend::Options(BACKEND_NAME) {}
    };

    explicit ProcessGroupGCS(int rank, int size, std::vector<int64_t>& global_ranks_in_group, c10::intrusive_ptr<Store>& store);
    virtual ~ProcessGroupGCS();
    
    const std::string getBackendName() const override {
      return BACKEND_NAME;
    }

    bool supportsSplitting() const override {
      return false;
    }

    c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
      return c10::static_intrusive_pointer_cast<Backend::Options>(options_);
    }

    c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

    c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

    c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

    c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
        AllreduceCoalescedOptions()) override;

    c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

    c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      const AllgatherOptions& opts = AllgatherOptions()) override;

    c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

    c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& output_lists,
      std::vector<at::Tensor>& input_list,
      const AllgatherOptions& opts = AllgatherOptions()) override;

    c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

    c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const GatherOptions& opts = GatherOptions()) override;

    c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ScatterOptions& opts = ScatterOptions()) override;
    
    c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

    c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

    c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

    c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputCounts,
      std::vector<int64_t>& inputCounts,
      const AllToAllOptions& opts = AllToAllOptions()) override;

    c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

    c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

    c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

    c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

    c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  private:
    c10::intrusive_ptr<Options> options_; // nahuy nado?
    c10::intrusive_ptr<Store> store_; // for IPC, in my case it is to randezvous before collective comm and agree on start time
    std::unordered_map<c10::DeviceIndex, c10::StreamId> comm_streams_; // 1 per device, each PG creates own instance of this class
    std::vector<int64_t> participants_;
    c10::StreamId create_or_get_comm_stream(c10::DeviceIndex device_id);
    double rendezvous(uint64_t seq, double ready);
    std::atomic<uint64_t> seq_{0};
    double rendezvous_p2p(int src, int dst, uint64_t seq_p2p, double ready);
    std::vector<std::uint64_t> send_seq_;
    std::vector<std::uint64_t> recv_seq_;
    c10::intrusive_ptr<Work> submit_comm_op_helper(
      std::string name, 
      bool asyncOp,
      std::vector<at::Tensor>& tensors, 
      int root=0, 
      std::vector<int64_t> input_counts = {}, 
      std::vector<int64_t> output_counts = {});
    c10::intrusive_ptr<Work> submit_p2p_op_helper(
      std::string name,
      std::vector<at::Tensor>& tensors,
      int src,
      int dst
    );
};

}