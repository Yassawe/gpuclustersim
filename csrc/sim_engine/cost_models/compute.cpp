#include "common.h"
#include "compute.h"
#include <vector>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <algorithm>


namespace gcs::sim::cost_models {

// helpers

int64_t tensor_bytes(ArgSpec& arg){
  if (arg.type == ArgSpec::Type::Tensor){
    if (arg.tensor.defined && !arg.tensor.is_input_write_buffer) return arg.tensor.numel*arg.tensor.dtype_size;
  }
  if (arg.type == ArgSpec::Type::TensorList){
    int64_t total = 0;
    for (TensorSpec t : arg.tensor_list){
      if (t.defined && !t.is_input_write_buffer) total+=t.numel*t.dtype_size;
    }
    return total;
  }
  return 0;
}

int64_t total_tensor_bytes(std::vector<ArgSpec>& args){
  int64_t total = 0;
  for (ArgSpec arg : args){
    total+=tensor_bytes(arg);
  }
  return total;
}

DataType get_dominant_dtype(std::vector<ArgSpec>& args){
  // in general an op will have a single tensor dtype that is used for all tensors, so just checking the first one would have been fine, this is precaution.
  // there are some mixed precision ops where that might not be the case, checking for the largest tensor and adopting its dtype is good enough heuristic to determine which dtype the op is dispatched in.

  int64_t largest_numel = 0;
  DataType dominant = DataType::FP32;

  for (ArgSpec arg : args){
    if (arg.type==ArgSpec::Type::Tensor && arg.tensor.defined){
      if(arg.tensor.numel>largest_numel){
        dominant = arg.tensor.dtype;
        largest_numel = arg.tensor.numel;
      } 
    }
    if (arg.type==ArgSpec::Type::TensorList){
      for (TensorSpec t : arg.tensor_list){
        if(t.defined && t.numel>largest_numel){
          dominant = t.dtype;
          largest_numel = t.numel;
        }
      }
    }
  }

  return dominant;
}

// cost functions 

// GEMM

OpCost mm_cost(OpSpec& op_spec){
  // (Tensor self, Tensor mat2) -> Tensor
  // self:(M,K), mat2:(K,N) -> out:(M,N)

  ArgSpec self = op_spec.inputs[0];
  ArgSpec mat2 = op_spec.inputs[1];
  ArgSpec out = op_spec.outputs[0];

  int64_t M = self.tensor.dims[0];
  int64_t K = self.tensor.dims[1];
  int64_t N = mat2.tensor.dims[1];

  int64_t flops = 2*M*N*K;
  int64_t bytes = tensor_bytes(self) + tensor_bytes(mat2) + tensor_bytes(out);

  return OpCost{flops, bytes, get_dominant_dtype(op_spec.inputs)};
}

OpCost addmm_cost(OpSpec& op_spec){
  // (Tensor self, Tensor mat1, Tensor mat2, Scalar beta, Scalar alpha) -> Tensor
  // self:(M,N), mat1:(M,K), mat2:(K,N) -> out:(M,N)

  ArgSpec self = op_spec.inputs[0];
  ArgSpec mat1 = op_spec.inputs[1];
  ArgSpec mat2 = op_spec.inputs[2];
  ArgSpec out = op_spec.outputs[0];

  int64_t M = mat1.tensor.dims[0];
  int64_t K = mat1.tensor.dims[1];
  int64_t N = self.tensor.dims[0];

  int64_t flops = 2*M*N*K+3*M*N;
  int64_t bytes = tensor_bytes(self) + tensor_bytes(mat1) + tensor_bytes(mat2) + tensor_bytes(out);


  return OpCost{flops, bytes, get_dominant_dtype(op_spec.inputs)};
}

OpCost bmm_cost(OpSpec& op_spec){
  // (Tensor self, Tensor mat2) -> Tensor
  // self:(B,M,K), mat2:(B,K,N) -> out:(B,M,N)

  ArgSpec self = op_spec.inputs[0];
  ArgSpec mat2 = op_spec.inputs[1];
  ArgSpec out = op_spec.outputs[0];

  int64_t B = self.tensor.dims[0];
  int64_t M = self.tensor.dims[1];
  int64_t K = self.tensor.dims[2];
  int64_t N = mat2.tensor.dims[2];

  int64_t flops = 2*B*M*N*K;
  int64_t bytes = tensor_bytes(self) + tensor_bytes(mat2) + tensor_bytes(out);

  return OpCost{flops, bytes, get_dominant_dtype(op_spec.inputs)};
}

OpCost baddbmm_cost(OpSpec& op_spec){
  // (Tensor self, Tensor batch1, Tensor batch2) -> Tensor
  // self:(B,M,N), batch1:(B,M,K), batch2:(B,K,N) -> out:(B,M,N)

  ArgSpec self = op_spec.inputs[0];
  ArgSpec batch1 = op_spec.inputs[1];
  ArgSpec batch2 = op_spec.inputs[2];
  ArgSpec out = op_spec.outputs[0];

  int64_t B = self.tensor.dims[0];
  int64_t M = self.tensor.dims[1];
  int64_t N = self.tensor.dims[2];
  int64_t K = batch1.tensor.dims[2];

  int64_t flops = 2*B*M*N*K + 3*B*M*N;
  int64_t bytes = tensor_bytes(self)+tensor_bytes(batch1)+tensor_bytes(batch2)+tensor_bytes(out);

  return OpCost{flops, bytes, get_dominant_dtype(op_spec.inputs)};
}


// CONV

OpCost conv_cost(OpSpec& op_spec){
  // (Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
  // input: (N, C_in, ..., *S_in), weight: (C_out, C_in/groups, ..., *K) -> out: (N, C_out, ..., *S_out)

  ArgSpec input = op_spec.inputs[0];
  ArgSpec weight = op_spec.inputs[1];
  ArgSpec out = op_spec.outputs[0];

  int64_t N = input.tensor.dims[0];
  int64_t C = weight.tensor.dims[0]*weight.tensor.dims[1]; // C_out*C_in/groups

  int64_t K = 1;
  for (int i=2; i<weight.tensor.dims.size(); i++) K*=weight.tensor.dims[i];

  int64_t S_out = 1;
  for (int i=2; i<out.tensor.dims.size(); i++) S_out*=out.tensor.dims[i];

  int64_t flops = 2*N*C*K*S_out;
  int64_t bytes = tensor_bytes(input) + tensor_bytes(weight) + tensor_bytes(out);

  return OpCost{flops, bytes, get_dominant_dtype(op_spec.inputs)};
}

OpCost conv_backward_cost(OpSpec& op_spec){
  // (Tensor grad_output, Tensor input, Tensor weight, int[]? bias_sizes, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
  // grad_output: (N, C_out, *S_g), input: (N, C_in, *S_in), weight: (C_out, C_in/groups, *K) -> 
  // dX: (N, C_in, *S_in), dW: (C_out, C_in/groups, *K), db: (C_out)

  ArgSpec grad_output = op_spec.inputs[0];
  ArgSpec input = op_spec.inputs[1];
  ArgSpec weight = op_spec.inputs[2];
  ArgSpec mask = op_spec.inputs.back();

  ArgSpec dX = op_spec.outputs[0];
  ArgSpec dW = op_spec.outputs[1];
  ArgSpec db = op_spec.outputs[2];

  int64_t N = input.tensor.dims[0];
  int64_t C = weight.tensor.dims[0]*weight.tensor.dims[1];
  int64_t C_out = grad_output.tensor.dims[1];

  int64_t S_g = 1;
  for (int i=2; i<grad_output.tensor.dims.size(); i++) S_g*=grad_output.tensor.dims[i];

  int64_t S_in = 1;
  for (int i=2; i<input.tensor.dims.size(); i++) S_in*=input.tensor.dims[i];

  int64_t K = 1;
  for (int i=2; i<weight.tensor.dims.size(); i++) K*=weight.tensor.dims[i];

  int64_t m0, m1, m2;
  m0 = static_cast<int>(mask.scalar_list[0].value); // if dX is computed
  m1 = static_cast<int>(mask.scalar_list[1].value); // if dW is computed
  m2 = static_cast<int>(mask.scalar_list[2].value); // if db is computed
  
  int64_t flops = 2*N*C*K*(S_in*m0 + S_g*m1) + N*C_out*S_g*m2;
  int64_t bytes = tensor_bytes(grad_output) + tensor_bytes(input) + tensor_bytes(weight) + m0*tensor_bytes(dX) + m1*tensor_bytes(dW) + m2*tensor_bytes(db);

  return OpCost{flops, bytes, get_dominant_dtype(op_spec.inputs)};
}

// ATTENTION

OpCost attention_cost(OpSpec& op_spec){
  // (Tensor query, Tensor key, Tensor value, ... params) -> (bunch of outputs, dominated by 1 big tensor)
  // query, key, value: (B, H, S_q/k/v, D) -> out: (B, H, S_q, D)
  ArgSpec Q = op_spec.inputs[0];
  ArgSpec K = op_spec.inputs[1];
  ArgSpec V = op_spec.inputs[2];
  ArgSpec out = op_spec.outputs[0];

  int64_t B = Q.tensor.dims[0];
  int64_t H = Q.tensor.dims[1];
  int64_t S_q = Q.tensor.dims[2];
  int64_t S_k = K.tensor.dims[2];
  int64_t S_v = V.tensor.dims[2];
  int64_t D = Q.tensor.dims[3];

  int64_t flops = 4*B*H*S_q*S_k*D;
  int64_t bytes = tensor_bytes(Q) + tensor_bytes(K) + tensor_bytes(V) + tensor_bytes(out);


  return OpCost{flops, bytes, get_dominant_dtype(op_spec.inputs)};
}

OpCost attention_backward_cost(OpSpec& op_spec){
  // (Tensor grad_out, Tensor query, Tensor key, Tensor value, ... params) -> (bunch of outputs, 3 grads)
  // grad_output: (B, H, S_q, D), query, key, value: (B, H, S_q/k/v, D) -> dQ, dK, dV: (B, H, S_q/k/v, D)

  ArgSpec grad_out = op_spec.inputs[0];
  ArgSpec Q = op_spec.inputs[1];
  ArgSpec K = op_spec.inputs[2];
  ArgSpec V = op_spec.inputs[3];

  int64_t B  = Q.tensor.dims[0];
  int64_t H  = Q.tensor.dims[1];
  int64_t S_q = Q.tensor.dims[2];
  int64_t S_k = K.tensor.dims[2];
  int64_t S_v = V.tensor.dims[2];
  int64_t D  = Q.tensor.dims[3];

  int64_t flops = 8*B*H*S_q*S_k*D;
  int64_t bytes = tensor_bytes(grad_out) + tensor_bytes(Q) + tensor_bytes(K) + tensor_bytes(V) + total_tensor_bytes(op_spec.outputs);

  return OpCost{flops, bytes, get_dominant_dtype(op_spec.inputs)};
}


// memory bound ops

OpCost elementwise_cost(OpSpec& op_spec){
  // (Tensor self) -> Tensor
  
  ArgSpec self = op_spec.inputs[0];
  ArgSpec out = op_spec.outputs[0];
  
  int64_t flops = out.tensor.numel; // todo: revise, 1 pass is not the case for a lot of ops, also can read multiple times
  int64_t bytes = total_tensor_bytes(op_spec.inputs) + total_tensor_bytes(op_spec.outputs);

  return OpCost{flops, bytes, get_dominant_dtype(op_spec.inputs)};
}


OpCost reduction_cost(OpSpec& op_spec){
  // (Tensor self, ...) -> Tensor, like elementwise except the input<->output can be different shapes

  ArgSpec self = op_spec.inputs[0];
  ArgSpec out = op_spec.outputs[0];

  int64_t flops = std::max(self.tensor.numel, out.tensor.numel);
  int64_t bytes = total_tensor_bytes(op_spec.inputs) + total_tensor_bytes(op_spec.outputs);

  return OpCost{flops, bytes, get_dominant_dtype(op_spec.inputs)};
}

OpCost sort_cost(OpSpec& op_spec){
  int64_t n = op_spec.outputs[0].tensor.numel;
  int64_t flops = n * static_cast<int64_t>(std::log2(static_cast<double>(n))); // should be nlogn in general
  int64_t bytes = total_tensor_bytes(op_spec.inputs) + total_tensor_bytes(op_spec.outputs);

  return OpCost{flops, bytes, get_dominant_dtype(op_spec.inputs)};
}


OpCost cost_router(OpSpec& op_spec){
  // the selection of ops to model was taken from two main sources: torch's own flop_counter (https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py)
  // and deepspeed's flop profiler (https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/profiling/flops_profiler/profiler.py) 
  // this list is broader than both, and includes membound ops and more variations of main ops, as well as backward leaf level ops (the others just profile the forward, then multiply by 2 to get backward)
  // the actual op names and schemas are generated deterministically by torch on installation, stored in torch/include/Aten/ops (in installation folder)

  static const std::unordered_map<std::string, std::function<OpCost(OpSpec&)>> registry = {
    //gemm
    {"aten::mm", mm_cost},
    {"aten::addmm", addmm_cost},
    {"aten::bmm", bmm_cost},
    {"aten::baddbmm", baddbmm_cost},

    //conv
    {"aten::conv1d", conv_cost},
    {"aten::conv2d", conv_cost},
    {"aten::conv3d", conv_cost},
    {"aten::conv_transpose1d", conv_cost},
    {"aten::conv_transpose2d", conv_cost},
    {"aten::conv_transpose3d", conv_cost},
    {"aten::convolution", conv_cost},
    {"aten::_convolution", conv_cost},
    {"aten::convolution_backward", conv_backward_cost},
    {"aten::convolution_overrideable", conv_cost},
    {"aten::convolution_backward_overrideable", conv_backward_cost},

    //attention
    {"aten::_scaled_dot_product_efficient_attention", attention_cost},
    {"aten::_scaled_dot_product_flash_attention", attention_cost},
    {"aten::_scaled_dot_product_cudnn_attention", attention_cost},
    {"aten::_flash_attention_forward", attention_cost},
    {"aten::_efficient_attention_forward", attention_cost},
    {"aten::_scaled_dot_product_efficient_attention_backward", attention_backward_cost},
    {"aten::_scaled_dot_product_flash_attention_backward", attention_backward_cost},
    {"aten::_scaled_dot_product_cudnn_attention_backward", attention_backward_cost},
    {"aten::_flash_attention_backward", attention_backward_cost},
    {"aten::_efficient_attention_backward", attention_backward_cost},

    //elementwise
    {"aten::relu", elementwise_cost},
    {"aten::threshold_backward", elementwise_cost}, 
    {"aten::gelu", elementwise_cost},
    {"aten::gelu_backward", elementwise_cost}, 
    {"aten::silu", elementwise_cost},
    {"aten::_softmax", elementwise_cost},
    {"aten::_softmax_backward_data", elementwise_cost}, 
    {"aten::_log_softmax", elementwise_cost},
    {"aten::_log_softmax_backward_data", elementwise_cost}, 
    {"aten::tanh", elementwise_cost},
    {"aten::tanh_backward", elementwise_cost}, 
    {"aten::sigmoid", elementwise_cost},
    {"aten::sigmoid_backward", elementwise_cost}, 
    {"aten::add", elementwise_cost},
    {"aten::mul", elementwise_cost},
    {"aten::div", elementwise_cost},
    {"aten::sub", elementwise_cost},
    {"aten::pow", elementwise_cost},
    {"aten::sqrt", elementwise_cost},
    {"aten::rsqrt", elementwise_cost},
    {"aten::reciprocal", elementwise_cost},
    {"aten::neg", elementwise_cost},
    {"aten::abs", elementwise_cost},
    {"aten::exp", elementwise_cost},
    {"aten::log", elementwise_cost},
    {"aten::sin", elementwise_cost},
    {"aten::cos", elementwise_cost},
    {"aten::erf", elementwise_cost},
    {"aten::clamp", elementwise_cost},
    {"aten::addcmul", elementwise_cost},
    {"aten::where", elementwise_cost},
    {"aten::isneginf", elementwise_cost},
    {"aten::fill_", elementwise_cost},
    {"aten::zero_", elementwise_cost},
    {"aten::native_dropout", elementwise_cost},
    {"aten::native_dropout_backward", elementwise_cost},
    {"aten::_thnn_fused_lstm_cell_backward_impl", elementwise_cost},
    {"aten::cat", elementwise_cost},                  
    {"aten::gather", elementwise_cost},           
    {"aten::index_select", elementwise_cost},
    {"aten::_thnn_fused_lstm_cell", elementwise_cost},  
    {"aten::normal_", elementwise_cost},

    //reduction
    {"aten::sum", reduction_cost},
    {"aten::mean", reduction_cost},
    {"aten::all", reduction_cost},
    {"aten::amax", reduction_cost},
    {"aten::amin", reduction_cost},
    {"aten::argmax", reduction_cost},
    {"aten::max", reduction_cost},
    {"aten::min", reduction_cost},
    {"aten::std", reduction_cost},
    {"aten::var", reduction_cost},
    {"aten::var_mean", reduction_cost},
    {"aten::sort", sort_cost},
    {"aten::topk", sort_cost},


    //norm
    {"aten::native_batch_norm", reduction_cost},
    {"aten::native_batch_norm_backward", reduction_cost},
    {"aten::native_layer_norm", reduction_cost},
    {"aten::native_layer_norm_backward", reduction_cost},
    {"aten::native_group_norm", reduction_cost}, 
    {"aten::native_group_norm_backward", reduction_cost},

    //pooling
    {"aten::max_pool2d_with_indices", elementwise_cost},
    {"aten::max_pool2d_with_indices_backward", elementwise_cost}, 
    {"aten::avg_pool2d", elementwise_cost},
    {"aten::_adaptive_avg_pool2d", elementwise_cost}, 
    {"aten::avg_pool2d_backward", elementwise_cost}, 
    {"aten::_adaptive_avg_pool2d_backward", elementwise_cost}, 

    // misc
    {"aten::embedding", elementwise_cost},
    {"aten::embedding_dense_backward", elementwise_cost}, 
    {"aten::nll_loss_forward", elementwise_cost},
    {"aten::nll_loss_backward", elementwise_cost}, 
    {"aten::mse_loss", elementwise_cost},
    {"aten::mse_loss_backward", elementwise_cost}, 
    {"aten::smooth_l1_loss", elementwise_cost},
    {"aten::smooth_l1_loss_backward", elementwise_cost}, 
  };
  

  auto it = registry.find(op_spec.name);
  
  if (it!=registry.end()){
    return it->second(op_spec);
  }

  // for unrecognized ops, just output 0
  // good enough tradeoff, there are some esoteric ops, and it is better to ignore them, rather than to construct a
  // general flops formula (e.g. from num of elements, or bytes in input tensors), because you can have some weird rare case adding big amount of time to the timeline, thus "poisoning" the run with bs data. Safer to just ignore, as they are rare 
  return OpCost{0, 0, DataType::FP32};
}

double duration_fn(OpCost& op_cost, DeviceSpec& device_spec){
  
  double target_tflops;

  if(op_cost.dominant_dtype==DataType::FP64) target_tflops=device_spec.fp64_tflops;
  else if(op_cost.dominant_dtype==DataType::FP32) target_tflops=device_spec.fp32_tflops;
  else if(op_cost.dominant_dtype==DataType::FP16) target_tflops=device_spec.fp16_tflops;
  else if(op_cost.dominant_dtype==DataType::FP8) target_tflops=device_spec.fp8_tflops;
  else target_tflops=device_spec.fp32_tflops;

  if(target_tflops==0) target_tflops=device_spec.fp32_tflops;

  double t_comp = 1e9*static_cast<double>(op_cost.flops)/(target_tflops*1e12); // flops/(flops/s)*10^9 -> 10^9*s -> ns
  double t_mem = 1e9*static_cast<double>(op_cost.bytes)/(device_spec.mem_bandwidth*1e9);

  return std::max(t_comp, t_mem); 
}


double estimate_compute_duration(OpSpec& op_spec, DeviceSpec& device_spec){
  OpCost op_cost = cost_router(op_spec);
  double duration = duration_fn(op_cost, device_spec);
  return duration;
}

}