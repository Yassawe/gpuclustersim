#include "common.h"
#include "compute.h"
#include <vector>
#include <unordered_map>
#include <functional>


namespace gcs::sim::cost_models {

// helpers

int64_t tensor_bytes(ArgSpec& arg){
  if (arg.type == ArgSpec::Type::Tensor && arg.tensor.defined){
    return arg.tensor.numel*arg.tensor.dtype_size;
  }
  if (arg.type == ArgSpec::Type::TensorList){
    int64_t total = 0;
    for (TensorSpec t : arg.tensor_list){
      if (t.defined) total+=t.numel*t.dtype_size; 
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
  return OpCost{0, 0, DataType::FP32};
}

OpCost addmm_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost bmm_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost baddbmm_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost matmul_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost linear_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost scaled_mm_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}


// CONV

OpCost convd_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost conv_transpose_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost convolution_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost slow_conv2d_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost conv_backward_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

// ATTENTION

OpCost attention_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost attention_backward_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

// NORM

OpCost layer_norm_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost batch_norm_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost group_norm_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

OpCost instance_norm_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}

// MISC

OpCost elementwise_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}


OpCost reduction_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}


OpCost pool_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}


OpCost embedding_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}


OpCost loss_cost(OpSpec& op_spec){
  return OpCost{0, 0, DataType::FP32};
}


OpCost cost_router(OpSpec& op_spec){
  // the selection of ops to model was taken from two main sources: torch's own flop_counter (https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py)
  // and deepspeed's flop profiler (https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/profiling/flops_profiler/profiler.py) 
  // this list is broader than both, and includes membound ops and more variations of main ops
  // the actual op names and schemas are generated deterministically by torch on installation, stored in torch/include/Aten/ops (in installation folder)

  static const std::unordered_map<std::string, std::function<OpCost(OpSpec&)>> registry = {
    //gemm
    {"aten::mm", mm_cost},
    {"aten::addmm", addmm_cost},
    {"aten::bmm", bmm_cost},
    {"aten::baddbmm", baddbmm_cost},
    {"aten::_scaled_mm", scaled_mm_cost},
    {"aten::matmul", matmul_cost},
    {"aten::linear", linear_cost},

    //conv
    {"aten::conv1d", convd_cost},
    {"aten::conv2d", convd_cost},
    {"aten::conv3d", convd_cost},
    {"aten::conv_transpose1d", conv_transpose_cost},
    {"aten::conv_transpose2d", conv_transpose_cost},
    {"aten::conv_transpose3d", conv_transpose_cost},
    {"aten::convolution", convolution_cost},
    {"aten::_convolution", convolution_cost},
    {"aten::cudnn_convolution", convolution_cost},
    {"aten::convolution_overrideable", convolution_cost},
    {"aten::_slow_conv2d_forward", slow_conv2d_cost},
    {"aten::convolution_backward", conv_backward_cost},

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

    //norm
    {"aten::native_batch_norm", batch_norm_cost},
    {"aten::_native_batch_norm_legit", batch_norm_cost},
    {"aten::native_layer_norm", layer_norm_cost},
    {"aten::native_group_norm", group_norm_cost},
    {"aten::instance_norm", instance_norm_cost},

    //elementwise
    //TODO: add backward coverage
    {"aten::relu", elementwise_cost},
    {"aten::relu_", elementwise_cost},
    {"aten::gelu", elementwise_cost},
    {"aten::silu", elementwise_cost},
    {"aten::silu_", elementwise_cost},
    {"aten::softmax", elementwise_cost},
    {"aten::_softmax", elementwise_cost},
    {"aten::log_softmax", elementwise_cost},
    {"aten::_log_softmax", elementwise_cost},
    {"aten::tanh", elementwise_cost},
    {"aten::tanh_", elementwise_cost},
    {"aten::sigmoid", elementwise_cost},
    {"aten::sigmoid_", elementwise_cost},
    {"aten::add", elementwise_cost},
    {"aten::add_", elementwise_cost},
    {"aten::mul", elementwise_cost},
    {"aten::mul_", elementwise_cost},
    {"aten::div", elementwise_cost},
    {"aten::div_", elementwise_cost},
    {"aten::sub", elementwise_cost},
    {"aten::sub_", elementwise_cost},
    {"aten::pow", elementwise_cost},
    {"aten::pow_", elementwise_cost},
    {"aten::sqrt", elementwise_cost},
    {"aten::sqrt_", elementwise_cost},
    {"aten::rsqrt", elementwise_cost},
    {"aten::rsqrt_", elementwise_cost},
    {"aten::reciprocal", elementwise_cost},
    {"aten::reciprocal_", elementwise_cost},
    {"aten::neg", elementwise_cost},
    {"aten::neg_", elementwise_cost},
    {"aten::abs", elementwise_cost},
    {"aten::abs_", elementwise_cost},

    //reduction
    {"aten::sum", reduction_cost},
    {"aten::mean", reduction_cost},

    //pooling
    {"aten::max_pool2d", pool_cost},
    {"aten::max_pool2d_with_indices", pool_cost},
    {"aten::avg_pool2d", pool_cost},
    {"aten::adaptive_avg_pool2d", pool_cost},

    // misc
    {"aten::embedding", embedding_cost},
    {"aten::cross_entropy_loss", loss_cost},
    {"aten::nll_loss_forward", loss_cost},
    {"aten::nll_loss_backward", loss_cost},
    {"aten::mse_loss", loss_cost},
    {"aten::l1_loss", loss_cost},
    {"aten::smooth_l1_loss", loss_cost},
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
  return 0;
}


double estimate_compute_duration(OpSpec& op_spec, DeviceSpec& device_spec){
  OpCost op_cost = cost_router(op_spec);
  double duration = duration_fn(op_cost, device_spec);
  return duration;

}

}