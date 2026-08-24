#include "Common.h"
#include "runtime/DeviceFunctions.h"
#include "runtime/Streams.h"
#include <iostream>
#include <sim_engine.h>


namespace at::native::gpuclustersim {

// helper functions may look complicated, but all they do is traverse an execution stack and do actions based on the 
// type of IValue encountered. for more info look at: Aten/core/stack.h, Aten/core/ivalue.h, Aten/core/function_schema.h in torch 

c10::Device infer_target_device(torch::jit::Stack& stack){
  // honestly i am not sure if this is even non redundant and if gcsCurrentDevice() would have worked fine. this is mostly
  // precaution, maybe wasteful. I don't know whether there is a guarantee that torch always uses guard to set_device() 
  // before any op can be dispatched.

  int num_entries = stack.size();

  for(int i=0; i<num_entries; i++){ 
    c10::IValue& iv = stack[i];
    if (iv.isDevice()){
      return iv.toDevice();
    }
  }

  for(int i=0; i<num_entries; i++){
    c10::IValue& iv = stack[i];
    if (iv.isTensor()) {
      at::Tensor& t = iv.toTensor();
      if (t.defined() && t.device().type() == c10::DeviceType::PrivateUse1) return t.device();
      
    }
    else if (iv.isTensorList()){
      const c10::List<at::Tensor>& tl = iv.toTensorList();
      int list_size = tl.size();
      for(int j = 0; j<list_size; j++){
        const at::Tensor& t = tl[j];
        if (t.defined() && t.device().type() == c10::DeviceType::PrivateUse1) return t.device();

      }
    }
  }

  // fallback to default current device getter if everything else failed
  return c10::Device(c10::DeviceType::PrivateUse1, c10::gpuclustersim::gcsCurrentDevice());
}


torch::jit::Stack cast_stack_to_device(torch::jit::Stack& stack, c10::Device device){
  int num_entries = stack.size();
  torch::jit::Stack new_stack;
  new_stack.reserve(num_entries);

  for (int i = 0; i<num_entries; i++){
    c10::IValue& iv = stack[i];

    if(iv.isTensor()) {
      at::Tensor& t = iv.toTensor();
      if (t.defined()) {
        new_stack.push_back(at::empty_strided(t.sizes(), t.strides(), t.options().device(device)));
      } 
      else {
        new_stack.push_back(t);
      }
    }

    else if (iv.isTensorList()){
      const c10::List<at::Tensor>& tl = iv.toTensorList();
      int list_size = tl.size();
      c10::List<at::Tensor> new_list;
      new_list.reserve(list_size);
      for (int j=0; j<list_size; j++){
        const at::Tensor& t = tl[j];
        if (t.defined()) {
          new_list.push_back(at::empty_strided(t.sizes(), t.strides(), t.options().device(device)));
        } 
        else {
          new_list.push_back(t);
        }
      }
      new_stack.push_back(std::move(new_list));
    }

    else if (iv.isDevice()){
      new_stack.push_back(device);
    }

    else {
      new_stack.push_back(iv);
    }
  }

  return new_stack;
}


gcs::sim::cost_models::DataType map_dtype(at::ScalarType st){
  switch (st) {
    case at::ScalarType::Double: return gcs::sim::cost_models::DataType::FP64;
    case at::ScalarType::Float: return gcs::sim::cost_models::DataType::FP32;
    case at::ScalarType::Half: return gcs::sim::cost_models::DataType::FP16;
    case at::ScalarType::BFloat16: return gcs::sim::cost_models::DataType::FP16; 
    case at::ScalarType::Float8_e5m2: return gcs::sim::cost_models::DataType::FP8; 
    case at::ScalarType::Float8_e4m3fn: return gcs::sim::cost_models::DataType::FP8;
    case at::ScalarType::Float8_e5m2fnuz: return gcs::sim::cost_models::DataType::FP8;
    case at::ScalarType::Float8_e4m3fnuz: return gcs::sim::cost_models::DataType::FP8;
    case at::ScalarType::Float8_e8m0fnu: return gcs::sim::cost_models::DataType::FP8;
                                       
    case at::ScalarType::Byte: return gcs::sim::cost_models::DataType::INT8;
    case at::ScalarType::Char: return gcs::sim::cost_models::DataType::INT8;
    case at::ScalarType::Short: return gcs::sim::cost_models::DataType::INT16;
    case at::ScalarType::Int: return gcs::sim::cost_models::DataType::INT32;
    case at::ScalarType::Long: return gcs::sim::cost_models::DataType::INT64;
    case at::ScalarType::UInt16: return gcs::sim::cost_models::DataType::INT16; //unsigned are same costwise&memorywise
    case at::ScalarType::UInt32: return gcs::sim::cost_models::DataType::INT32;
    case at::ScalarType::UInt64: return gcs::sim::cost_models::DataType::INT64;

    case at::ScalarType::Bool: return gcs::sim::cost_models::DataType::BOOL;

    default: return gcs::sim::cost_models::DataType::Other;
  }
}


std::vector<gcs::sim::cost_models::ArgSpec> capture_args(torch::jit::Stack& stack, const auto& arg_names){

  std::vector<gcs::sim::cost_models::ArgSpec> result;

  result.reserve(arg_names.size());

  for (int i=0; i<arg_names.size(); i++){
    gcs::sim::cost_models::ArgSpec arg;
    arg.name = arg_names[i].name();

    c10::IValue& iv = stack[i];

    if (iv.isTensor()){
      arg.type = gcs::sim::cost_models::ArgSpec::Type::Tensor;
      at::Tensor& t = iv.toTensor();
      gcs::sim::cost_models::TensorSpec t_spec;
      t_spec.defined = t.defined();
      if(t.defined()){
        t_spec.dims = t.sizes().vec();
        t_spec.dtype = map_dtype(t.scalar_type());
        t_spec.dtype_size = static_cast<int>(t.element_size());
      }
      arg.tensor = t_spec;
    }

    else if (iv.isTensorList()){
      arg.type = gcs::sim::cost_models::ArgSpec::Type::TensorList;
      const c10::List<at::Tensor>& tl = iv.toTensorList();
      std::vector<gcs::sim::cost_models::TensorSpec> t_spec_list;
      t_spec_list.reserve(tl.size());

      for(int j=0; j<tl.size(); j++){
        const at::Tensor& t = tl[j];
        gcs::sim::cost_models::TensorSpec t_spec;
        t_spec.defined = t.defined();
        if(t.defined()){
          t_spec.dims = t.sizes().vec();
          t_spec.dtype = map_dtype(t.scalar_type());
          t_spec.dtype_size = static_cast<int>(t.element_size());
        }
        t_spec_list.push_back(t_spec);
      }

      arg.tensor_list = t_spec_list;
    }
    else if (iv.isInt() || iv.isDouble() || iv.isBool()) {

      arg.type = gcs::sim::cost_models::ArgSpec::Type::Scalar;
      gcs::sim::cost_models::ScalarSpec s_spec;

      if (iv.isInt()){
        s_spec.dtype = gcs::sim::cost_models::DataType::INT32; 
        s_spec.value = static_cast<double>(iv.toInt()); 
      }
      else if (iv.isDouble()){
        s_spec.dtype = gcs::sim::cost_models::DataType::FP32;
        s_spec.value = iv.toDouble();
      }
      else if (iv.isBool()){
        s_spec.dtype = gcs::sim::cost_models::DataType::BOOL;
        s_spec.value = iv.toBool() ? 1.0 : 0.0;
      }

      arg.scalar = s_spec;
    }
    else if (iv.isIntList() || iv.isDoubleList() || iv.isBoolList()) {
      arg.type = gcs::sim::cost_models::ArgSpec::Type::ScalarList;
      std::vector<gcs::sim::cost_models::ScalarSpec> s_spec_list;

      if (iv.isIntList()){
        for (int v : iv.toIntList()) {
          s_spec_list.push_back({gcs::sim::cost_models::DataType::INT32, static_cast<double>(v)});
        }
      }
      else if (iv.isDoubleList()) {
        for (double v : iv.toDoubleList()){
          s_spec_list.push_back({gcs::sim::cost_models::DataType::FP32, v});
        }
      }
      else if (iv.isBoolList()) {
        for (bool v : iv.toBoolList()){
          s_spec_list.push_back({gcs::sim::cost_models::DataType::BOOL, (v ? 1.0 : 0.0)});
        }
      }

      arg.scalar_list = s_spec_list; 

    }
    else {
      arg.type = gcs::sim::cost_models::ArgSpec::Type::Other; // irrelevant for cost estimation
    }

    result.push_back(arg);
  }

  return result;

}

void op_interceptor(const c10::OperatorHandle& op, torch::jit::Stack* stack) {


  c10::Device device = infer_target_device(*stack);
  c10::DeviceIndex device_id = device.index();
  c10::StreamId stream_id = c10::gpuclustersim::gcsGetStream(device_id);

  std::vector<gcs::sim::cost_models::ArgSpec> inputs = capture_args(*stack, op.schema().arguments());

  // functional correctness part, cast to meta, redispatch to get output shapes, cast back for coninuity
  // if meta implementation is lacking for that op, fallback to cpu 
  torch::jit::Stack meta_stack = cast_stack_to_device(*stack, c10::Device(c10::kMeta));
  try {
    c10::DispatchKeySet meta_ks(c10::DispatchKey::Meta);
    c10::Dispatcher::singleton().redispatchBoxed(op, meta_ks, &meta_stack);
    *stack = cast_stack_to_device(meta_stack, device);
  }
  catch (const c10::Error& e) {
    at::native::cpu_fallback(op, stack); // icky, todo: investigate
    *stack = cast_stack_to_device(*stack, device); 
  }  

  // after dispatch the stack contains outputs
  std::vector<gcs::sim::cost_models::ArgSpec> outputs = capture_args(*stack, op.schema().returns());
  
  gcs::sim::cost_models::OpSpec op_spec = {op.schema().name(), inputs, outputs};

  // sim_engine/core/controller.cpp
  gcs::sim::submit_compute_op(static_cast<int>(device_id), static_cast<int>(stream_id), op_spec); 

}

} // namespace at::native::gpuclustersim


