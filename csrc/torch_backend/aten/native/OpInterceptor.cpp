#include "Common.h"
#include "runtime/DeviceFunctions.h"
#include "runtime/Streams.h"
#include <iostream>
#include <sim_engine.h>


namespace at::native::gpuclustersim {

c10::Device _infer_target_device(torch::jit::Stack& stack){
  // honestly i am not sure if this is even non redundant and if current_device() would have worked fine. this is mostly
  // precaution, maybe wasteful. I don't know whether there is a guarantee that torch always uses guard to set_device() 
  // before any op can be dispatched.

  int num_entries = stack.size();

  // first try to infer from explicit device member of the stack
  for(int i=0; i<num_entries; i++){ 
    c10::IValue& iv = stack[i];
    if (iv.isDevice()){
      return iv.toDevice();
    }
  }

  // then try to infer from allocated tensors
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

  // fallback to default current device getter
  return c10::Device(c10::DeviceType::PrivateUse1, c10::gpuclustersim::current_device());
}


torch::jit::Stack _cast_stack_to_device(torch::jit::Stack& stack, c10::Device device){
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


void op_interceptor(const c10::OperatorHandle& op, torch::jit::Stack* stack) {

  c10::Device device = _infer_target_device(*stack);
  c10::DeviceIndex device_id = device.index();
  c10::StreamId stream_id = c10::gpuclustersim::getSimStream(device_id);


  gcs::sim::submit_computation_op();


  torch::jit::Stack meta_stack = _cast_stack_to_device(*stack, c10::Device(c10::kMeta));

  try {
    c10::DispatchKeySet meta_ks(c10::DispatchKey::Meta);
    c10::Dispatcher::singleton().redispatchBoxed(op, meta_ks, &meta_stack);
    *stack = _cast_stack_to_device(meta_stack, device);
  }
  catch (const c10::Error& e) {
    at::native::cpu_fallback(op, stack); // icky
    *stack = _cast_stack_to_device(*stack, device); 
  }  
}

} // namespace at::native::gpuclustersim


