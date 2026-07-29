#include "Common.h"

#include <iostream>

namespace at::native::gpuclustersim {

void op_interceptor(const c10::OperatorHandle& op, torch::jit::Stack* stack) {

  std::cout << "[Simulator] Intercepted: " << op.schema().name() << std::endl;
  
  const auto num_inputs = stack->size();
  torch::jit::Stack meta_stack;
  meta_stack.reserve(num_inputs);
  
  // ─── Step 1: Convert stack to meta ───

  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& iv = (*stack)[i];
    
    if (iv.isTensor()) {
      const auto& t = iv.toTensor();
      if (t.defined()) {
        auto meta = at::empty(t.sizes(), t.options().device(c10::kMeta));
        meta_stack.push_back(std::move(meta));
      } else {
        meta_stack.push_back(t);
      }
    }
    else if (iv.isTensorList()) {
      const auto& list = iv.toTensorList();
      c10::List<at::Tensor> meta_list;
      meta_list.reserve(list.size());
      for (const auto& ref : list) {
        const at::Tensor& t = ref;
        if (t.defined()) {
          meta_list.push_back(at::empty(t.sizes(), t.options().device(c10::kMeta)));
        } else {
          meta_list.push_back(t);
        }
      }
      meta_stack.push_back(std::move(meta_list));
    }
    else if (iv.isDevice()) {
      meta_stack.push_back(c10::Device(c10::kMeta));
    }
    else {
      meta_stack.push_back(iv);
    }
  }
  
  // ─── Step 2: Explicitly dispatch to Meta kernel ───
  try {
    c10::DispatchKeySet meta_ks(c10::DispatchKey::Meta);
    c10::Dispatcher::singleton().redispatchBoxed(op, meta_ks, &meta_stack);
  }
  catch (const c10::Error& e) {
    std::cerr << "[Simulator] Meta error for " << op.schema().name() 
        << ": " << e.what() << std::endl;
    // Fall back to CPU
    at::native::cpu_fallback(op, stack);
    return;
  }
  
  // ─── Step 3: Convert meta outputs back to PrivateUse1 ───
  stack->clear();
  for (size_t i = 0; i < meta_stack.size(); ++i) {
    const auto& iv = meta_stack[i];
    
    if (iv.isTensor()) {
      const auto& meta_t = iv.toTensor();
      if (meta_t.defined()) {
        auto pu_t = at::empty_strided(
          meta_t.sizes(),
          meta_t.strides(),
          meta_t.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0))
        );
        stack->push_back(std::move(pu_t));
      } else {
        stack->push_back(meta_t);
      }
    }
    else if (iv.isTensorList()) {
      const auto& meta_list = iv.toTensorList();
      c10::List<at::Tensor> pu_list;
      pu_list.reserve(meta_list.size());
      for (const auto& ref : meta_list) {
        const at::Tensor& meta_t = ref;
        if (meta_t.defined()) {
          pu_list.push_back(at::empty_strided(
            meta_t.sizes(),
            meta_t.strides(),
            meta_t.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0))
          ));
        } else {
          pu_list.push_back(meta_t);
        }
      }
      stack->push_back(std::move(pu_list));
    }
    else {
      stack->push_back(iv);
    }
  }
}

} // namespace at::native::gpuclustersim


