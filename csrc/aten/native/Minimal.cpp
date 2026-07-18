#include "Minimal.h"

#include <unordered_set>

namespace at::native::gpuclustersim {


// CREATION OPS

at::Tensor empty_memory_format(
  c10::IntArrayRef size,
  std::optional<c10::ScalarType> dtype_opt,
  std::optional<c10::Layout> layout_opt,
  std::optional<c10::Device> device_opt,
  std::optional<bool> pin_memory_opt,
  std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
    c10::layout_or_default(layout_opt) == c10::Layout::Strided,
    "Non strided layout not supported");
  TORCH_CHECK(
    !c10::pinned_memory_or_default(pin_memory_opt),
    "Pin memory can only be on CPU");
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_generic(
    size, allocator, pu1_dks, dtype, memory_format_opt);
}

at::Tensor empty_strided(
  c10::IntArrayRef size,
  c10::IntArrayRef stride,
  std::optional<c10::ScalarType> dtype_opt,
  std::optional<c10::Layout> layout_opt,
  std::optional<c10::Device> device_opt,
  std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
    c10::layout_or_default(layout_opt) == c10::Layout::Strided,
    "Non strided layout not supported");
  TORCH_CHECK(
    !c10::pinned_memory_or_default(pin_memory_opt),
    "Pin memory can only be on CPU");
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_strided_generic(
    size, stride, allocator, pu1_dks, dtype);
}


// VIEW OPS

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

at::Tensor as_strided(
  const at::Tensor& self,
  c10::SymIntArrayRef size,
  c10::SymIntArrayRef stride,
  std::optional<c10::SymInt> storage_offset) {
  return at::cpu::as_strided_symint(self, size, stride, storage_offset);
}

const at::Tensor& resize_(
  const at::Tensor& self,
  c10::SymIntArrayRef size,
  ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize_(
    self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
}

at::Tensor _reshape_alias(
  const at::Tensor& self,
  c10::SymIntArrayRef size,
  c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(
    self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}


// COPY OPS

// noops, fake transfer, nothing happens

at::Tensor _copy_from(
  const at::Tensor& self,
  const at::Tensor& dst,
  bool non_blocking) {
  TORCH_CHECK(self.defined(), "Source tensor (self) is not defined.");
  TORCH_CHECK(dst.defined(), "Destination tensor (dst) is not defined.");

  
  // TODO: call cost_model to analyze memcpy in simulation

  return dst;
}

at::Tensor _copy_from_and_resize(
  const at::Tensor& self,
  const at::Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);

  // TODO: call cost_model here

  return dst;
}

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}


at::Scalar _local_scalar_dense(const at::Tensor& self) {
  // returning dummy scalar
  return at::Scalar(0.0);
}


at::Tensor& set_source_Tensor_(at::Tensor& self, const at::Tensor& source) {
  return at::native::set_tensor_(self, source);
}

at::Tensor& set_source_Storage_(at::Tensor& self, at::Storage source) {
  return at::native::set_(self, source);
}

at::Tensor& set_source_Storage_storage_offset_(
  at::Tensor& result,
  at::Storage storage,
  int64_t storage_offset,
  c10::IntArrayRef size,
  c10::IntArrayRef stride) {
  return at::cpu::set_(result, storage, storage_offset, size, stride);
}


// FALLBACK

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();

  // 1. Pop original inputs from the stack
  auto inputs = torch::jit::pop(*stack, schema.arguments().size());

  // 2. Create META versions of inputs (shape-only, no data)
  std::vector<c10::IValue> meta_inputs;
  for (auto& iv : inputs) {
  if (iv.isTensor()) {
    auto t = iv.toTensor();
    // Create a meta tensor with SAME shape but NO storage
    auto meta_t = at::empty(t.sizes(), t.options().device(at::kMeta));
    meta_inputs.push_back(meta_t);
  } else {
    meta_inputs.push_back(iv);  // Non-tensor args pass through
  }
  }

  // 3. Run the op on META tensors
  // This calls PyTorch's built-in meta kernel for this op
  // Meta kernels only compute output shapes, never touch data
  torch::jit::push(*stack, meta_inputs);
  op.redispatch(c10::DispatchKey::Meta, stack);
  auto meta_results = torch::jit::pop(*stack, schema.returns().size());

  // 4. Create PrivateUse1 outputs with meta-inferred shapes
  std::vector<c10::IValue> results;
  for (auto& iv : meta_results) {
  if (iv.isTensor()) {
    auto meta_t = iv.toTensor();
    // Create your device tensor with correct shape, 1-byte storage
    auto out = at::empty(meta_t.sizes(), 
      meta_t.options().device(at::kPrivateUse1));
    results.push_back(out);
  } else {
    results.push_back(iv);  // Non-tensor returns pass through
  }
  }

  // TODO: call cost_model here with opname and shapes

  std::cout << "[GCS] Fallback op: " << op.schema().name() 
      << " | shapes: ";
  for (auto& iv : inputs) {
  if (iv.isTensor()) {
    auto t = iv.toTensor();
    std::cout << "[" << t.sizes() << "] ";
  }
  }
  std::cout << std::endl;

  torch::jit::push(*stack, results);
}


}
