#include "Minimal.h"

#include <ATen/native/CPUFallback.h>
#include <unordered_set>
#include <iostream>  

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

at::Scalar _local_scalar_dense(const at::Tensor& self) {
  // returning dummy scalar, explicitly gonna break data dependent paths
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

void meta_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // vibecoded, check for correctness later

  std::cout << "[GCS] Fallback: " << op.schema().name() << std::endl;
    
    // Just use CPU fallback - it works, it's correct, it's slow
  at::native::cpu_fallback(op, stack);
}

}
