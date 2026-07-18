#include <ATen/EmptyTensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/blob.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/_local_scalar_dense_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/resize_as_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/view_native.h>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function_hook.h>

#include <c10/core/Allocator.h>


namespace at::native::gpuclustersim {

at::Tensor empty_memory_format(
  c10::IntArrayRef size,
  std::optional<c10::ScalarType> dtype_opt,
  std::optional<c10::Layout> layout_opt,
  std::optional<c10::Device> device_opt,
  std::optional<bool> pin_memory_opt,
  std::optional<c10::MemoryFormat> memory_format_opt);

at::Tensor empty_strided(
  c10::IntArrayRef size,
  c10::IntArrayRef stride,
  std::optional<c10::ScalarType> dtype_opt,
  std::optional<c10::Layout> layout_opt,
  std::optional<c10::Device> device_opt,
  std::optional<bool> pin_memory_opt);

at::Tensor as_strided(
  const at::Tensor& self,
  c10::SymIntArrayRef size,
  c10::SymIntArrayRef stride,
  std::optional<c10::SymInt> storage_offset);

const at::Tensor& resize_(
  const at::Tensor& self,
  c10::SymIntArrayRef size,
  ::std::optional<at::MemoryFormat> memory_format);

at::Tensor _reshape_alias(
  const at::Tensor& self,
  c10::SymIntArrayRef size,
  c10::SymIntArrayRef stride);

at::Tensor _copy_from(
  const at::Tensor& self,
  const at::Tensor& dst,
  bool non_blocking);

at::Tensor _copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst);

at::Scalar _local_scalar_dense(const at::Tensor& self);

at::Tensor& set_source_Tensor_(at::Tensor& self, const at::Tensor& source);

at::Tensor& set_source_Storage_(at::Tensor& self, at::Storage source);

at::Tensor& set_source_Storage_storage_offset_(
  at::Tensor& result,
  at::Storage storage,
  int64_t storage_offset,
  c10::IntArrayRef size,
  c10::IntArrayRef stride);

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size);

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

} // namespace at::native::gpuclustersim
