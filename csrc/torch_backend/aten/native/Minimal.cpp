#include "Common.h"

namespace at::native::gpuclustersim {

// creation ops
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
  const c10::DeviceGuard device_guard(device);
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
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_strided_generic(
  size, stride, allocator, pu1_dks, dtype);
}

// view ops

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

// needed by FSDP 
void record_stream(at::Tensor& self, c10::Stream s) {
  (void) self;
  (void) s;
}

// copy ops

// noops, fake transfer, TODO: record datasize to simulate memcpy time between devices

at::Tensor _copy_from(
  const at::Tensor& self,
  const at::Tensor& dst,
  bool non_blocking) {
  TORCH_CHECK(self.defined(), "Source tensor (self) is not defined.");
  TORCH_CHECK(dst.defined(), "Destination tensor (dst) is not defined.");

  return dst;
}

at::Tensor _copy_from_and_resize(
  const at::Tensor& self,
  const at::Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);

  return dst;
}

at::Scalar _local_scalar_dense(const at::Tensor& self) {
  // returning dummy zero, explicitly gonna break data dependent paths
  return at::Scalar(0.0);
}

bool _has_compatible_shallow_copy_type(const at::Tensor& self, const at::Tensor& from) {
  return true;
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
  return at::cpu::set_(result, storage, storage_offset, size, stride); // TODO: this is a sus operation can cause segfaults on some weird cases, maybe just ignore it and do set_?
}

static std::vector<int64_t> conv_out_spatial(
  c10::IntArrayRef in,
  c10::IntArrayRef kernel,
  c10::IntArrayRef stride,
  c10::IntArrayRef padding,
  c10::IntArrayRef dilation,
  bool transposed,
  c10::IntArrayRef output_padding) {
  std::vector<int64_t> out(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    if (transposed) {
      out[i] = (in[i] - 1) * stride[i] - 2 * padding[i] + dilation[i] * (kernel[i] - 1) + output_padding[i] + 1;
    } else {
      out[i] = (in[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1)/stride[i] + 1;
    }
  }
  return out;
}

at::Tensor meta_convolution_overrideable(
  const at::Tensor& input,
  const at::Tensor& weight,
  const std::optional<at::Tensor>& bias,
  c10::SymIntArrayRef stride,
  c10::SymIntArrayRef padding,
  c10::SymIntArrayRef dilation,
  bool transposed,
  c10::SymIntArrayRef output_padding,
  c10::SymInt groups) {
  const int64_t group_size = groups.expect_int();
  std::vector<int64_t> sizes(2 + input.dim() - 2);
  sizes[0] = input.size(0);
  sizes[1] = transposed ? weight.size(1) * group_size : weight.size(0);
  std::vector<int64_t> stride_v, padding_v, dilation_v, output_padding_v;
  for (const auto& s : stride) stride_v.push_back(s.expect_int());
  for (const auto& p : padding) padding_v.push_back(p.expect_int());
  for (const auto& d : dilation) dilation_v.push_back(d.expect_int());
  for (const auto& p : output_padding) output_padding_v.push_back(p.expect_int());
  auto spatial = conv_out_spatial(
    input.sizes().slice(2), weight.sizes().slice(2), stride_v, padding_v,
    dilation_v, transposed, output_padding_v);
  for (size_t i = 0; i < spatial.size(); ++i) sizes[2 + i] = spatial[i];
  return at::empty(sizes, input.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> meta_convolution_backward_overrideable(
  const at::Tensor& grad_output,
  const at::Tensor& input,
  const at::Tensor& weight,
  c10::SymIntArrayRef stride,
  c10::SymIntArrayRef padding,
  c10::SymIntArrayRef dilation,
  bool transposed,
  c10::SymIntArrayRef output_padding,
  c10::SymInt groups,
  ::std::array<bool, 3> output_mask) {
  (void)grad_output;
  (void)stride;
  (void)padding;
  (void)dilation;
  (void)output_padding;
  (void)output_mask;
  const int64_t group_size = groups.expect_int();
  int64_t cout = transposed ? weight.size(1) * group_size : weight.size(0);
  at::Tensor grad_input = at::empty(input.sizes(), input.options());
  at::Tensor grad_weight = at::empty(weight.sizes(), weight.options());
  at::Tensor grad_bias = at::empty({cout}, weight.options());
  return {grad_input, grad_weight, grad_bias};
}

} // namespace at::native::gpuclustersim