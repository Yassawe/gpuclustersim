#include "native/Common.h"

namespace at::gpuclustersim {

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", at::native::gpuclustersim::empty_memory_format);
  m.impl("empty_strided", at::native::gpuclustersim::empty_strided);
  m.impl("as_strided", at::native::gpuclustersim::as_strided);
  m.impl("resize_", at::native::gpuclustersim::resize_);
  m.impl("_reshape_alias", at::native::gpuclustersim::_reshape_alias);
  m.impl("_copy_from", at::native::gpuclustersim::_copy_from);
  m.impl("_copy_from_and_resize", at::native::gpuclustersim::_copy_from_and_resize);
  m.impl("_local_scalar_dense", at::native::gpuclustersim::_local_scalar_dense);
  m.impl("_has_compatible_shallow_copy_type", at::native::gpuclustersim::_has_compatible_shallow_copy_type);
  m.impl("set_.source_Tensor", at::native::gpuclustersim::set_source_Tensor_);
  m.impl("set_.source_Storage", at::native::gpuclustersim::set_source_Storage_);
  m.impl("set_.source_Storage_storage_offset", at::native::gpuclustersim::set_source_Storage_storage_offset_);
  m.impl("view", at::native::gpuclustersim::view);
  m.impl("convolution_overrideable",
      torch::CppFunction::makeFromBoxedFunction<&at::native::gpuclustersim::op_interceptor>());
  m.impl("convolution_backward_overrideable",
      torch::CppFunction::makeFromBoxedFunction<&at::native::gpuclustersim::op_interceptor>());
}

TORCH_LIBRARY_IMPL(aten, Meta, m) {
  m.impl("convolution_overrideable",
      at::native::gpuclustersim::meta_convolution_overrideable);
  m.impl("convolution_backward_overrideable",
      at::native::gpuclustersim::meta_convolution_backward_overrideable);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&at::native::gpuclustersim::op_interceptor>());
}

} // namespace at::gpuclustersim