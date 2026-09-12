#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/Context.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <runtime/DeviceFunctions.h>
#include <distributed/c10d/ProcessGroupGCS.h>
#include <sim_engine.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {

  m.def("_init", []() {
    torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
    at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);
  });

  m.def("_create_simccl_backend", &c10d::gpuclustersim::create_simccl_backend);

  m.def("_get_device_count", []() {
    torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
    return static_cast<int>(c10::gpuclustersim::gcsDeviceCount());
  });

  m.def("_get_device", []() {
    torch::utils::device_lazy_init(at::kPrivateUse1);
    return static_cast<int>(c10::gpuclustersim::gcsCurrentDevice());
  });

  m.def("_set_device", [](int device) {
    torch::utils::device_lazy_init(at::kPrivateUse1);
    c10::gpuclustersim::gcsSetDevice(device);
  });

  m.def("_exchangeDevice", [](int device_index) -> int32_t {
    if (device_index < 0) return -1;
    torch::utils::device_lazy_init(at::kPrivateUse1);
    return static_cast<int32_t>(c10::gpuclustersim::gcsExchangeDevice(device_index));
  });

  m.def("_init_device_group", [](py::dict spec, int n) {
    gcs::sim::DeviceSpec device_spec{};
    device_spec.fp64_tflops = spec["fp64_tflops"].cast<double>();
    device_spec.fp32_tflops = spec["fp32_tflops"].cast<double>();
    device_spec.fp16_tflops = spec["fp16_tflops"].cast<double>();
    device_spec.fp8_tflops = spec["fp8_tflops"].cast<double>();
    device_spec.mem_size = spec["mem_size"].cast<double>();
    device_spec.mem_bandwidth = spec["mem_bandwidth"].cast<double>();
    gcs::sim::init_device_group(device_spec, n);
  });

   m.def("_get_timeline", []() {
    auto timeline = gcs::sim::get_timeline();
    py::list py_timeline;
    for (const auto& d : timeline) {
      py::list streams;
      for (const auto& s : d) {
        py::dict stream;
        stream["current_time"] = s.current_time;
        py::list ops;
        for (const auto& op : s.ops) {
          py::dict o;
          o["name"] = op.name;
          o["start_time"] = op.start_time;
          o["end_time"] = op.end_time;
          ops.append(o);
        }
        stream["ops"] = ops;
        streams.append(stream);
      }
      py_timeline.append(streams);
    }
    return py_timeline;
  });

  m.def("_reset_timeline", []() {
  gcs::sim::reset_timeline();
  });

}