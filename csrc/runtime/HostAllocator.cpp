#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>

namespace c10::gpuclustersim {

struct DummyHostAllocator : at::HostAllocator {
    DummyHostAllocator() = default;

    static void ReportAndDelete(void* ptr) {
        if (ptr) {
            free(ptr);
        }
    }

    at::DataPtr allocate(size_t nbytes) override {
        void* data = nullptr;
        if (nbytes > 0) {
            data = malloc(nbytes); // change here
        }
        return {data, data, &ReportAndDelete, at::Device(at::kCPU)};
    }

    at::DeleterFnPtr raw_deleter() const override {
        return &ReportAndDelete;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const final {
        memcpy(dest, src, count);
    }

    // nothing happens
    bool record_event(void* ptr, void* ctx, c10::Stream stream) override {
        return true;
    }

    void empty_cache() override {}

    at::HostStats get_stats() override {
        return at::HostStats();
    }

    void reset_accumulated_stats() override {}

    void reset_peak_stats() override {}
};

DummyHostAllocator g_host_allocator;

REGISTER_HOST_ALLOCATOR(at::kPrivateUse1, &g_host_allocator);

} // namespace c10::gpuclustersim