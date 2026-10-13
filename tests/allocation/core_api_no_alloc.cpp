#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>

#include "rabitqlib/core/api.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"

#ifdef _MSC_VER
#include <malloc.h>
#define TEST_NOINLINE __declspec(noinline)
#else
#define TEST_NOINLINE __attribute__((noinline))
#endif

namespace {
bool watching = false;
size_t new_calls = 0;
size_t c_calls = 0;
volatile float sink = 0;
struct Watch {
    Watch() { watching = true; }
    ~Watch() { watching = false; }
};
}  // namespace

#ifdef RABITQ_WRAP_MALLOC
extern "C" {
void* __real_malloc(size_t);
void* __real_calloc(size_t, size_t);
void* __real_realloc(void*, size_t);
void* __real_aligned_alloc(size_t, size_t);
int __real_posix_memalign(void**, size_t, size_t);
void* __wrap_malloc(size_t n) {
    if (watching)
        ++c_calls;
    return __real_malloc(n);
}
void* __wrap_calloc(size_t n, size_t width) {
    if (watching)
        ++c_calls;
    return __real_calloc(n, width);
}
void* __wrap_realloc(void* p, size_t n) {
    if (watching)
        ++c_calls;
    return __real_realloc(p, n);
}
void* __wrap_aligned_alloc(size_t alignment, size_t n) {
    if (watching)
        ++c_calls;
    return __real_aligned_alloc(alignment, n);
}
int __wrap_posix_memalign(void** p, size_t alignment, size_t n) {
    if (watching)
        ++c_calls;
    return __real_posix_memalign(p, alignment, n);
}
}
#endif

TEST_NOINLINE void* operator new(size_t n) {
    if (watching)
        ++new_calls;
    if (void* p = std::malloc(n == 0 ? 1 : n))
        return p;
    throw std::bad_alloc();
}
TEST_NOINLINE void* operator new[](size_t n) { return ::operator new(n); }
TEST_NOINLINE void operator delete(void* p) noexcept { std::free(p); }
TEST_NOINLINE void operator delete[](void* p) noexcept { ::operator delete(p); }
TEST_NOINLINE void operator delete(void* p, size_t) noexcept { ::operator delete(p); }
TEST_NOINLINE void operator delete[](void* p, size_t) noexcept { ::operator delete(p); }
TEST_NOINLINE void* operator new(size_t n, std::align_val_t alignment) {
    if (watching)
        ++new_calls;
    const auto a = static_cast<size_t>(alignment);
#ifdef _MSC_VER
    if (void* p = _aligned_malloc(n == 0 ? 1 : n, a))
        return p;
#else
    if (void* p = std::aligned_alloc(a, ((n == 0 ? 1 : n) + a - 1) / a * a))
        return p;
#endif
    throw std::bad_alloc();
}
TEST_NOINLINE void* operator new[](size_t n, std::align_val_t a) {
    return ::operator new(n, a);
}
TEST_NOINLINE void operator delete(void* p, std::align_val_t) noexcept {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    std::free(p);
#endif
}
TEST_NOINLINE void operator delete[](void* p, std::align_val_t a) noexcept {
    ::operator delete(p, a);
}
TEST_NOINLINE void operator delete(void* p, size_t, std::align_val_t a) noexcept {
    ::operator delete(p, a);
}
TEST_NOINLINE void operator delete[](void* p, size_t, std::align_val_t a) noexcept {
    ::operator delete(p, a);
}

#ifdef RABITQ_FORCE_SCALAR
// Isolated executable: force the actual cached dispatch to use scalar kernels.
// This guarantees the exact event-heap fallback is exercised, even on AVX-512 hosts.
namespace rabitqlib::cpu {
bool has_avx2() { return false; }
bool has_avx512_core() { return false; }
bool has_avx512_popcnt() { return false; }
bool has_neon() { return false; }
}  // namespace rabitqlib::cpu
#endif

int main() {
    namespace api = rabitqlib::quant::v1;
    constexpr size_t max_dim = 4096;
    alignas(std::max_align_t) std::array<std::byte, max_dim * 64 + 1024> memory{};
    std::array<float, max_dim> data{}, centroid{}, query{}, rotated{};
    std::array<uint8_t, max_dim * 2> raw{}, restored{};
    std::array<uint8_t, max_dim / 8> signs{};
    std::array<uint8_t, max_dim> extra{};
    for (size_t i = 0; i < max_dim; ++i) {
        data[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 11;
        centroid[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 13;
        query[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 17;
    }
    try {
        for (size_t dim : {32U, 96U, 512U, 4096U}) {
            for (size_t bits : {1U, 4U, 9U}) {
                for (auto metric : {rabitqlib::METRIC_L2, rabitqlib::METRIC_IP}) {
                    for (double scale : {-1.0, 7.25}) {
                        const api::Parameters p{dim, bits, metric, scale};
                        api::Requirements r;
                        {
                            Watch guard;
                            r = api::requirements(p, api::CodeLayout::SplitPacked);
                        }
                        if (r.workspace_bytes > memory.size())
                            return 1;
                        api::Buffer workspace{memory.data(), r.workspace_bytes};
                        api::CodeOutput a{
                            api::CodeLayout::UnpackedU16, {raw.data(), raw.size()}, {}};
                        api::CodeOutput b{
                            api::CodeLayout::SplitPacked,
                            {signs.data(), signs.size()},
                            {extra.data(), extra.size()}};
                        // No warmup: the first valid call and the first call at each size
                        // count.
                        {
                            Watch guard;
                            auto q = api::prepare_query(p, query.data(), centroid.data());
                            auto f =
                                api::encode(p, data.data(), centroid.data(), a, workspace);
                            api::convert_codes(
                                p, {a.layout, {raw.data(), raw.size()}, {}}, b, workspace
                            );
                            sink = api::compute(
                                q,
                                {b.layout,
                                 {signs.data(), signs.size()},
                                 {extra.data(), extra.size()}},
                                f
                            );
                            api::convert_codes(
                                p,
                                {b.layout,
                                 {signs.data(), signs.size()},
                                 {extra.data(), extra.size()}},
                                {a.layout, {restored.data(), restored.size()}, {}},
                                workspace
                            );
                            f = api::encode(p, data.data(), nullptr, b, workspace);
                            q = api::prepare_query(p, query.data(), nullptr);
                            sink = api::compute(
                                q,
                                {b.layout,
                                 {signs.data(), signs.size()},
                                 {extra.data(), extra.size()}},
                                f
                            );
                            f = api::encode(
                                p, centroid.data(), centroid.data(), a, workspace
                            );
                            sink = api::compute(
                                q, {a.layout, {raw.data(), raw.size()}, {}}, f
                            );
                        }
                        if (!std::isfinite(sink))
                            return 2;
                    }
                }
            }
        }
        for (size_t dim : {2U, 3U, 15U, 33U, 65U}) {
            Watch guard;
            const api::Parameters p{dim, 9};
            const auto r = api::requirements(p, api::CodeLayout::UnpackedU16);
            const auto f = api::encode(
                p,
                data.data(),
                nullptr,
                {api::CodeLayout::UnpackedU16, {raw.data(), raw.size()}, {}},
                {memory.data(), r.workspace_bytes}
            );
            sink = api::compute(
                api::prepare_query(p, query.data(), nullptr),
                {api::CodeLayout::UnpackedU16, {raw.data(), raw.size()}, {}},
                f
            );
        }
        // Rotation remains optional. The existing default FHT path is allocation-free
        // after constructing its state, including its first rotate call.
        std::unique_ptr<rabitqlib::Rotator<float>> rotation(
            rabitqlib::choose_rotator<float>(
                65, rabitqlib::RotatorType::FhtKacRotator, 0, 42
            )
        );
        {
            Watch guard;
            rotation->rotate(data.data(), rotated.data());
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Unexpected error: %s\n", error.what());
        return 3;
    }
    std::printf("C++ allocation calls: %zu; C allocation calls: %zu\n", new_calls, c_calls);
    return new_calls == 0 && c_calls == 0 ? 0 : 4;
}
