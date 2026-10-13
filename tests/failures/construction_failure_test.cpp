#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <omp.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/index/types.hpp"

namespace {
std::atomic<bool> fail_parallel{false};
std::atomic<size_t> remaining{0};
std::atomic<bool> fail_constructor{false};
bool tracking = false;
std::array<void*, 4096> live{};
size_t live_count = 0;

bool should_fail() {
#if _OPENMP >= 200805
    const bool in_region = omp_get_level() > 0;
#else
    const bool in_region = omp_in_parallel() != 0;
#endif
    if (fail_constructor || (fail_parallel && in_region)) {
        if (remaining.fetch_sub(1) == 0) {
            fail_constructor = false;
            fail_parallel = false;
            return true;
        }
    }
    return false;
}
void record(void* ptr) {
    if (tracking && ptr != nullptr) {
        if (live_count == live.size()) {
            std::abort();
        }
        live[live_count++] = ptr;
    }
}
void forget(void* ptr) {
    if (!tracking) {
        return;
    }
    for (size_t i = 0; i < live_count; ++i) {
        if (live[i] == ptr) {
            live[i] = live[--live_count];
            return;
        }
    }
}
}  // namespace

#if defined(RABITQ_WRAP_CONSTRUCTION_ALLOCATIONS)
extern "C" void* __real_malloc(size_t);
extern "C" void* __real_aligned_alloc(size_t, size_t);
extern "C" void __real_free(void*);
extern "C" void* __wrap_malloc(size_t size) {
    if (fail_constructor && should_fail()) {
        return nullptr;
    }
    void* ptr = __real_malloc(size);
    record(ptr);
    return ptr;
}
extern "C" void* __wrap_aligned_alloc(size_t alignment, size_t size) {
    if (fail_constructor && should_fail()) {
        return nullptr;
    }
    void* ptr = __real_aligned_alloc(alignment, size);
    record(ptr);
    return ptr;
}
extern "C" void __wrap_free(void* ptr) {
    forget(ptr);
    __real_free(ptr);
}
#endif

GTEST_NO_INLINE_ void* operator new(size_t size) {
    if (should_fail()) {
        throw std::bad_alloc();
    }
#if defined(RABITQ_WRAP_CONSTRUCTION_ALLOCATIONS)
    void* ptr = __real_malloc(size == 0 ? 1 : size);
#else
    void* ptr = std::malloc(size == 0 ? 1 : size);
#endif
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    record(ptr);
    return ptr;
}
GTEST_NO_INLINE_ void* operator new[](size_t size) { return ::operator new(size); }
GTEST_NO_INLINE_ void* operator new(size_t size, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(size);
    } catch (...) { return nullptr; }
}
GTEST_NO_INLINE_ void* operator new[](size_t size, const std::nothrow_t&) noexcept {
    try {
        return ::operator new[](size);
    } catch (...) { return nullptr; }
}
GTEST_NO_INLINE_ void operator delete(void* ptr) noexcept {
#if !defined(RABITQ_WRAP_CONSTRUCTION_ALLOCATIONS)
    forget(ptr);
#endif
    std::free(ptr);  // NOLINT(clang-analyzer-cplusplus.NewDelete)
}
GTEST_NO_INLINE_ void operator delete[](void* ptr) noexcept { ::operator delete(ptr); }
GTEST_NO_INLINE_ void operator delete(void* ptr, size_t) noexcept {
    ::operator delete(ptr);
}
GTEST_NO_INLINE_ void operator delete[](void* ptr, size_t) noexcept {
    ::operator delete(ptr);
}

namespace {
template <typename Function>
void expect_parallel_failure(Function fn, size_t skip = 0) {
    remaining = skip;
    fail_parallel = true;
    bool caught = false;
    try {
        fn();
    } catch (const std::bad_alloc&) { caught = true; }
    fail_parallel = false;
    EXPECT_TRUE(caught);
}

std::vector<float> vectors() {
    std::vector<float> data(64 * 64);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 37) / 37.0F;
    }
    return data;
}

TEST(ConstructionFailure, HnswConstructorReleasesEveryAllocation) {
#if !defined(RABITQ_WRAP_CONSTRUCTION_ALLOCATIONS)
    GTEST_SKIP() << "C allocation ownership tracking requires linker wrapping";
#else
    remaining = std::numeric_limits<size_t>::max();
    fail_constructor = true;
    { rabitqlib::hnsw::HierarchicalNSW index(32, 64, 4, 4, 16, 100); }
    fail_constructor = false;
    const size_t allocation_count = std::numeric_limits<size_t>::max() - remaining;
    RecordProperty("constructor_allocations", static_cast<int>(allocation_count));
    size_t failures = 0;
    bool completed = false;
    for (size_t allocation = 0; allocation <= allocation_count; ++allocation) {
        SCOPED_TRACE(allocation);
        remaining = allocation;
        tracking = true;
        fail_constructor = true;
        bool caught = false;
        try {
            rabitqlib::hnsw::HierarchicalNSW index(32, 64, 4, 4, 16, 100);
        } catch (const std::bad_alloc&) {
            caught = true;
        } catch (const std::runtime_error&) { caught = true; }
        fail_constructor = false;
        tracking = false;
        ASSERT_EQ(live_count, 0U);
        if (!caught) {
            completed = true;
            break;
        }
        ++failures;
    }
    EXPECT_TRUE(completed);
    EXPECT_GT(failures, 5U);
#endif
}

class ConstructionWorkerFailure : public ::testing::TestWithParam<size_t> {
    void SetUp() override {
#if _OPENMP < 200805
        if (GetParam() == 1) {
            bool detectable = false;
#pragma omp parallel num_threads(1)
            { detectable = omp_in_parallel() != 0; }
            if (!detectable) {
                GTEST_SKIP() << "This OpenMP runtime cannot identify one-thread regions";
            }
        }
#endif
        omp_set_dynamic(0);
    }
};

INSTANTIATE_TEST_SUITE_P(Threads, ConstructionWorkerFailure, ::testing::Values(1U, 4U));

TEST_P(ConstructionWorkerFailure, IvfWorkerErrorsReachCallerAndAllowRetry) {
    const auto data = vectors();
    std::vector<float> centers(2 * 64, 0.0F);
    std::vector<rabitqlib::PID> ids(64);
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = i % 2;
    }
    const size_t threads = GetParam();
    {
        for (const size_t bits : {1U, 4U, 32U}) {
            rabitqlib::ivf::IVF index(64, 64, 2, bits);
            expect_parallel_failure([&] {
                index.construct(data.data(), centers.data(), ids.data(), false, threads);
            });
            EXPECT_EQ(index.max_elements(), 64U);
            EXPECT_EQ(index.num_stored_points(), 0U);
            EXPECT_EQ(index.num_points(), 0U);
            ASSERT_NO_THROW(
                index.construct(data.data(), centers.data(), ids.data(), false, threads)
            );
            EXPECT_EQ(index.num_stored_points(), 64U);
            EXPECT_EQ(index.num_points(), 64U);
            rabitqlib::PID id = 0;
            float distance = 0;
            ASSERT_NO_THROW(index.search(data.data(), 1, 1, &id, &distance));
        }
    }
}

TEST_P(ConstructionWorkerFailure, QgInitializationWorkerErrorsReachCaller) {
    using namespace rabitqlib;
    const auto data = vectors();
    const size_t threads = GetParam();
    {
        for (const auto init :
             {symqg::QGInitialization::Random, symqg::QGInitialization::PiPNN}) {
            for (const size_t bits : {0U, 4U, 8U}) {
                symqg::QuantizedGraph<float> graph(
                    64, 64, 32, METRIC_L2, RotatorType::FhtKacRotator, bits, 42
                );
                expect_parallel_failure([&] {
                    symqg::QGBuilder builder(
                        graph, 32, data.data(), threads, init, 42, true
                    );
                });
                EXPECT_EQ(graph.max_elements(), 64U);
                EXPECT_EQ(graph.num_stored_points(), 0U);
                EXPECT_EQ(graph.num_points(), 0U);
                ASSERT_NO_THROW(
                    (symqg::QGBuilder(graph, 32, data.data(), threads, init, 42, true))
                );
            }
        }
    }
}

TEST_P(ConstructionWorkerFailure, QgRefinementWorkerErrorsReachCaller) {
    using namespace rabitqlib;
    const auto data = vectors();
    const size_t threads = GetParam();
    {
        for (const size_t skip : {0U, 4U, 64U}) {
            symqg::QuantizedGraph<float> graph(
                64, 64, 32, METRIC_L2, RotatorType::FhtKacRotator, 4, 42
            );
            symqg::QGBuilder builder(
                graph, 32, data.data(), threads, symqg::QGInitialization::Random, 42
            );
            expect_parallel_failure([&] { builder.build(); }, skip);
            EXPECT_EQ(graph.num_stored_points(), 0U);
            EXPECT_EQ(graph.num_points(), 0U);
            ASSERT_NO_THROW(builder.reset(data.data()));
            ASSERT_NO_THROW(builder.build());
            EXPECT_EQ(graph.num_stored_points(), 64U);
            EXPECT_EQ(graph.num_points(), 64U);
        }
    }
}
}  // namespace
