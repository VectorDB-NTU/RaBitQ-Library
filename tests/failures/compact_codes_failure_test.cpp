#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <thread>
#include <vector>

#include "rabitqlib/core/detail/rabitq_impl.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/utils/openmp.hpp"
#ifdef RABITQ_TEST_FASTSCAN
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/detail/encoder.hpp"
#include "rabitqlib/fastscan/encoder.hpp"
#include "rabitqlib/fastscan/query.hpp"
#endif

namespace {
std::atomic<bool> watching{false};
std::atomic<size_t> allocation_count{0};
std::atomic<int> deepest_region{0};
size_t failure_at = std::numeric_limits<size_t>::max();

bool fail_allocation() {
    if (!watching) {
        return false;
    }
#if RABITQ_USE_OPENMP
#if _OPENMP >= 200805
    const int level = omp_get_level();
#else
    // OpenMP 2.0 can identify active teams, but not serialized regions.
    const int level = omp_in_parallel();
#endif
    int previous = deepest_region.load();
    while (previous < level && !deepest_region.compare_exchange_weak(previous, level)) {}
#endif
    return allocation_count.fetch_add(1) == failure_at;
}
}  // namespace

#ifdef RABITQ_WRAP_COMPACT_ALLOCATIONS
extern "C" void* __real_malloc(size_t);
extern "C" void* __real_realloc(void*, size_t);
extern "C" void* __wrap_malloc(size_t bytes) {
    return fail_allocation() ? nullptr : __real_malloc(bytes);
}
extern "C" void* __wrap_realloc(void* ptr, size_t bytes) {
    return fail_allocation() ? nullptr : __real_realloc(ptr, bytes);
}
#endif

GTEST_NO_INLINE_ void* operator new(size_t bytes) {
    if (fail_allocation()) {
        throw std::bad_alloc();
    }
#ifdef RABITQ_WRAP_COMPACT_ALLOCATIONS
    void* ptr = __real_malloc(bytes == 0 ? 1 : bytes);
#else
    void* ptr = std::malloc(bytes == 0 ? 1 : bytes);
#endif
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
}
GTEST_NO_INLINE_ void* operator new[](size_t bytes) { return ::operator new(bytes); }
GTEST_NO_INLINE_ void* operator new(size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(bytes);
    } catch (...) { return nullptr; }
}
GTEST_NO_INLINE_ void* operator new[](size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return ::operator new[](bytes);
    } catch (...) { return nullptr; }
}
GTEST_NO_INLINE_ void operator delete(void* ptr) noexcept {
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
template <typename T>
struct Outcome {
    bool caught = false;
    bool first_ok = false;
    bool retry_ok = false;
    size_t allocations = 0;
    int level = 0;
    bool outer_active = false;
    std::vector<uint8_t> codes;
    std::vector<T> add, rescale, error;

    bool matches(
        const uint8_t* actual_codes,
        const std::vector<T>& actual_add,
        const std::vector<T>& actual_rescale,
        const std::vector<T>& actual_error
    ) const {
        return std::equal(codes.begin(), codes.end(), actual_codes) && add == actual_add &&
               rescale == actual_rescale && error == actual_error;
    }
};

template <typename T, bool Parallel, bool Batch, bool Public = false>
Outcome<T> run_cold_thread(
    bool grow, bool outer_region, size_t fail_at, const Outcome<T>* reference = nullptr
) {
    Outcome<T> outcome;
    // A new native thread gives both it and its OpenMP team cold TLS buffers.
    std::thread worker([&] {
#if RABITQ_USE_OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(2);
#endif
        constexpr size_t kRows = 4, kDim = 128;
        std::vector<T> data(kRows * kDim, T{0.5}), centroid(kDim, T{0});
        std::vector<uint8_t> codes((Batch ? 32 : kRows) * kDim / 8);
        std::vector<T> add(kRows), rescale(kRows), error(kRows);
#ifdef RABITQ_TEST_FASTSCAN
        std::vector<char> batch(rabitqlib::BatchDataMap<T>::data_bytes(kDim));
#endif
        const auto encode = [&](size_t dim) {
#ifdef RABITQ_TEST_FASTSCAN
            if constexpr (Public) {
                rabitqlib::quant::quantize_one_batch<T, Parallel>(
                    data.data(), centroid.data(), kRows, dim, batch.data()
                );
                rabitqlib::BatchDataMap<T> view(batch.data(), dim);
                view.f_add().copy_to(add.data(), kRows);
                view.f_rescale().copy_to(rescale.data(), kRows);
                view.f_error().copy_to(error.data(), kRows);
            } else if constexpr (Batch) {
                rabitqlib::quant::rabitq_impl::one_bit::one_bit_batch_code<T, Parallel>(
                    data.data(),
                    centroid.data(),
                    kRows,
                    dim,
                    codes.data(),
                    add.data(),
                    rescale.data(),
                    error.data()
                );
            } else
#endif
            {
                rabitqlib::quant::rabitq_impl::one_bit::
                    one_bit_compact_codes<T, uint8_t, Parallel>(
                        data.data(),
                        centroid.data(),
                        kRows,
                        dim,
                        codes.data(),
                        add.data(),
                        rescale.data(),
                        error.data()
                    );
            }
        };
        if (grow) {
            encode(32);
        }
        const uint8_t* code_data = codes.data();
#ifdef RABITQ_TEST_FASTSCAN
        if constexpr (Public) {
            code_data = rabitqlib::ConstBatchDataMap<T>(batch.data(), kDim).bin_code();
        }
#endif
        const auto valid_result = [&] {
            for (size_t i = 0; i < kRows; ++i) {
                if (!std::isfinite(add[i]) || !std::isfinite(rescale[i]) ||
                    !std::isfinite(error[i])) {
                    return false;
                }
            }
            return reference == nullptr ||
                   reference->matches(code_data, add, rescale, error);
        };
        const auto exercise = [&] {
#if RABITQ_USE_OPENMP
            outcome.outer_active = omp_in_parallel() != 0;
#endif
            failure_at = fail_at;
            allocation_count = 0;
            deepest_region = 0;
            watching = true;
            try {
                encode(kDim);
            } catch (const std::bad_alloc&) { outcome.caught = true; }
            watching = false;
            outcome.allocations = allocation_count;
            outcome.level = deepest_region;
            outcome.first_ok = !outcome.caught && valid_result();
        };
#if RABITQ_USE_OPENMP
        if (outer_region) {
#pragma omp parallel num_threads(1)
            { exercise(); }
        } else
#else
        (void)outer_region;
#endif
        {
            exercise();
        }
        // Capture the no-failure probe outside allocation tracking. Each trial
        // still starts on a fresh thread and compares against this fixed result.
        if (reference == nullptr) {
            outcome.codes.assign(code_data, code_data + codes.size());
            outcome.add = add;
            outcome.rescale = rescale;
            outcome.error = error;
            reference = &outcome;
        }
        encode(kDim);
        outcome.retry_ok = valid_result();
    });
    worker.join();
    return outcome;
}

template <typename T, bool Parallel, bool Batch, bool Public = false>
void check_failures(bool outer_region) {
    for (bool grow : {false, true}) {
        const auto probe = run_cold_thread<T, Parallel, Batch, Public>(
            grow, outer_region, std::numeric_limits<size_t>::max()
        );
        ASSERT_FALSE(probe.caught);
        ASSERT_TRUE(probe.first_ok);
        ASSERT_TRUE(probe.retry_ok);
        ASSERT_GT(probe.allocations, Batch ? 1U : 0U);
#if RABITQ_USE_OPENMP
#if _OPENMP >= 200805
        EXPECT_EQ(probe.level, Parallel ? 1 : (outer_region ? 1 : 0));
#else
        EXPECT_EQ(probe.level, Parallel || probe.outer_active ? 1 : 0);
#endif
#else
        EXPECT_EQ(probe.level, 0);
#endif
        size_t caught_count = 0;
        for (size_t point = 0; point < probe.allocations; ++point) {
            SCOPED_TRACE(::testing::Message() << "grow=" << grow << " failure=" << point);
            const auto result = run_cold_thread<T, Parallel, Batch, Public>(
                grow, outer_region, point, &probe
            );
            // An OpenMP runtime allocation can fail without stopping encoding.
            // In that case the first result must match the no-failure probe.
            EXPECT_TRUE(result.caught || result.first_ok);
            caught_count += result.caught;
            EXPECT_TRUE(result.retry_ok);
        }
        EXPECT_GT(caught_count, 0U);
    }
}

TEST(CompactCodesFailure, ReferenceRejectsIncompleteAndCorruptOutputs) {
    Outcome<float> reference;
    reference.codes = {0xAA, 0x55};
    reference.add = {1};
    reference.rescale = {2};
    reference.error = {3};
    auto actual = reference;
    const auto matches = [&] {
        return reference.matches(
            actual.codes.data(), actual.add, actual.rescale, actual.error
        );
    };
    EXPECT_TRUE(matches());
    actual.codes[0] ^= 1;
    EXPECT_FALSE(matches());
    actual = reference;
    for (auto* factors : {&actual.add, &actual.rescale, &actual.error}) {
        const float original = factors->front();
        factors->front() = 0;
        EXPECT_FALSE(matches());
        factors->front() = original;
    }
    // Unwritten output is finite, but cannot pass the reference comparison.
    std::fill(actual.codes.begin(), actual.codes.end(), 0);
    actual.add[0] = actual.rescale[0] = actual.error[0] = 0;
    EXPECT_FALSE(matches());
}

TEST(CompactCodesFailure, SerialColdScratchAndGrowth) {
    check_failures<float, false, false>(false);
    check_failures<double, false, false>(false);
}
TEST(CompactCodesFailure, ParallelColdScratchAndGrowth) {
    check_failures<float, true, false>(false);
    check_failures<double, true, false>(false);
}
TEST(CompactCodesFailure, SerialInsideOuterRegion) {
    check_failures<float, false, false>(true);
    check_failures<double, false, false>(true);
}
#ifdef RABITQ_TEST_FASTSCAN
TEST(SplitBatchQueryAllocation, ReusesCapacityAcrossQueriesModesAndDimensions) {
    std::vector<float> values(128, 0.5F), next(128, 0.25F), scratch(128 * 4);
    std::vector<uint16_t> hacc_scratch(128 * 4);
    rabitqlib::SplitBatchQuery<float> query(values.data(), 128, 3);
    const auto* storage = query.lut();
    failure_at = 0;
    allocation_count = 0;
    bool caught = false;
    watching = true;
    try {
        for (size_t repeat = 0; repeat < 3; ++repeat) {
            query.reset(
                next.data(), 64, 0, scratch.data(), nullptr, rabitqlib::METRIC_IP, false
            );
            query.reset(values.data(), 128, 8, scratch.data(), hacc_scratch.data());
        }
    } catch (const std::bad_alloc&) { caught = true; }
    watching = false;
    EXPECT_FALSE(caught);
    EXPECT_EQ(allocation_count, 0U);
    EXPECT_EQ(query.lut(), storage);
    EXPECT_EQ(query.padded_dim(), 128U);
    EXPECT_TRUE(query.use_hacc());
}

TEST(SplitBatchQueryAllocation, InitialPreparationAndGrowthFailurePreserveState) {
#if RABITQ_USE_OPENMP
    // The allocation hook queries OpenMP. Initialize its runtime before enabling
    // injection so runtime startup allocations cannot re-enter initialization.
    (void)omp_get_max_threads();
#endif
    std::vector<float> values(128, 0.5F), next(128, 0.25F), scratch(128 * 4);
    std::vector<uint16_t> hacc_scratch(128 * 4);
    for (bool prepared : {false, true}) {
        rabitqlib::SplitBatchQuery<float> query;
        if (prepared) {
            query.reset(
                values.data(), 32, 3, scratch.data(), nullptr, rabitqlib::METRIC_IP, false
            );
            query.set_g_add(3, 2);
        }
        const auto original = query;
        std::vector<uint8_t> bytes;
        if (prepared) {
            bytes.assign(query.lut(), query.lut() + 32 * 4);
        }
        failure_at = 0;
        allocation_count = 0;
        bool caught = false;
        watching = true;
        try {
            query.reset(next.data(), 128, 8, scratch.data(), hacc_scratch.data());
        } catch (const std::bad_alloc&) { caught = true; }
        watching = false;
        EXPECT_TRUE(caught);
        EXPECT_EQ(allocation_count, 1U);
        EXPECT_EQ(query.rotated_query(), original.rotated_query());
        EXPECT_EQ(query.padded_dim(), original.padded_dim());
        EXPECT_EQ(query.use_hacc(), original.use_hacc());
        EXPECT_EQ(query.delta(), original.delta());
        EXPECT_EQ(query.sum_vl_lut(), original.sum_vl_lut());
        EXPECT_EQ(query.g_k1xsumq(), original.g_k1xsumq());
        EXPECT_EQ(query.g_kbxsumq(), original.g_kbxsumq());
        EXPECT_EQ(query.g_add(), original.g_add());
        EXPECT_EQ(query.g_error(), original.g_error());
        if (prepared) {
            EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(), query.lut()));
            query.set_g_add(5, 4);
            EXPECT_EQ(query.g_add(), -4);
        }
        query.reset(next.data(), 128, 8, scratch.data(), hacc_scratch.data());
        EXPECT_EQ(query.rotated_query(), next.data());
        EXPECT_EQ(query.padded_dim(), 128U);
        EXPECT_EQ(query.g_add(), 0);
        EXPECT_EQ(query.g_error(), 0);
    }
}

TEST(CompactCodesFailure, PublicBatchSerialAndParallelPropagateAndRecover) {
    check_failures<float, false, true, true>(true);
    check_failures<float, true, true, true>(false);
    check_failures<double, true, true, true>(false);
}
TEST(CompactCodesFailure, BatchInnerScratchAfterOuterAllocation) {
    check_failures<float, false, true>(true);
    check_failures<double, false, true>(true);
    check_failures<float, true, true>(false);
}
#endif
}  // namespace
