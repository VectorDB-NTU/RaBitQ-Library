#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <omp.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <optional>
#include <random>
#include <vector>

#include "rabitqlib/clustering/rabitqkmeans.hpp"
#include "rabitqlib/index/ivf_index.hpp"
#include "rabitqlib/index/types.hpp"

namespace {
std::atomic<bool> fail_enabled{false};
std::atomic<size_t> allocations_before_failure{0};

class AllocationFailure {
   public:
    explicit AllocationFailure(size_t before_failure) {
        allocations_before_failure = before_failure;
        fail_enabled = true;
    }
    AllocationFailure(const AllocationFailure&) = delete;
    AllocationFailure& operator=(const AllocationFailure&) = delete;
    ~AllocationFailure() { fail_enabled = false; }
};

// One-shot, so that cleanup after the failure can allocate.
void maybe_fail() {
    if (fail_enabled && allocations_before_failure.fetch_sub(1) == 0) {
        fail_enabled = false;
        throw std::bad_alloc();
    }
}
}  // namespace

// This executable is separate from the normal suite so fault injection cannot affect
// unrelated tests.
GTEST_NO_INLINE_ void* operator new(size_t size) {
    maybe_fail();
    if (void* ptr = std::malloc(size == 0 ? 1 : size)) {
        return ptr;
    }
    throw std::bad_alloc();
}

GTEST_NO_INLINE_ void* operator new(size_t size, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(size);
    } catch (...) { return nullptr; }
}

// The analyzer models delete as freeing before entering these replacements.
GTEST_NO_INLINE_ void operator delete(void* ptr) noexcept {
    std::free(ptr);  // NOLINT(clang-analyzer-cplusplus.NewDelete)
}
GTEST_NO_INLINE_ void operator delete(void* ptr, size_t) noexcept {
    std::free(ptr);  // NOLINT(clang-analyzer-cplusplus.NewDelete)
}

namespace rabitqlib {
namespace {
constexpr size_t kDim = 64;
constexpr size_t kNumPoints = 200;
constexpr size_t kNumClusters = 4;

IvfIndex new_index() {
    IvfOptions options;
    options.seed = 11;
    IvfIndex index(kDim, kNumClusters, options);
    index.cp().niter = 2;
    return index;
}

// The first add trains a staged index and fills it, so a failed operator new anywhere,
// also after training has succeeded, leaves the index untrained and empty, and a retry
// succeeds. Aligned storage and Eigen allocate with malloc-style calls and are not
// injected here.
TEST(IvfIndexFirstAddAllocationTest, EveryOperatorNewFailureLeavesTheIndexUntrained) {
    // Runtime startup allocations are not index allocations and must happen before
    // the hook starts injecting failures.
    (void)omp_get_max_threads();
    std::mt19937 generator(3);
    std::normal_distribution<float> normal;
    std::vector<float> data(kNumPoints * kDim);
    for (auto& value : data) {
        value = normal(generator);
    }
    AddOptions options;
    options.num_threads = 1;

    size_t failures = 0;
    bool succeeded = false;
    for (size_t allocation = 0; allocation < 4096; ++allocation) {
        SCOPED_TRACE(allocation);
        IvfIndex index = new_index();
        bool failed = false;
        {
            const AllocationFailure failure(allocation);
            try {
                index.add(data.data(), kNumPoints, options);
            } catch (const std::bad_alloc&) { failed = true; }
        }
        if (!failed) {
            succeeded = true;
            break;
        }
        ++failures;
        ASSERT_FALSE(index.is_trained());
        ASSERT_EQ(index.num_stored_points(), 0U);
        ASSERT_EQ(index.seed(), std::optional<uint32_t>(11));
        ASSERT_EQ(index.add(data.data(), kNumPoints, options), 0);
        ASSERT_EQ(index.num_points(), kNumPoints);
    }
    EXPECT_GT(failures, 0U);
    EXPECT_TRUE(succeeded);
}

}  // namespace
}  // namespace rabitqlib
