#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <new>
#include <random>
#include <string>
#include <vector>

#include "rabitqlib/index/hnsw/hnsw.hpp"

namespace {
std::atomic<bool> fail_enabled{false};
std::atomic<size_t> allocations_before_failure{0};
size_t allocation_size = 0;
const rabitqlib::hnsw::HierarchicalNSW* watched_index = nullptr;
size_t watched_count = 0;

class AllocationFailure {
   public:
    explicit AllocationFailure(size_t before_failure, size_t size = 0) {
        allocations_before_failure = before_failure;
        allocation_size = size;
        fail_enabled = true;
    }
    ~AllocationFailure() { fail_enabled = false; }
};
}  // namespace

// This executable is separate from the normal suite so fault injection cannot
// affect unrelated tests. Each failure is one-shot, allowing cleanup to allocate.
GTEST_NO_INLINE_ void* operator new(size_t size) {
    if (fail_enabled && (allocation_size == 0 || allocation_size == size) &&
        (watched_index == nullptr || watched_index->num_points() == watched_count)) {
        if (allocations_before_failure.fetch_sub(1) == 0) {
            fail_enabled = false;
            throw std::bad_alloc();
        }
    }
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

namespace rabitqlib::hnsw {
namespace {
constexpr size_t kDim = 64;
constexpr size_t kAdded = 3;
constexpr size_t kCapacity = 8;

struct Fixture {
    std::array<float, kCapacity * kDim> data{};
    std::array<float, kDim> centroid{};
    std::array<PID, kCapacity> clusters{};

    Fixture() {
        std::mt19937 generator(42);
        std::uniform_real_distribution<float> distribution(-1, 1);
        for (auto& value : data) {
            value = distribution(generator);
        }
    }

    void build(HierarchicalNSW& index) const {
        auto ids = clusters;
        index.construct(1, centroid.data(), 1, data.data(), ids.data(), 1, false);
    }
};

TEST(HnswConstructionAllocationTest, EveryAllocationFailureLeavesEmptyReusableIndex) {
    Fixture fixture;
    for (const size_t threads : {1U, 4U}) {
        SCOPED_TRACE(threads);
        size_t failures = 0;
        bool succeeded = false;
        for (size_t allocation = 0; allocation < 512; ++allocation) {
            SCOPED_TRACE(allocation);
            HierarchicalNSW index(kCapacity, kDim, 4, 2, 10, 1);
            auto clusters = fixture.clusters;
            bool failed = false;
            {
                AllocationFailure failure(allocation);
                try {
                    index.construct(
                        1,
                        fixture.centroid.data(),
                        kAdded,
                        fixture.data.data(),
                        clusters.data(),
                        threads,
                        false
                    );
                } catch (const std::bad_alloc&) { failed = true; }
            }
            if (!failed) {
                succeeded = true;
                break;
            }
            ++failures;
            EXPECT_EQ(index.num_points(), 0U);
            EXPECT_EQ(index.num_clusters(), 0U);
            const auto empty = index.search(fixture.data.data(), 1, 1, kCapacity, 1);
            ASSERT_EQ(empty.size(), 1U);
            EXPECT_TRUE(empty[0].empty());
            ASSERT_NO_THROW(index.construct(
                1,
                fixture.centroid.data(),
                kAdded,
                fixture.data.data(),
                clusters.data(),
                threads,
                false
            ));
            EXPECT_EQ(index.num_points(), kAdded);
            EXPECT_EQ(index.num_clusters(), 1U);
            const auto results = index.search(fixture.data.data(), kAdded, 1, kCapacity, 1);
            for (size_t i = 0; i < kAdded; ++i) {
                ASSERT_EQ(results[i].size(), 1U);
                EXPECT_EQ(results[i][0].second, i);
            }
        }
        EXPECT_GT(failures, 0U);
        EXPECT_TRUE(succeeded);
    }
}

TEST(HnswAddAllocationTest, AllocatesReturnedIdsBeforeMutation) {
    Fixture fixture;
    HierarchicalNSW index(kCapacity, kDim, 4, 2, 10, 1);
    fixture.build(index);
    bool preparation_failed = false;
    {
        AllocationFailure failure(0, kAdded * sizeof(PID));
        try {
            index.add(fixture.data.data() + kDim, kAdded, fixture.clusters.data());
        } catch (const std::bad_alloc&) { preparation_failed = true; }
    }
    ASSERT_TRUE(preparation_failed);
    EXPECT_EQ(index.num_points(), 1U);
    watched_index = &index;
    watched_count = 1 + kAdded;
    std::vector<PID> ids;
    bool failed = false;
    {
        AllocationFailure failure(0, kAdded * sizeof(PID));
        try {
            ids = index.add(fixture.data.data() + kDim, kAdded, fixture.clusters.data());
        } catch (const std::bad_alloc&) { failed = true; }
    }
    watched_index = nullptr;
    EXPECT_FALSE(failed);
    EXPECT_EQ(ids, (std::vector<PID>{1, 2, 3}));
    EXPECT_EQ(index.num_points(), 1 + kAdded);
}

TEST(HnswAddAllocationTest, PreparationFailurePreservesIndex) {
    for (const size_t threads : {1U, 4U}) {
        SCOPED_TRACE(threads);
        Fixture fixture;
        HierarchicalNSW index(kCapacity, kDim, 4, 2, 10, 1);
        fixture.build(index);
        const auto before = index.search(fixture.data.data(), 1, 1, kCapacity, 1);
        bool failed = false;
        {
            AllocationFailure failure(0, kDim * sizeof(float));
            try {
                index.add(
                    fixture.data.data() + kDim,
                    kAdded,
                    fixture.clusters.data(),
                    false,
                    threads
                );
            } catch (const std::bad_alloc&) { failed = true; }
        }
        ASSERT_TRUE(failed);
        EXPECT_EQ(index.num_points(), 1U);
        EXPECT_EQ(index.search(fixture.data.data(), 1, 1, kCapacity, 1), before);
        EXPECT_EQ(
            index.add(fixture.data.data() + kDim, kAdded, fixture.clusters.data()),
            (std::vector<PID>{1, 2, 3})
        );
    }
}

TEST(HnswAddAllocationTest, EveryAllocationFailureLeavesUsableIndex) {
    Fixture fixture;
    const std::string path =
        ::testing::TempDir() + "rabitq_hnsw_add_allocation_failure.index";
    // Exercise interrupted promotions from level 0 (freeing the upper list) and
    // level 1 (retaining its lower part). Use this platform's default engine,
    // just as HNSW does, so the selected seeds work across standard libraries.
    std::array<size_t, 2> seeds{};
    for (size_t level = 0; level < seeds.size(); ++level) {
        for (size_t seed = 1; seed < 10000; ++seed) {
            std::default_random_engine generator(seed);
            std::uniform_real_distribution<double> distribution(0, 1);
            const auto built_level =
                static_cast<size_t>(-std::log(distribution(generator)) / std::log(2.0));
            const auto added_level =
                static_cast<size_t>(-std::log(distribution(generator)) / std::log(2.0));
            if (built_level == level && added_level > level) {
                seeds[level] = seed;
                break;
            }
        }
        ASSERT_NE(seeds[level], 0U);
    }
    for (const size_t seed : seeds) {
        SCOPED_TRACE(seed);
        for (const size_t threads : {1U, 4U}) {
            SCOPED_TRACE(threads);
            size_t failures = 0;
            size_t linking_failures = 0;
            bool succeeded = false;
            for (size_t allocation = 0; allocation < 256; ++allocation) {
                SCOPED_TRACE(allocation);
                HierarchicalNSW index(kCapacity, kDim, 4, 2, 10, seed);
                fixture.build(index);
                bool failed = false;
                {
                    AllocationFailure failure(allocation);
                    try {
                        index.add(
                            fixture.data.data() + kDim,
                            kAdded,
                            fixture.clusters.data(),
                            false,
                            threads
                        );
                    } catch (const std::bad_alloc&) { failed = true; }
                }
                failures += failed ? 1 : 0;
                linking_failures += (failed && index.num_points() > 1) ? 1 : 0;
                const auto results = index.search(fixture.data.data(), 1, 1, kCapacity, 1);
                ASSERT_EQ(results.size(), 1U);
                ASSERT_EQ(results[0].size(), 1U);
                EXPECT_EQ(results[0][0].second, 0U);
                ASSERT_NO_THROW(index.save(path.c_str()));
                HierarchicalNSW loaded;
                ASSERT_NO_THROW(loaded.load(path.c_str()));
                EXPECT_EQ(loaded.num_points(), index.num_points());
                EXPECT_EQ(loaded.search(fixture.data.data(), 1, 1, kCapacity, 1), results);
                const auto next_id = static_cast<PID>(loaded.num_points());
                EXPECT_EQ(
                    loaded.add(
                        fixture.data.data() + ((1 + kAdded) * kDim),
                        1,
                        fixture.clusters.data()
                    ),
                    (std::vector<PID>{next_id})
                );
                if (!failed) {
                    succeeded = true;
                    break;
                }
            }
            EXPECT_GT(failures, 0U);
            EXPECT_GT(linking_failures, 0U);
            EXPECT_TRUE(succeeded);
        }
    }
    std::filesystem::remove(path);
}

TEST(HnswRemoveAllocationTest, FailureRemovesNothing) {
    Fixture fixture;
    HierarchicalNSW index(kCapacity, kDim, 4, 2, 10, 1);
    auto ids = fixture.clusters;
    index.construct(
        1, fixture.centroid.data(), kCapacity, fixture.data.data(), ids.data(), 1, false
    );
    const auto before =
        index.search(fixture.data.data(), kCapacity, kCapacity, kCapacity, 1);
    const std::array<PID, 3> labels{0, 4, 7};
    bool failed = false;
    {
        // The only allocation in remove holds the resolved internal ids.
        AllocationFailure failure(0, labels.size() * sizeof(PID));
        try {
            index.remove(labels.data(), labels.size());
        } catch (const std::bad_alloc&) { failed = true; }
    }
    ASSERT_TRUE(failed);
    EXPECT_EQ(
        index.search(fixture.data.data(), kCapacity, kCapacity, kCapacity, 1), before
    );
    EXPECT_EQ(index.remove(labels.data(), labels.size()), labels.size());
}
}  // namespace
}  // namespace rabitqlib::hnsw
