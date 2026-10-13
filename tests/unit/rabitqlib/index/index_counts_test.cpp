#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib {
namespace {
constexpr size_t kCount = 40;
constexpr size_t kDim = 64;

std::vector<float> count_test_vectors() {
    std::vector<float> data((kCount + 4) * kDim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.17F);
    }
    return data;
}

template <class Index>
void expect_counts(const Index& index, size_t capacity, size_t stored, size_t active) {
    EXPECT_EQ(index.max_elements(), capacity);
    EXPECT_EQ(index.num_stored_points(), stored);
    EXPECT_EQ(index.num_points(), active);
    EXPECT_LE(index.num_points(), index.num_stored_points());
    EXPECT_LE(index.num_stored_points(), index.max_elements());
}

TEST(IndexCounts, IvfTracksRemovalGrowthReloadAndRebuild) {
    const auto data = count_test_vectors();
    const std::vector<float> centroid(kDim, 0.0F);
    const std::vector<PID> clusters(kCount + 4, 0);
    const std::array<PID, 3> removed{1, kCount - 1, 1};
    const std::string path = ::testing::TempDir() + "rabitq_ivf_counts.index";
    for (const auto metric : {METRIC_L2, METRIC_IP}) {
        for (size_t bits : {1UL, 4UL, 32UL}) {
            SCOPED_TRACE(::testing::Message() << "bits=" << bits << " metric=" << metric);
            ivf::IVF index(kCount, kDim, 1, bits, metric);
            expect_counts(index, kCount, 0, 0);
            index.construct(data.data(), centroid.data(), clusters.data(), false, 1);
            expect_counts(index, kCount, kCount, kCount);
            EXPECT_EQ(index.remove(removed.data(), removed.size()), 2U);
            EXPECT_EQ(index.remove(removed.data(), removed.size()), 0U);
            expect_counts(index, kCount, kCount, kCount - 2);
            EXPECT_THROW(index.add(nullptr, 1), std::invalid_argument);
            const std::array<PID, 2> invalid{0, kCount};
            EXPECT_THROW(
                index.remove(invalid.data(), invalid.size()), std::invalid_argument
            );
            expect_counts(index, kCount, kCount, kCount - 2);

            index.add(data.data() + kCount * kDim, 3, clusters.data(), false, 1);
            expect_counts(index, kCount + 3, kCount + 3, kCount + 1);
            index.save(path.c_str());
            ivf::IVF loaded;
            expect_counts(loaded, 0, 0, 0);
            loaded.load(path.c_str());
            expect_counts(loaded, kCount + 3, kCount + 3, kCount + 1);
            std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
            EXPECT_THROW(loaded.load(path.c_str()), std::runtime_error);
            expect_counts(loaded, kCount + 3, kCount + 3, kCount + 1);

            std::vector<PID> all(kCount + 3);
            std::iota(all.begin(), all.end(), PID{0});
            EXPECT_EQ(loaded.remove(all.data(), all.size()), kCount + 1);
            expect_counts(loaded, kCount + 3, kCount + 3, 0);
            loaded.save(path.c_str());
            loaded.load(path.c_str());
            expect_counts(loaded, kCount + 3, kCount + 3, 0);
            loaded.add(data.data() + (kCount + 3) * kDim, 1, clusters.data(), false, 1);
            expect_counts(loaded, kCount + 4, kCount + 4, 1);
            const PID appended_id = kCount + 3;
            EXPECT_EQ(loaded.remove(&appended_id, 1), 1U);
            expect_counts(loaded, kCount + 4, kCount + 4, 0);

            // Rebuilding repopulates every slot and clears all prior tombstones.
            loaded.construct(data.data(), centroid.data(), clusters.data(), false, 1);
            expect_counts(loaded, kCount + 4, kCount + 4, kCount + 4);
        }
    }
    std::filesystem::remove(path);
}

TEST(IndexCounts, HnswSeparatesCapacityStoredLabelsAndActivePoints) {
    const auto data = count_test_vectors();
    const std::vector<float> centroid(kDim, 0.0F);
    std::vector<PID> clusters(kCount, 0);
    constexpr size_t kCapacity = 64;
    const std::array<PID, 3> removed{1, kCount - 1, 1};
    const std::string path = ::testing::TempDir() + "rabitq_hnsw_counts.index";
    hnsw::HierarchicalNSW index(kCapacity, kDim, 4, 8, 40);
    expect_counts(index, kCapacity, 0, 0);
    index.construct(1, centroid.data(), kCount, data.data(), clusters.data(), 1);
    expect_counts(index, kCapacity, kCount, kCount);
    EXPECT_EQ(index.remove(removed.data(), removed.size()), 2U);
    EXPECT_EQ(index.remove(removed.data(), removed.size()), 0U);
    expect_counts(index, kCapacity, kCount, kCount - 2);
    EXPECT_THROW(index.resize(kCount - 2), std::invalid_argument);
    EXPECT_THROW(index.add(nullptr, 1), std::invalid_argument);
    expect_counts(index, kCapacity, kCount, kCount - 2);
    index.resize(2 * kCapacity);
    expect_counts(index, 2 * kCapacity, kCount, kCount - 2);
    EXPECT_EQ(
        index.add(data.data() + kCount * kDim, 3, clusters.data(), false, 1),
        (std::vector<PID>{kCount, kCount + 1, kCount + 2})
    );
    expect_counts(index, 2 * kCapacity, kCount + 3, kCount + 1);
    index.save(path.c_str());
    hnsw::HierarchicalNSW loaded;
    expect_counts(loaded, 0, 0, 0);
    loaded.load(path.c_str());
    expect_counts(loaded, 2 * kCapacity, kCount + 3, kCount + 1);
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    EXPECT_THROW(loaded.load(path.c_str()), std::runtime_error);
    expect_counts(loaded, 2 * kCapacity, kCount + 3, kCount + 1);
    std::vector<PID> all(kCount + 3);
    std::iota(all.begin(), all.end(), PID{0});
    EXPECT_EQ(loaded.remove(all.data(), all.size()), kCount + 1);
    expect_counts(loaded, 2 * kCapacity, kCount + 3, 0);
    loaded.save(path.c_str());
    loaded.load(path.c_str());
    expect_counts(loaded, 2 * kCapacity, kCount + 3, 0);
    EXPECT_EQ(
        loaded.add(data.data() + (kCount + 3) * kDim, 1, clusters.data(), false, 1),
        std::vector<PID>{kCount + 3}
    );
    expect_counts(loaded, 2 * kCapacity, kCount + 4, 1);
    std::filesystem::remove(path);
}

TEST(IndexCounts, SymphonyCountsOnlyCompletedBuilds) {
    const auto data = count_test_vectors();
    const std::string path = ::testing::TempDir() + "rabitq_symqg_counts.index";
    for (size_t bits : {0UL, 4UL, 8UL}) {
        symqg::QuantizedGraph<float> index(
            kCount, kDim, 32, METRIC_L2, RotatorType::FhtKacRotator, bits, 42
        );
        expect_counts(index, kCount, 0, 0);
        symqg::QGBuilder builder(index, kCount, data.data(), 1, 42);
        expect_counts(index, kCount, 0, 0);
        builder.build();
        expect_counts(index, kCount, kCount, kCount);
        index.save(path.c_str());
        symqg::QuantizedGraph<float> loaded;
        expect_counts(loaded, 0, 0, 0);
        loaded.load(path.c_str());
        expect_counts(loaded, kCount, kCount, kCount);
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
        EXPECT_THROW(loaded.load(path.c_str()), std::runtime_error);
        expect_counts(loaded, kCount, kCount, kCount);
        builder.reset(data.data());
        expect_counts(index, kCount, 0, 0);
        builder.refine();
        expect_counts(index, kCount, kCount, kCount);
    }
    std::filesystem::remove(path);
}
}  // namespace
}  // namespace rabitqlib
