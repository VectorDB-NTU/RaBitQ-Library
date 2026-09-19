#include "rabitqlib/index/hnsw/hnsw.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace rabitqlib::hnsw {
namespace {

TEST(HnswConfigurationTest, RejectsUnsupportedMetric) {
    EXPECT_THROW(
        (HierarchicalNSW(8, 64, 1, 2, 10, 100, static_cast<MetricType>(255))),
        std::invalid_argument
    );
}

TEST(HnswOneBitSearchTest, UsesBinaryEstimateForEntryPoint) {
    constexpr size_t dim = 64;
    constexpr size_t count = 8;
    std::vector<float> data(count * dim);
    std::vector<float> centroid(dim);
    std::vector<float> query(dim);

    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 5.0F;
        query[i] = centroid[i];
    }
    for (size_t point = 0; point < count; ++point) {
        for (size_t i = 0; i < dim; ++i) {
            data[(point * dim) + i] =
                centroid[i] + static_cast<float>((point + 1) * ((i % 11) + 1));
        }
    }

    std::vector<PID> cluster_ids(count, 0);
    HierarchicalNSW index(count, dim, 1, 2, 10, 100, METRIC_L2);
    index.construct(1, centroid.data(), count, data.data(), cluster_ids.data(), 1, false);

    const auto results = index.search(query.data(), 1, 1, 10, 1);

    ASSERT_EQ(results.size(), 1U);
    ASSERT_EQ(results[0].size(), 1U);
    ASSERT_EQ(results[0][0].second, 0U);
    EXPECT_TRUE(std::isfinite(results[0][0].first));
    const float exact_distance = euclidean_sqr(query.data(), data.data(), dim);
    EXPECT_NEAR(results[0][0].first, exact_distance, exact_distance * 0.1F);
}

TEST(HnswConstructionTest, ParallelConstructionProducesSearchableIndex) {
    constexpr size_t kDim = 64;
    constexpr size_t kCount = 512;
    constexpr size_t kQueries = 16;
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1, 1);
    std::vector<float> data(kCount * kDim);
    for (auto& value : data) {
        value = distribution(generator);
    }
    std::vector<float> centroid(kDim);
    std::vector<PID> cluster_ids(kCount, 0);
    HierarchicalNSW index(kCount, kDim, 4, 8, 50);
    index.construct(1, centroid.data(), kCount, data.data(), cluster_ids.data(), 8, false);

    const auto results = index.search(data.data(), kQueries, 1, kCount, 1);
    ASSERT_EQ(results.size(), kQueries);
    for (size_t i = 0; i < kQueries; ++i) {
        ASSERT_EQ(results[i].size(), 1U);
        EXPECT_EQ(results[i][0].second, i);
        EXPECT_TRUE(std::isfinite(results[i][0].first));
    }
}

class HnswSaveTest : public ::testing::Test {
   protected:
    static constexpr size_t kDim = 64;
    static constexpr size_t kCount = 8;
    std::vector<float> data_{std::vector<float>(kCount * kDim)};
    std::vector<float> centroid_{std::vector<float>(kDim)};
    std::vector<PID> cluster_ids_{std::vector<PID>(kCount, 0)};
    HierarchicalNSW index_{kCount, kDim, 4, 4, 10};
    std::string path_;

    void SetUp() override {
        path_ = ::testing::TempDir() + "rabitq_hnsw_" +
                ::testing::UnitTest::GetInstance()->current_test_info()->name() + ".index";
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] = static_cast<float>((i * 37) % 127) / 32.0F;
        }
        index_.construct(
            1, centroid_.data(), kCount, data_.data(), cluster_ids_.data(), 1, false
        );
    }

    void TearDown() override { std::remove(path_.c_str()); }
};

TEST_F(HnswSaveTest, RejectsUnopenableDestination) {
    try {
        index_.save((path_ + "/index").c_str());
        FAIL() << "Saving to a missing directory must fail";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "HNSW: cannot open index file for writing");
    }
}

TEST_F(HnswSaveTest, ReportsWriteOrCloseFailure) {
    if (!std::filesystem::exists("/dev/full")) {
        GTEST_SKIP() << "/dev/full is unavailable";
    }
    try {
        index_.save("/dev/full");
        FAIL() << "Saving to a full destination must fail";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "HNSW: failed to write index file");
    }
}

TEST_F(HnswSaveTest, SuccessfulSavePreservesSearchAfterLoading) {
    ASSERT_NO_THROW(index_.save(path_.c_str()));
    HierarchicalNSW loaded;
    ASSERT_NO_THROW(loaded.load(path_.c_str()));
    EXPECT_EQ(loaded.dimension(), index_.dimension());
    EXPECT_EQ(loaded.num_clusters(), index_.num_clusters());
    EXPECT_EQ(loaded.nbits(), index_.nbits());
    EXPECT_EQ(
        loaded.search(data_.data(), kCount, 2, kCount, 1),
        index_.search(data_.data(), kCount, 2, kCount, 1)
    );
}

}  // namespace
}  // namespace rabitqlib::hnsw
