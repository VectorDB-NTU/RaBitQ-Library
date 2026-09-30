#include "rabitqlib/index/hnsw/hnsw.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
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

// Builds an index over the first `built` of `total` random vectors, leaving the
// rest as capacity for add.
struct AddFixture {
    static constexpr size_t kDim = 64;
    std::vector<float> data;
    std::vector<float> centroid;
    std::vector<PID> cluster_ids;

    AddFixture(size_t total, unsigned seed)
        : data(total * kDim), centroid(kDim, 0.0F), cluster_ids(total, 0) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(-1, 1);
        for (auto& value : data) {
            value = distribution(generator);
        }
    }
};

TEST(HnswAddTest, AddedPointsAreSearchable) {
    constexpr size_t kBuilt = 256;
    constexpr size_t kAdded = 64;
    constexpr size_t kTotal = kBuilt + kAdded;
    AddFixture fixture(kTotal, 7);

    HierarchicalNSW index(kTotal, AddFixture::kDim, 4, 8, 50);
    index.construct(
        1,
        fixture.centroid.data(),
        kBuilt,
        fixture.data.data(),
        fixture.cluster_ids.data(),
        1,
        false
    );
    ASSERT_EQ(index.num_points(), kBuilt);

    const auto labels = index.add(
        fixture.data.data() + (kBuilt * AddFixture::kDim),
        kAdded,
        fixture.cluster_ids.data(),
        false
    );

    ASSERT_EQ(labels.size(), kAdded);
    for (size_t i = 0; i < kAdded; ++i) {
        EXPECT_EQ(labels[i], kBuilt + i);
    }
    EXPECT_EQ(index.num_points(), kTotal);

    // Every added vector must find itself when used as its own query.
    const auto results = index.search(
        fixture.data.data() + (kBuilt * AddFixture::kDim), kAdded, 1, kTotal, 1
    );
    ASSERT_EQ(results.size(), kAdded);
    size_t exact = 0;
    for (size_t i = 0; i < kAdded; ++i) {
        ASSERT_EQ(results[i].size(), 1U);
        EXPECT_TRUE(std::isfinite(results[i][0].first));
        exact += (results[i][0].second == kBuilt + i) ? 1 : 0;
    }
    EXPECT_EQ(exact, kAdded);

    // The points placed by construct must still be reachable afterwards.
    const auto old_results = index.search(fixture.data.data(), 16, 1, kTotal, 1);
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(old_results[i][0].second, i);
    }
}

TEST(HnswAddTest, SurvivesSaveAndLoad) {
    constexpr size_t kBuilt = 128;
    constexpr size_t kAdded = 32;
    constexpr size_t kTotal = kBuilt + kAdded;
    AddFixture fixture(kTotal, 11);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "hnsw_add_roundtrip.index";

    {
        HierarchicalNSW index(kTotal, AddFixture::kDim, 4, 8, 50);
        index.construct(
            1,
            fixture.centroid.data(),
            kBuilt,
            fixture.data.data(),
            fixture.cluster_ids.data(),
            1,
            false
        );
        index.add(
            fixture.data.data() + (kBuilt * AddFixture::kDim),
            kAdded,
            fixture.cluster_ids.data(),
            false
        );
        index.save(path.string().c_str());
    }

    HierarchicalNSW loaded;
    loaded.load(path.string().c_str());
    EXPECT_EQ(loaded.num_points(), kTotal);
    EXPECT_EQ(loaded.max_elements(), kTotal);

    const auto results = loaded.search(
        fixture.data.data() + (kBuilt * AddFixture::kDim), kAdded, 1, kTotal, 1
    );
    for (size_t i = 0; i < kAdded; ++i) {
        EXPECT_EQ(results[i][0].second, kBuilt + i);
    }
    std::filesystem::remove(path);
}

TEST(HnswAddTest, RejectsBadInputWithoutDisturbingTheIndex) {
    constexpr size_t kBuilt = 64;
    constexpr size_t kTotal = kBuilt + 8;
    AddFixture fixture(kTotal, 13);

    HierarchicalNSW index(kTotal, AddFixture::kDim, 4, 8, 50);
    index.construct(
        1,
        fixture.centroid.data(),
        kBuilt,
        fixture.data.data(),
        fixture.cluster_ids.data(),
        1,
        false
    );

    const float* extra = fixture.data.data() + (kBuilt * AddFixture::kDim);
    std::vector<PID> bad_cluster(8, 5);

    EXPECT_THROW(index.add(nullptr, 8, fixture.cluster_ids.data()), std::invalid_argument);
    EXPECT_THROW(index.add(extra, 8, bad_cluster.data()), std::invalid_argument);
    // More points than the remaining capacity.
    EXPECT_THROW(index.add(extra, 9, fixture.cluster_ids.data()), std::invalid_argument);

    EXPECT_EQ(index.num_points(), kBuilt);
    EXPECT_EQ(index.add(extra, 0, fixture.cluster_ids.data()).size(), 0U);

    // The index is still searchable and still holds only what construct placed.
    const auto results = index.search(fixture.data.data(), 8, 1, kBuilt, 1);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(results[i][0].second, i);
    }
}

// Several clusters, so the per-point centroid actually varies and the correction
// terms have to pick the right one. Both metrics, since their g_add differ.
TEST(HnswAddTest, AddedPointsAreSearchableWithManyClusters) {
    constexpr size_t kBuilt = 256;
    constexpr size_t kAdded = 64;
    constexpr size_t kTotal = kBuilt + kAdded;
    constexpr size_t kClusters = 8;

    for (const MetricType metric : {METRIC_L2, METRIC_IP}) {
        AddFixture fixture(kTotal, 23);
        std::vector<float> centroids(kClusters * AddFixture::kDim);
        std::mt19937 generator(5);
        std::uniform_real_distribution<float> distribution(-1, 1);
        for (auto& value : centroids) {
            value = distribution(generator);
        }
        std::vector<PID> cluster_ids(kTotal);
        for (size_t i = 0; i < kTotal; ++i) {
            cluster_ids[i] = static_cast<PID>(i % kClusters);
        }

        HierarchicalNSW index(kTotal, AddFixture::kDim, 4, 8, 50, 100, metric);
        index.construct(
            kClusters,
            centroids.data(),
            kBuilt,
            fixture.data.data(),
            cluster_ids.data(),
            1,
            false
        );
        const auto labels = index.add(
            fixture.data.data() + (kBuilt * AddFixture::kDim),
            kAdded,
            cluster_ids.data() + kBuilt,
            false
        );
        ASSERT_EQ(labels.size(), kAdded);
        EXPECT_EQ(index.num_points(), kTotal);

        const auto results = index.search(
            fixture.data.data() + (kBuilt * AddFixture::kDim), kAdded, 1, kTotal, 1
        );
        size_t exact = 0;
        for (size_t i = 0; i < kAdded; ++i) {
            exact += (results[i][0].second == kBuilt + i) ? 1 : 0;
        }
        EXPECT_EQ(exact, kAdded);
    }
}

// Two indexes with different shapes, used one after the other: the per-thread
// query scratch must not carry over between them.
TEST(HnswAddTest, DoesNotReuseScratchAcrossIndexes) {
    constexpr size_t kCount = 64;
    for (const size_t dim : {64U, 128U}) {
        std::vector<float> data(kCount * dim);
        std::mt19937 generator(31);
        std::uniform_real_distribution<float> distribution(-1, 1);
        for (auto& value : data) {
            value = distribution(generator);
        }
        std::vector<float> centroid(dim, 0.0F);
        std::vector<PID> cluster_ids(kCount, 0);

        HierarchicalNSW index(kCount, dim, 4, 8, 50);
        index.construct(
            1, centroid.data(), kCount / 2, data.data(), cluster_ids.data(), 1, false
        );
        index.add(
            data.data() + ((kCount / 2) * dim), kCount / 2, cluster_ids.data(), false
        );

        const auto results = index.search(data.data(), kCount, 1, kCount, 1);
        for (size_t i = 0; i < kCount; ++i) {
            EXPECT_EQ(results[i][0].second, i);
        }
    }
}

TEST(HnswAddTest, RoutesToTheNearestCentroidWhenNoClustersGiven) {
    constexpr size_t kBuilt = 128;
    constexpr size_t kAdded = 32;
    constexpr size_t kTotal = kBuilt + kAdded;
    constexpr size_t kClusters = 4;
    AddFixture fixture(kTotal, 29);

    // Well-separated centroids, so the nearest one is unambiguous.
    std::vector<float> centroids(kClusters * AddFixture::kDim, 0.0F);
    for (size_t c = 0; c < kClusters; ++c) {
        centroids[(c * AddFixture::kDim) + c] = 50.0F;
    }
    std::vector<PID> cluster_ids(kTotal);
    for (size_t i = 0; i < kTotal; ++i) {
        cluster_ids[i] = static_cast<PID>(i % kClusters);
        // Push each point hard towards its centroid.
        for (size_t d = 0; d < AddFixture::kDim; ++d) {
            fixture.data[(i * AddFixture::kDim) + d] +=
                centroids[(cluster_ids[i] * AddFixture::kDim) + d];
        }
    }

    HierarchicalNSW index(kTotal, AddFixture::kDim, 4, 8, 50);
    index.construct(
        kClusters,
        centroids.data(),
        kBuilt,
        fixture.data.data(),
        cluster_ids.data(),
        1,
        false
    );
    const auto labels =
        index.add(fixture.data.data() + (kBuilt * AddFixture::kDim), kAdded, nullptr);
    ASSERT_EQ(labels.size(), kAdded);

    for (size_t i = 0; i < kAdded; ++i) {
        EXPECT_EQ(index.cluster_id_of(labels[i]), cluster_ids[kBuilt + i]);
    }
    const auto results = index.search(
        fixture.data.data() + (kBuilt * AddFixture::kDim), kAdded, 1, kTotal, 1
    );
    for (size_t i = 0; i < kAdded; ++i) {
        EXPECT_EQ(results[i][0].second, kBuilt + i);
    }
}

TEST(HnswResizeTest, GrowsCapacityAndKeepsTheGraph) {
    constexpr size_t kBuilt = 128;
    constexpr size_t kAdded = 64;
    constexpr size_t kTotal = kBuilt + kAdded;
    AddFixture fixture(kTotal, 37);

    HierarchicalNSW index(kBuilt, AddFixture::kDim, 4, 8, 50);
    index.construct(
        1,
        fixture.centroid.data(),
        kBuilt,
        fixture.data.data(),
        fixture.cluster_ids.data(),
        1,
        false
    );
    ASSERT_EQ(index.max_elements(), kBuilt);

    // Full: adding anything must fail, and leave the index usable.
    EXPECT_THROW(
        index.add(fixture.data.data() + (kBuilt * AddFixture::kDim), 1, nullptr),
        std::invalid_argument
    );
    EXPECT_EQ(index.num_points(), kBuilt);

    EXPECT_THROW(index.resize(kBuilt - 1), std::invalid_argument);
    index.resize(kTotal);
    EXPECT_EQ(index.max_elements(), kTotal);
    EXPECT_EQ(index.num_points(), kBuilt);

    // Points placed before the resize survive it.
    const auto before = index.search(fixture.data.data(), 16, 1, kBuilt, 1);
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(before[i][0].second, i);
    }

    index.add(fixture.data.data() + (kBuilt * AddFixture::kDim), kAdded, nullptr);
    EXPECT_EQ(index.num_points(), kTotal);

    const auto results = index.search(fixture.data.data(), kTotal, 1, kTotal, 1);
    for (size_t i = 0; i < kTotal; ++i) {
        EXPECT_EQ(results[i][0].second, i);
    }
}

TEST(HnswResizeTest, SurvivesSaveAndLoad) {
    constexpr size_t kBuilt = 64;
    constexpr size_t kTotal = 96;
    AddFixture fixture(kTotal, 41);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "hnsw_resize_roundtrip.index";
    {
        HierarchicalNSW index(kBuilt, AddFixture::kDim, 4, 8, 50);
        index.construct(
            1,
            fixture.centroid.data(),
            kBuilt,
            fixture.data.data(),
            fixture.cluster_ids.data(),
            1,
            false
        );
        index.resize(kTotal);
        index.add(
            fixture.data.data() + (kBuilt * AddFixture::kDim), kTotal - kBuilt, nullptr
        );
        index.save(path.string().c_str());
    }

    HierarchicalNSW loaded;
    loaded.load(path.string().c_str());
    EXPECT_EQ(loaded.max_elements(), kTotal);
    EXPECT_EQ(loaded.num_points(), kTotal);
    const auto results = loaded.search(fixture.data.data(), kTotal, 1, kTotal, 1);
    for (size_t i = 0; i < kTotal; ++i) {
        EXPECT_EQ(results[i][0].second, i);
    }
    std::filesystem::remove(path);
}

TEST(HnswAddTest, AddIntoAnEmptyIndexBuildsFromScratch) {
    constexpr size_t kCount = 96;
    AddFixture fixture(kCount, 17);

    HierarchicalNSW index(kCount, AddFixture::kDim, 4, 8, 50);
    // construct one point so the centroids and rotator exist, then add the rest.
    index.construct(
        1,
        fixture.centroid.data(),
        1,
        fixture.data.data(),
        fixture.cluster_ids.data(),
        1,
        false
    );
    const auto labels = index.add(
        fixture.data.data() + AddFixture::kDim,
        kCount - 1,
        fixture.cluster_ids.data(),
        false
    );
    ASSERT_EQ(labels.size(), kCount - 1);
    EXPECT_EQ(index.num_points(), kCount);

    const auto results = index.search(fixture.data.data(), kCount, 1, kCount, 1);
    size_t exact = 0;
    for (size_t i = 0; i < kCount; ++i) {
        exact += (results[i][0].second == i) ? 1 : 0;
    }
    // A graph built entirely from one-bit reconstructions still has to route a
    // vector to itself; the codes distinguish points far better than they order
    // near neighbors.
    EXPECT_EQ(exact, kCount);
}

TEST(HnswConstructionTest, RejectsInvalidInputsBeforeChangingIndex) {
    constexpr size_t kDim = 64;
    std::vector<float> data(kDim, 1.0F);
    std::vector<float> centroid(kDim, 0.0F);
    PID invalid_cluster[] = {1};
    PID valid_cluster[] = {0};
    HierarchicalNSW index(1, kDim, 4, 4, 10);

    try {
        index.construct(1, centroid.data(), 1, data.data(), invalid_cluster, 1, false);
        FAIL() << "Out-of-range cluster ID must be rejected";
    } catch (const std::invalid_argument& error) {
        EXPECT_STREQ(error.what(), "HNSW cluster ID is out of range");
    }
    EXPECT_THROW(
        index.construct(1, centroid.data(), 2, data.data(), valid_cluster, 1, false),
        std::invalid_argument
    );
    EXPECT_THROW(
        index.construct(1, centroid.data(), 1, nullptr, valid_cluster, 1, false),
        std::invalid_argument
    );
    EXPECT_NO_THROW(
        index.construct(1, centroid.data(), 1, data.data(), valid_cluster, 1, false)
    );
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

TEST_F(HnswSaveTest, RejectsPointCountLargerThanCapacity) {
    index_.save(path_.c_str());
    {
        std::fstream file(path_, std::ios::binary | std::ios::in | std::ios::out);
        const size_t invalid_capacity = 1;
        file.write(
            reinterpret_cast<const char*>(&invalid_capacity), sizeof(invalid_capacity)
        );
        ASSERT_TRUE(file.good());
    }

    HierarchicalNSW loaded;
    try {
        loaded.load(path_.c_str());
        FAIL() << "Invalid HNSW point count must be rejected";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("HNSW"), std::string::npos);
    }
}

TEST_F(HnswSaveTest, RejectsTruncatedRotatorWithoutLosingExistingIndex) {
    index_.save(path_.c_str());
    HierarchicalNSW loaded;
    loaded.load(path_.c_str());
    const auto expected = loaded.search(data_.data(), 1, 2, kCount, 1);

    std::filesystem::resize_file(path_, std::filesystem::file_size(path_) - 1);
    try {
        loaded.load(path_.c_str());
        FAIL() << "Truncated HNSW rotator must be rejected";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("HNSW"), std::string::npos);
    }
    EXPECT_EQ(loaded.search(data_.data(), 1, 2, kCount, 1), expected);
}

}  // namespace
}  // namespace rabitqlib::hnsw
