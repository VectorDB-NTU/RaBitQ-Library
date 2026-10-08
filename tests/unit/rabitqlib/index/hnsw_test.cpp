#include "rabitqlib/index/hnsw/hnsw.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/utils/cpu_features.hpp"

namespace rabitqlib::hnsw {

struct HnswPruningTestAccess {
    static void quantize(HierarchicalNSW& index, PID id, const float* rotated) {
        const auto* centroid = reinterpret_cast<const float*>(index.centroids_memory_);
        quant::quantize_compact_one_bit(
            rotated, centroid, index.padded_dim_, index.get_bindata_by_internalid(id)
        );
    }

    static float distance(const HierarchicalNSW& index, PID target, PID query) {
        return index.get_quant_dist(target, query);
    }

    static std::vector<PID> connect(HierarchicalNSW& index, PID incoming) {
        PID* neighbors = index.get_linklist0(0);
        index.set_list_count(neighbors, index.maxM0_);
        for (size_t i = 0; i < index.maxM0_; ++i) {
            neighbors[i + 1] = static_cast<PID>(i + 1);
        }
        std::fill_n(index.get_linklist0(incoming), index.maxM0_ + 1, PID{0});
        detail::quant_query().clear();
        maxheap<std::pair<float, PID>> candidates;
        candidates.emplace(index.get_quant_dist(0, incoming), 0);
        index.mutually_connect_quant(incoming, candidates, 0);
        return {neighbors + 1, neighbors + 1 + index.get_list_count(neighbors)};
    }

    static PID entry_point_label(const HierarchicalNSW& index) {
        return index.get_external_label(index.enterpoint_node_);
    }

    static PID internal_id(const HierarchicalNSW& index, PID label) {
        return index.label_lookup_.at(label);
    }

    static void set_raw_cluster_id(HierarchicalNSW& index, PID label, PID value) {
        std::memcpy(
            index.get_clusterid_pt(index.label_lookup_.at(label)), &value, sizeof(PID)
        );
    }

    // Give internal slot i the label count - 1 - i, as parallel construction can
    // when threads take slots out of order.
    static void reverse_labels(HierarchicalNSW& index) {
        const size_t count = index.num_points();
        index.label_lookup_.clear();
        for (size_t i = 0; i < count; ++i) {
            const auto label = static_cast<PID>(count - 1 - i);
            index.set_external_label(static_cast<PID>(i), label);
            index.label_lookup_.emplace(label, static_cast<PID>(i));
        }
    }
};

namespace {

TEST(HnswPruningTest, ScoresNewAndExistingNeighborsWithTheSameQuery) {
    constexpr size_t kDim = 64;
    constexpr size_t kCount = 6;
    constexpr PID kIncoming = kCount - 1;
    std::vector<float> data(kCount * kDim, 2.0F);
    std::fill_n(data.begin(), kDim, 1.0F);
    // A sparse residual reconstructs with a much larger norm than its source.
    // Scoring its codes and using its reconstruction as a query differ sharply.
    std::fill_n(data.begin() + (kIncoming * kDim), kDim, 0.01F);
    data[kIncoming * kDim] = 8.0F;
    std::vector<float> centroid(kDim, 0.0F);
    std::vector<PID> clusters(kCount, 0);
    HierarchicalNSW index(kCount, kDim, 1, 2, 10);
    index.construct(1, centroid.data(), kCount, data.data(), clusters.data(), 1, false);

    // Set codes directly in the rotated domain to avoid random rotation changing
    // the asymmetry, then fill node 0's base-layer list to force pruning.
    for (PID id = 0; id < kCount; ++id) {
        HnswPruningTestAccess::quantize(index, id, data.data() + (id * kDim));
    }
    detail::quant_query().clear();
    const float existing_distance = HnswPruningTestAccess::distance(index, 1, 0);
    EXPECT_LT(HnswPruningTestAccess::distance(index, kIncoming, 0), existing_distance);
    EXPECT_GT(HnswPruningTestAccess::distance(index, 0, kIncoming), existing_distance);

    // The new point is closest when node 0 is the query. Its duplicate existing
    // neighbors are pruned; the reversed estimate instead retains an old point.
    EXPECT_EQ(
        HnswPruningTestAccess::connect(index, kIncoming), std::vector<PID>{kIncoming}
    );
}

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
    EXPECT_EQ(index.num_points(), kTotal);

    for (size_t i = 0; i < kAdded; ++i) {
        EXPECT_EQ(labels[i], kBuilt + i);
        EXPECT_EQ(index.cluster_id_of(labels[i]), cluster_ids[kBuilt + i]);
    }
    // ANN point IDs can vary with random rotation. Search should still return
    // valid results in the query's well-separated cluster.
    const auto results = index.search(
        fixture.data.data() + (kBuilt * AddFixture::kDim), kAdded, 1, kTotal, 1
    );
    ASSERT_EQ(results.size(), kAdded);
    for (size_t i = 0; i < kAdded; ++i) {
        ASSERT_EQ(results[i].size(), 1U);
        EXPECT_TRUE(std::isfinite(results[i][0].first));
        ASSERT_LT(results[i][0].second, kTotal);
        EXPECT_EQ(index.cluster_id_of(results[i][0].second), cluster_ids[kBuilt + i]);
    }
}

TEST(HnswResizeTest, RejectsUninitializedIndex) {
    HierarchicalNSW index;
    for (const size_t capacity : {10U, 0U}) {
        SCOPED_TRACE(capacity);
        try {
            index.resize(capacity);
            FAIL() << "Resizing an uninitialized index must fail";
        } catch (const std::logic_error& error) {
            EXPECT_STREQ(error.what(), "HNSW index must be initialized before resize");
        }
        EXPECT_EQ(index.max_elements(), 0U);
        EXPECT_EQ(index.num_points(), 0U);
    }
}

TEST(HnswResizeTest, AllowsResizeBeforeConstruction) {
    constexpr size_t kBuilt = 64;
    constexpr size_t kCapacity = 128;
    AddFixture fixture(kBuilt, 43);
    HierarchicalNSW index(kBuilt, AddFixture::kDim, 4, 8, 50);

    ASSERT_NO_THROW(index.resize(kCapacity));
    EXPECT_EQ(index.max_elements(), kCapacity);
    EXPECT_EQ(index.num_points(), 0U);
    ASSERT_NO_THROW(index.construct(
        1,
        fixture.centroid.data(),
        kBuilt,
        fixture.data.data(),
        fixture.cluster_ids.data(),
        1,
        false
    ));
    const auto results = index.search(fixture.data.data(), 8, 1, kBuilt, 1);
    ASSERT_EQ(results.size(), 8U);
    for (size_t i = 0; i < results.size(); ++i) {
        ASSERT_EQ(results[i].size(), 1U);
        EXPECT_EQ(results[i][0].second, i);
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
    ASSERT_NO_THROW(loaded.resize(kTotal + 8));
    EXPECT_EQ(loaded.max_elements(), kTotal + 8);
    const auto results = loaded.search(fixture.data.data(), kTotal, 1, kTotal, 1);
    for (size_t i = 0; i < kTotal; ++i) {
        EXPECT_EQ(results[i][0].second, i);
    }
    std::filesystem::remove(path);
}

// Labels must not depend on the order threads happen to finish in, and every
// added point must still be reachable.
TEST(HnswAddTest, ParallelAddKeepsLabelsAndTheGraph) {
    constexpr size_t kBuilt = 256;
    constexpr size_t kAdded = 256;
    constexpr size_t kTotal = kBuilt + kAdded;
    AddFixture fixture(kTotal, 53);

    HierarchicalNSW index(kTotal, AddFixture::kDim, 4, kTotal, kTotal);
    index.construct(
        1,
        fixture.centroid.data(),
        kBuilt,
        fixture.data.data(),
        fixture.cluster_ids.data(),
        8,
        false
    );
    const auto labels = index.add(
        fixture.data.data() + (kBuilt * AddFixture::kDim),
        kAdded,
        fixture.cluster_ids.data(),
        false,
        8
    );

    ASSERT_EQ(labels.size(), kAdded);
    for (size_t i = 0; i < kAdded; ++i) {
        EXPECT_EQ(labels[i], kBuilt + i);
    }
    EXPECT_EQ(index.num_points(), kTotal);

    const auto results = index.search(fixture.data.data(), kTotal, 1, kTotal, 1);
    for (size_t i = 0; i < kTotal; ++i) {
        EXPECT_EQ(results[i][0].second, i);
    }
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

TEST(HnswConstructionTest, RejectsTooSmallM) {
    for (const size_t M : {0U, 1U}) {
        SCOPED_TRACE(M);
        try {
            HierarchicalNSW index(8, 64, 4, M, 10);
            FAIL() << "M below 2 must be rejected";
        } catch (const std::invalid_argument& error) {
            EXPECT_STREQ(error.what(), "HNSW M must be at least 2");
        }
    }
    EXPECT_NO_THROW(HierarchicalNSW(8, 64, 4, 2, 10));
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

TEST_F(HnswSaveTest, RejectedRebuildPreservesConstructedAndLoadedIndexes) {
    // Multiple original centroids expose a rejected rebuild replacing the
    // centroid buffer with a smaller one while retaining the old cluster IDs.
    std::vector<float> centroids(2 * kDim, 0.0F);
    centroids[kDim] = 1.0F;
    for (size_t i = 0; i < kCount; ++i) {
        cluster_ids_[i] = i % 2;
    }
    HierarchicalNSW original(kCount, kDim, 4, 4, 10);
    original.construct(2, centroids.data(), kCount, data_.data(), cluster_ids_.data(), 1);
    original.save(path_.c_str());
    HierarchicalNSW loaded;
    loaded.load(path_.c_str());
    std::vector<PID> new_clusters(kCount, 0);
    for (auto* index : {&original, &loaded}) {
        const auto before = index->search(data_.data(), kCount, 2, kCount, 1);
        try {
            index->construct(
                1, centroid_.data(), kCount, data_.data(), new_clusters.data(), 1
            );
            FAIL() << "A second construction must be rejected";
        } catch (const std::logic_error& error) {
            EXPECT_STREQ(error.what(), "HNSW index is already constructed or loaded");
        }
        EXPECT_EQ(index->num_points(), kCount);
        EXPECT_EQ(index->num_clusters(), 2U);
        EXPECT_EQ(index->search(data_.data(), kCount, 2, kCount, 1), before);
        index->save(path_.c_str());
        HierarchicalNSW roundtrip;
        roundtrip.load(path_.c_str());
        EXPECT_EQ(roundtrip.search(data_.data(), kCount, 2, kCount, 1), before);
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

using Results = std::vector<std::vector<std::pair<float, PID>>>;

struct RemoveCase {
    MetricType metric;
    size_t bits;
};

std::vector<RemoveCase> remove_cases() {
    std::vector<RemoveCase> cases;
    for (const MetricType metric : {METRIC_L2, METRIC_IP}) {
        for (const size_t bits : {1U, 2U, 4U, 8U, 9U}) {
            cases.push_back({metric, bits});
        }
    }
    return cases;
}

// Constructs the first `count` points of `fixture` into an index that has room
// for `capacity`.
std::unique_ptr<HierarchicalNSW> build_for_remove(
    AddFixture& fixture,
    size_t count,
    size_t capacity,
    const RemoveCase& config,
    size_t num_threads = 1,
    size_t M = 8
) {
    auto index = std::make_unique<HierarchicalNSW>(
        capacity, AddFixture::kDim, config.bits, M, 50, 100, config.metric
    );
    index->construct(
        1,
        fixture.centroid.data(),
        count,
        fixture.data.data(),
        fixture.cluster_ids.data(),
        num_threads,
        false
    );
    return index;
}

// With k and ef at the point count, the search returns every point reachable
// from the entry point. HNSW does not promise that every point is reachable, so
// the tests compare against the same index without removals.
Results search_everything(
    HierarchicalNSW& index, const AddFixture& fixture, size_t queries
) {
    const size_t count = index.num_points();
    return index.search(fixture.data.data(), queries, count, count, 1);
}

// `before` without the removed labels, in the same order.
Results without_removed(const Results& before, const std::vector<bool>& removed) {
    Results kept(before.size());
    for (size_t q = 0; q < before.size(); ++q) {
        std::copy_if(
            before[q].begin(),
            before[q].end(),
            std::back_inserter(kept[q]),
            [&](const std::pair<float, PID>& hit) { return !removed[hit.second]; }
        );
    }
    return kept;
}

// A copy through save and load: the same rotator, codes and graph.
std::unique_ptr<HierarchicalNSW> copy_of(
    const HierarchicalNSW& index, const std::string& name
) {
    const std::string path = ::testing::TempDir() + name;
    index.save(path.c_str());
    auto copy = std::make_unique<HierarchicalNSW>();
    copy->load(path.c_str());
    std::remove(path.c_str());
    return copy;
}

std::vector<char> saved_bytes(const HierarchicalNSW& index, const std::string& path) {
    index.save(path.c_str());
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

TEST(HnswRemoveTest, ExcludesRemovedPointsAndKeepsSurvivors) {
    constexpr size_t kCount = 120;
    constexpr size_t kQueries = 8;
    for (const auto& config : remove_cases()) {
        SCOPED_TRACE(
            ::testing::Message()
            << "metric " << static_cast<int>(config.metric) << " bits " << config.bits
        );
        AddFixture fixture(kCount, 59);
        auto index = build_for_remove(fixture, kCount, kCount, config);
        const Results before = search_everything(*index, fixture, kQueries);

        // Every third point, the last one, and a repeated label.
        std::vector<PID> labels;
        std::vector<bool> removed(kCount, false);
        for (PID label = 0; label < kCount; label += 3) {
            labels.push_back(label);
            removed[label] = true;
        }
        labels.push_back(kCount - 1);
        removed[kCount - 1] = true;
        labels.push_back(0);
        const auto unique =
            static_cast<size_t>(std::count(removed.begin(), removed.end(), true));

        EXPECT_EQ(index->remove(labels.data(), labels.size()), unique);
        EXPECT_EQ(index->remove(labels.data(), labels.size()), 0U);
        EXPECT_EQ(index->num_points(), kCount);

        // Survivors keep their order and their distances.
        EXPECT_EQ(
            search_everything(*index, fixture, kQueries), without_removed(before, removed)
        );

        // A small ef still fills k with live points only.
        const auto top = index->search(fixture.data.data(), kQueries, 10, 10, 1);
        for (const auto& hits : top) {
            EXPECT_EQ(hits.size(), 10U);
            for (const auto& hit : hits) {
                EXPECT_FALSE(removed[hit.second]) << hit.second;
                EXPECT_TRUE(std::isfinite(hit.first));
            }
        }
    }
}

TEST(HnswRemoveTest, RemovingTheEntryPointKeepsSearchWorking) {
    constexpr size_t kCount = 160;
    for (const auto& config : remove_cases()) {
        SCOPED_TRACE(
            ::testing::Message()
            << "metric " << static_cast<int>(config.metric) << " bits " << config.bits
        );
        AddFixture fixture(kCount, 61);
        auto index = build_for_remove(fixture, kCount, kCount, config);
        const Results before = search_everything(*index, fixture, 1);
        const PID entry = HnswPruningTestAccess::entry_point_label(*index);
        std::vector<bool> removed(kCount, false);
        removed[entry] = true;
        ASSERT_EQ(index->remove(&entry, 1), 1U);

        // Every search starts at the removed entry point and must still return k
        // live points without it.
        const auto results = index->search(fixture.data.data(), kCount, 10, 10, 1);
        for (const auto& hits : results) {
            EXPECT_EQ(hits.size(), 10U);
            for (const auto& hit : hits) {
                EXPECT_NE(hit.second, entry);
            }
        }
        EXPECT_EQ(search_everything(*index, fixture, 1), without_removed(before, removed));
    }
}

TEST(HnswRemoveTest, SearchReachesLivePointsThroughRemovedOnes) {
    constexpr size_t kCount = 200;
    constexpr size_t kStride = 10;
    for (const auto& config : remove_cases()) {
        SCOPED_TRACE(
            ::testing::Message()
            << "metric " << static_cast<int>(config.metric) << " bits " << config.bits
        );
        AddFixture fixture(kCount, 67);
        auto index = build_for_remove(fixture, kCount, kCount, config, 1, 4);
        const Results before = search_everything(*index, fixture, kCount);

        // Keep one point in ten. Most paths between the survivors now pass through
        // removed points, which search must still walk through to reach every
        // survivor it reached before.
        std::vector<PID> labels;
        std::vector<bool> removed(kCount, false);
        for (PID label = 0; label < kCount; ++label) {
            if (label % kStride != 0) {
                labels.push_back(label);
                removed[label] = true;
            }
        }
        ASSERT_EQ(index->remove(labels.data(), labels.size()), labels.size());

        EXPECT_EQ(
            search_everything(*index, fixture, kCount), without_removed(before, removed)
        );
    }
}

TEST(HnswRemoveTest, UsesLabelsNotInternalIds) {
    constexpr size_t kCount = 256;
    const RemoveCase config{METRIC_L2, 4};
    AddFixture fixture(kCount, 71);
    auto index = build_for_remove(fixture, kCount, kCount, config);
    HnswPruningTestAccess::reverse_labels(*index);
    const Results before = search_everything(*index, fixture, 1);

    // Every fifth label below kCount / 2: no removed label shares a slot with
    // another removed label, so treating labels as slots would hide other points.
    std::vector<PID> labels;
    std::vector<bool> removed(kCount, false);
    for (PID label = 0; label < kCount / 2; label += 5) {
        labels.push_back(label);
        removed[label] = true;
    }
    ASSERT_EQ(index->remove(labels.data(), labels.size()), labels.size());

    EXPECT_EQ(search_everything(*index, fixture, 1), without_removed(before, removed));
}

TEST(HnswRemoveTest, SurvivesSaveLoadResizeAndLaterAdds) {
    constexpr size_t kBuilt = 96;
    constexpr size_t kAdded = 32;
    constexpr size_t kTotal = kBuilt + kAdded;
    const std::string path = ::testing::TempDir() + "rabitq_hnsw_remove_roundtrip.index";
    for (const auto& config : remove_cases()) {
        SCOPED_TRACE(
            ::testing::Message()
            << "metric " << static_cast<int>(config.metric) << " bits " << config.bits
        );
        AddFixture fixture(kTotal, 73);
        auto index = build_for_remove(fixture, kBuilt, kBuilt, config);
        const auto plain = copy_of(*index, "rabitq_hnsw_remove_plain.index");
        std::vector<PID> labels;
        std::vector<bool> removed(kTotal, false);
        for (PID label = 1; label < kBuilt; label += 4) {
            labels.push_back(label);
            removed[label] = true;
        }
        ASSERT_EQ(index->remove(labels.data(), labels.size()), labels.size());
        const Results expected = search_everything(*index, fixture, 4);

        index->save(path.c_str());
        HierarchicalNSW loaded;
        loaded.load(path.c_str());
        EXPECT_EQ(search_everything(loaded, fixture, 4), expected);
        EXPECT_EQ(loaded.remove(labels.data(), labels.size()), 0U);

        // Resizing copies the marks with the rest of the base layer.
        index->resize(kTotal);
        loaded.resize(kTotal);
        EXPECT_EQ(search_everything(loaded, fixture, 4), expected);

        // New points can be linked through removed ones. Only the two loaded
        // copies share insertion RNG state; save/load does not preserve it.
        const float* extra = fixture.data.data() + (kBuilt * AddFixture::kDim);
        plain->resize(kTotal);
        for (HierarchicalNSW* copy : {index.get(), &loaded, plain.get()}) {
            copy->add(extra, kAdded, fixture.cluster_ids.data());
        }
        const Results grown = search_everything(*index, fixture, kTotal);
        EXPECT_EQ(grown, without_removed(grown, removed));
        const auto reloaded = copy_of(*index, "rabitq_hnsw_remove_grown.index");
        EXPECT_EQ(search_everything(*reloaded, fixture, kTotal), grown);
        EXPECT_EQ(
            search_everything(loaded, fixture, kTotal),
            without_removed(search_everything(*plain, fixture, kTotal), removed)
        );
    }
    std::remove(path.c_str());
}

TEST(HnswRemoveTest, RejectsBadInputAndRemovesNothingOnFailure) {
    constexpr size_t kCount = 64;
    const PID one = 3;
    HierarchicalNSW unbuilt;
    try {
        unbuilt.remove(&one, 1);
        FAIL() << "Removing from an index that was never built must fail";
    } catch (const std::logic_error& error) {
        EXPECT_STREQ(
            error.what(), "HNSW index must be constructed or loaded before remove"
        );
    }
    HierarchicalNSW unconstructed(kCount, AddFixture::kDim, 4, 8, 50);
    EXPECT_THROW(unconstructed.remove(&one, 1), std::logic_error);

    AddFixture fixture(kCount, 79);
    auto index = build_for_remove(fixture, kCount, kCount, {METRIC_L2, 4});
    const std::string path = ::testing::TempDir() + "rabitq_hnsw_remove_rejects.index";
    const auto expected = saved_bytes(*index, path);

    const std::vector<PID> unknown{3, kCount};
    try {
        index->remove(unknown.data(), unknown.size());
        FAIL() << "An unknown label must be rejected";
    } catch (const std::invalid_argument& error) {
        EXPECT_STREQ(error.what(), "HNSW remove label is not in the index");
    }
    EXPECT_THROW(index->remove(nullptr, 1), std::invalid_argument);
    EXPECT_EQ(index->remove(nullptr, 0), 0U);
    EXPECT_EQ(saved_bytes(*index, path), expected);

    // Label 3 was valid in the rejected call, so it is still there to remove.
    EXPECT_EQ(index->remove(&one, 1), 1U);
    std::remove(path.c_str());
}

TEST(HnswRemoveTest, RemovingEveryPointReturnsNoResults) {
    constexpr size_t kBuilt = 64;
    constexpr size_t kAdded = 16;
    AddFixture fixture(kBuilt + kAdded, 83);
    auto index = build_for_remove(fixture, kBuilt, kBuilt, {METRIC_IP, 1});
    std::vector<PID> labels(kBuilt);
    for (PID label = 0; label < kBuilt; ++label) {
        labels[label] = label;
    }
    ASSERT_EQ(index->remove(labels.data(), labels.size()), kBuilt);
    for (const auto& hits : index->search(fixture.data.data(), 4, 10, 10, 1)) {
        EXPECT_TRUE(hits.empty());
    }
    for (const auto& hits : search_everything(*index, fixture, 4)) {
        EXPECT_TRUE(hits.empty());
    }

    // The graph of removed points still carries new ones.
    index->resize(kBuilt + kAdded);
    const float* extra = fixture.data.data() + (kBuilt * AddFixture::kDim);
    index->add(extra, kAdded, fixture.cluster_ids.data());
    const auto results = index->search(extra, kAdded, 1, kBuilt + kAdded, 1);
    for (size_t i = 0; i < kAdded; ++i) {
        ASSERT_EQ(results[i].size(), 1U);
        EXPECT_GE(results[i][0].second, kBuilt);
    }
}

TEST(HnswRemoveTest, LoadRejectsAnOutOfRangeClusterEvenWithTheMark) {
    constexpr size_t kCount = 32;
    AddFixture fixture(kCount, 89);
    auto index = build_for_remove(fixture, kCount, kCount, {METRIC_L2, 4});
    const PID one = 5;
    ASSERT_EQ(index->remove(&one, 1), 1U);

    const std::string path = ::testing::TempDir() + "rabitq_hnsw_remove_bad_cluster.index";
    index->save(path.c_str());
    HierarchicalNSW loaded;
    ASSERT_NO_THROW(loaded.load(path.c_str()));

    // Cluster 1 does not exist in a one-cluster index, marked or not.
    HnswPruningTestAccess::set_raw_cluster_id(*index, one, 0x80000001U);
    index->save(path.c_str());
    try {
        loaded.load(path.c_str());
        FAIL() << "A marked out-of-range cluster ID must be rejected";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "HNSW: invalid or truncated index file");
    }
    std::remove(path.c_str());
}

}  // namespace
}  // namespace rabitqlib::hnsw

TEST(HnswPaddingTest, TailDimensionsMatchEveryAvailableSearchBackend) {
    using namespace rabitqlib;
    using namespace rabitqlib::hnsw;
    constexpr size_t count = 24;
    using Search = decltype(&hnsw::detail::search_knn_generic);
    std::vector<Search> backends{hnsw::detail::search_knn};
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2())
        backends.push_back(hnsw::detail::search_knn_avx2);
    if (cpu::has_avx512_core())
        backends.push_back(hnsw::detail::search_knn_avx512_core);
    if (cpu::has_avx512_popcnt())
        backends.push_back(hnsw::detail::search_knn_avx512_popcnt);
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    if (cpu::has_neon())
        backends.push_back(hnsw::detail::search_knn_neon);
#endif
    for (size_t dim : {64U, 96U, 128U, 160U, 256U, 544U, 576U}) {
        std::mt19937 random(642);
        std::uniform_real_distribution<float> value(-1, 1);
        std::vector<float> data(count * dim), centroid(dim, 0), query(dim);
        for (auto& x : data)
            x = value(random);
        for (auto& x : query)
            x = value(random);
        for (auto metric : {METRIC_L2, METRIC_IP}) {
            for (size_t bits = 1; bits <= 9; ++bits) {
                SCOPED_TRACE(dim);
                SCOPED_TRACE(bits);
                SCOPED_TRACE(metric);
                std::vector<PID> clusters(count, 0);
                HierarchicalNSW index(count, dim, bits, 8, count, 42, metric);
                index.construct(
                    1, centroid.data(), count - 4, data.data(), clusters.data(), 1, true
                );
                // Insertion reconstructs stored sign/extra-bit tails before linking.
                index.add(data.data() + (count - 4) * dim, 4, clusters.data(), true, 1);
                const auto reference =
                    hnsw::detail::search_knn_generic(index, query.data(), 5, count);
                for (auto search : backends) {
                    auto expected = reference;
                    auto actual = search(index, query.data(), 5, count);
                    ASSERT_EQ(actual.size(), expected.size());
                    while (!expected.empty()) {
                        EXPECT_EQ(actual.top().second, expected.top().second);
                        EXPECT_NEAR(
                            actual.top().first,
                            expected.top().first,
                            2e-5F * std::max(1.0F, std::abs(expected.top().first))
                        );
                        actual.pop();
                        expected.pop();
                    }
                }
            }
        }
    }
}

TEST(HnswSearchTest, ConcurrentCallsKeepIndependentSearchWindows) {
    using namespace rabitqlib;
    constexpr size_t kCount = 129, kDim = 65, kQueries = 13, k = 5;
    std::mt19937 random(314);
    std::normal_distribution<float> value;
    std::vector<float> data(kCount * kDim), queries(kQueries * kDim), centroid(kDim, 0);
    for (auto& x : data)
        x = value(random);
    for (auto& x : queries)
        x = value(random);
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (size_t bits : {1U, 4U, 9U}) {
            SCOPED_TRACE(metric);
            SCOPED_TRACE(bits);
            std::vector<PID> clusters(kCount, 0);
            hnsw::HierarchicalNSW index(kCount, kDim, bits, 8, 64, 42, metric);
            index.construct(
                1, centroid.data(), kCount, data.data(), clusters.data(), 1, true
            );
            const auto narrow = index.search(queries.data(), kQueries, k, k, 1);
            const auto wide = index.search(queries.data(), kQueries, k, 96, 1);
            std::promise<void> start;
            const auto ready = start.get_future().share();
            const auto run = [&](size_t ef, size_t threads, const auto& expected) {
                ready.wait();
                for (size_t i = 0; i < 16; ++i) {
                    if (index.search(queries.data(), kQueries, k, ef, threads) != expected)
                        return false;
                }
                return true;
            };
            auto first = std::async(std::launch::async, [&] { return run(k, 1, narrow); });
            auto second = std::async(std::launch::async, [&] { return run(96, 2, wide); });
            start.set_value();
            EXPECT_TRUE(first.get());
            EXPECT_TRUE(second.get());
        }
    }
}
