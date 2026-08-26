#include "rabitqlib/index/hnsw/hnsw.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace rabitqlib::hnsw {
namespace {

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

}  // namespace
}  // namespace rabitqlib::hnsw
