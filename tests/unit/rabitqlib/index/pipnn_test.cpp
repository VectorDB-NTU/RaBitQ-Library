#include "rabitqlib/index/symqg/detail/pipnn.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace rabitqlib::symqg::detail {
namespace {
TEST(PipnnTest, DenseDistancesMatchReference) {
    constexpr size_t kCount = 1024, kDim = 65;
    std::vector<float> data(kCount * kDim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.17F);
    }
    pipnn_impl::ScratchPool scratch(kCount, kDim, 2);
    // Reuse each worker's slice with changing matrix shapes, including the
    // largest leaf, and check both distance conventions against scalar doubles.
    for (size_t worker : {0U, 1U}) {
        auto work = scratch.worker(worker);
        for (size_t count : {33U, 1024U, 257U}) {
            pipnn_impl::Bucket ids(count);
            std::iota(ids.begin(), ids.end(), PID{0});
            pipnn_impl::gather(data.data(), kDim, ids, work);
            for (auto metric : {METRIC_L2, METRIC_IP}) {
                pipnn_impl::pairwise(work, count, kDim, metric);
                for (size_t i = 0; i < count; ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        double expected = 0;
                        for (size_t d = 0; d < kDim; ++d) {
                            const double a = data[i * kDim + d], b = data[j * kDim + d];
                            expected += metric == METRIC_L2 ? (a - b) * (a - b) : -a * b;
                        }
                        EXPECT_NEAR(work.distances[i * count + j], expected, 5e-5);
                    }
                }
            }
        }
    }
}

TEST(PipnnTest, LocalSeedRetainsNearestNeighborsAndValidIds) {
    constexpr size_t kCount = 97, kDim = 65;
    std::vector<float> data(kCount * kDim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.13F);
    }
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        const auto seed = build_initial_graph(data.data(), kCount, kDim, 32, metric, 2);
        ASSERT_EQ(seed.offsets.size(), kCount + 1);
        ASSERT_EQ(seed.offsets.front(), 0U);
        ASSERT_EQ(seed.offsets.back(), seed.neighbors.size());
        for (size_t i = 0; i < kCount; ++i) {
            std::vector<PID> row(
                seed.neighbors.begin() + static_cast<ptrdiff_t>(seed.offsets[i]),
                seed.neighbors.begin() + static_cast<ptrdiff_t>(seed.offsets[i + 1])
            );
            ASSERT_FALSE(row.empty());
            ASSERT_LE(row.size(), 32U);
            std::sort(row.begin(), row.end());
            EXPECT_EQ(std::unique(row.begin(), row.end()), row.end());
            EXPECT_EQ(std::find(row.begin(), row.end(), i), row.end());
            EXPECT_LT(row.back(), kCount);
            float best = std::numeric_limits<float>::max();
            PID nearest = 0;
            for (PID j = 0; j < kCount; ++j) {
                if (i == j) {
                    continue;
                }
                const float distance =
                    metric == METRIC_L2
                        ? euclidean_sqr<float>(
                              data.data() + i * kDim, data.data() + j * kDim, kDim
                          )
                        : dot_product_dis<float>(
                              data.data() + i * kDim, data.data() + j * kDim, kDim
                          );
                if (distance < best) {
                    best = distance;
                    nearest = j;
                }
            }
            EXPECT_NE(std::find(row.begin(), row.end(), nearest), row.end());
        }
    }
}

TEST(PipnnTest, OverlappingPartitionsTerminateForDuplicateVectors) {
    constexpr size_t kCount = 1100, kDim = 7;
    std::vector<float> data(kCount * kDim, 0.5F);
    pipnn_impl::ScratchPool scratch(kCount, kDim, 2);
    auto leaves = pipnn_impl::cluster(data.data(), kCount, kDim, METRIC_L2, 2, scratch);
    std::vector<size_t> memberships(kCount, 0);
    for (auto& leaf : leaves) {
        EXPECT_LE(leaf.size(), pipnn_impl::kLeafSize);
        std::sort(leaf.begin(), leaf.end());
        EXPECT_EQ(std::unique(leaf.begin(), leaf.end()), leaf.end());
        for (PID id : leaf) {
            ASSERT_LT(id, kCount);
            ++memberships[id];
        }
    }
    for (size_t membership : memberships) {
        EXPECT_GE(membership, 1U);
        EXPECT_LE(membership, 30U);
    }
}

TEST(PipnnTest, RejectsInvalidConfiguration) {
    std::vector<float> data(33 * 7, 0);
    EXPECT_THROW(build_initial_graph(nullptr, 33, 7, 32), std::invalid_argument);
    EXPECT_THROW(build_initial_graph(data.data(), 33, 0, 32), std::invalid_argument);
    EXPECT_THROW(build_initial_graph(data.data(), 33, 7, 16), std::invalid_argument);
    EXPECT_THROW(build_initial_graph(data.data(), 32, 7, 32), std::invalid_argument);
}
}  // namespace
}  // namespace rabitqlib::symqg::detail
