#include "rabitqlib/index/ivf/initializer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace rabitqlib::ivf {
namespace {

TEST(CentroidL2SpaceTest, MatchesDispatchedDistanceForUnalignedInputsAndTails) {
    for (size_t dim : {1U, 7U, 8U, 15U, 16U, 17U, 63U, 64U, 129U}) {
        CentroidL2Space space(dim);
        std::vector<float> a(dim + 1), b(dim + 1);
        double expected = 0;
        for (size_t i = 1; i <= dim; ++i) {
            a[i] = static_cast<float>(i) / 8.0F;
            b[i] = static_cast<float>(i % 7) / 4.0F;
            const double delta = static_cast<double>(a[i]) - b[i];
            expected += delta * delta;
        }
        EXPECT_EQ(space.get_data_size(), dim * sizeof(float));
        const float actual =
            space.get_dist_func()(a.data() + 1, b.data() + 1, space.get_dist_func_param());
        EXPECT_FLOAT_EQ(actual, euclidean_sqr(a.data() + 1, b.data() + 1, dim));
        EXPECT_NEAR(actual, expected, 2e-6 * expected);
    }
}

TEST(HNSWInitializerTest, RoutesToExactCentroidsWithEuclideanDistances) {
    constexpr size_t kDim = 17;
    constexpr size_t kCount = 32;
    std::vector<float> centroids(kCount * kDim);
    for (size_t i = 0; i < kCount; ++i) {
        centroids[i * kDim] = static_cast<float>(i) * 2.0F;
    }
    HNSWInitializer initializer(kDim, kCount);
    initializer.add_vectors(centroids.data(), 1);
    std::array<float, kDim> query{};
    query[0] = 20.25F;
    std::vector<AnnCandidate<float>> candidates(3);
    initializer.centroids_distances(query.data(), candidates.size(), candidates);
    for (const auto& candidate : candidates) {
        EXPECT_TRUE(candidate.id == 9 || candidate.id == 10 || candidate.id == 11);
        EXPECT_FLOAT_EQ(
            candidate.distance, std::abs(query[0] - static_cast<float>(candidate.id) * 2.0F)
        );
    }
}

TEST(ParallelForTest, AutomaticThreadCountProcessesEveryItem) {
    std::atomic<size_t> calls{0};
    parallel_for(0, 100, 0, [&](size_t, size_t) { ++calls; });
    EXPECT_EQ(calls, 100U);
}

TEST(ParallelForTest, EmptyRangeDoesNotInvokeFunction) {
    size_t calls = 0;
    parallel_for(5, 5, 0, [&](size_t, size_t) { ++calls; });
    EXPECT_EQ(calls, 0U);
}

TEST(ParallelForTest, JoinsWorkersBeforeRethrowingFunctionException) {
    std::atomic<size_t> active{0};
    EXPECT_THROW(
        parallel_for(
            0,
            100,
            4,
            [&](size_t id, size_t) {
                ++active;
                --active;
                if (id == 0) {
                    throw std::runtime_error("parallel failure");
                }
            }
        ),
        std::runtime_error
    );
    EXPECT_EQ(active, 0U);
}

}  // namespace
}  // namespace rabitqlib::ivf
