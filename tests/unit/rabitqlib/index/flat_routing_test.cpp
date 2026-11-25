#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "ivf_search.hpp"
#include "rabitqlib/clustering/detail/kmeans.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::ivf {
namespace {

TEST(IvfFlatRoutingTest, TopNMatchesScalarAcrossTilesAndTies) {
    constexpr size_t kClusters = 513, kQueries = 129;
    std::mt19937 rng(218);
    std::uniform_real_distribution<float> random(-1, 1);
    detail::TopNAssignmentScratch scratch;
    for (size_t dim : {65U, 960U}) {
        for (float scale : {1.0F, 1.0e-8F, 1000.0F}) {
            std::vector<float> centroids(kClusters * dim), queries(kQueries * dim);
            // Large common offsets exercise cancellation in norm_sum - 2*dot.
            const float offset = scale == 1000.0F ? 100000.0F : 0.0F;
            std::generate(centroids.begin(), centroids.end(), [&] {
                return offset + scale * random(rng);
            });
            std::generate(queries.begin(), queries.end(), [&] {
                return offset + scale * random(rng);
            });
            for (size_t cluster = 1; cluster < kClusters; cluster += 2) {
                std::copy_n(centroids.data(), dim, centroids.data() + cluster * dim);
            }
            std::copy_n(centroids.data(), dim, queries.data());
            std::vector<double> norms(kClusters);
            for (size_t cluster = 0; cluster < kClusters; ++cluster) {
                norms[cluster] = clustering::detail::l2norm_sqr_f64(
                    centroids.data() + cluster * dim, dim
                );
            }
            for (auto metric : {METRIC_L2, METRIC_IP}) {
                FlatInitializer scalar(dim, kClusters, metric);
                scalar.add_vectors(centroids.data(), 1);
                for (size_t nprobe : {1U, 17U, 128U, 512U, 513U}) {
                    SCOPED_TRACE(
                        ::testing::Message()
                        << dim << ' ' << scale << ' ' << metric << ' ' << nprobe
                    );
                    std::vector<AnnCandidate<float>> actual(kQueries * nprobe), expected;
#pragma omp parallel num_threads(1)
                    {
                        omp_set_num_threads(1);
                        detail::exact_assign_topn(
                            queries.data(),
                            centroids.data(),
                            kQueries,
                            dim,
                            kClusters,
                            nprobe,
                            metric == METRIC_IP,
                            norms.data(),
                            actual.data(),
                            scratch
                        );
                    }
                    for (size_t row = 0; row < kQueries; ++row) {
                        scalar.centroids_distances(
                            queries.data() + row * dim, nprobe, expected
                        );
                        for (size_t i = 0; i < nprobe; ++i) {
                            ASSERT_EQ(actual[row * nprobe + i].id, expected[i].id);
                            ASSERT_EQ(
                                actual[row * nprobe + i].distance, expected[i].distance
                            );
                        }
                    }
                }
            }
        }
    }
}

TEST(IvfFlatRoutingTest, L2TiesUseReturnedFloatDistance) {
    constexpr size_t kDim = 64, kClusters = 257;
    std::vector<float> query(kDim), centroids(kDim * kClusters, 10.0F);
    std::fill_n(centroids.data(), 2 * kDim, 0.0F);
    centroids[0] = centroids[kDim] = 1.0F;
    centroids[1] = std::sqrt(std::numeric_limits<float>::epsilon());
    const float farther = euclidean_sqr(query.data(), centroids.data(), kDim);
    const float nearer = euclidean_sqr(query.data(), centroids.data() + kDim, kDim);
    ASSERT_GT(farther, nearer);
    ASSERT_EQ(std::sqrt(farther), std::sqrt(nearer));
    std::vector<double> norms(kClusters);
    for (size_t i = 0; i < kClusters; ++i) {
        norms[i] = clustering::detail::l2norm_sqr_f64(centroids.data() + i * kDim, kDim);
    }
    FlatInitializer scalar(kDim, kClusters);
    scalar.add_vectors(centroids.data(), 1);
    std::vector<AnnCandidate<float>> expected;
    scalar.centroids_distances(query.data(), 1, expected);
    EXPECT_EQ(expected[0].id, 0U);
    AnnCandidate<float> actual;
    detail::TopNAssignmentScratch scratch;
#pragma omp parallel num_threads(1)
    {
        omp_set_num_threads(1);
        detail::exact_assign_topn(
            query.data(),
            centroids.data(),
            1,
            kDim,
            kClusters,
            1,
            false,
            norms.data(),
            &actual,
            scratch
        );
    }
    EXPECT_EQ(actual.id, 0U);
    EXPECT_EQ(actual.distance, expected[0].distance);
}

TEST(IvfFlatRoutingTest, CommonL2OffsetRetainsTiesAndFindsLateCentroids) {
    constexpr size_t kDim = 64, kClusters = 515, kProbes = 2;
    std::vector<float> query(kDim, 1024.0F), centroids(kDim * kClusters, 1024.0F);
    std::vector<double> norms(kClusters);
    for (size_t id = 0; id < kClusters; ++id) {
        float* centroid = centroids.data() + id * kDim;
        centroid[0] += 1.0F;
    }
    // IDs 257 and 258 tie across the end of the first 256-centroid screening
    // block. The closest centroid is the very last one in the direct scan.
    centroids[257 * kDim] = centroids[258 * kDim] = 1024.5F;
    centroids[(kClusters - 1) * kDim] = 1024.25F;
    for (size_t id = 0; id < kClusters; ++id) {
        norms[id] = clustering::detail::l2norm_sqr_f64(centroids.data() + id * kDim, kDim);
    }
    std::array<AnnCandidate<float>, kProbes> actual{};
    detail::TopNAssignmentScratch scratch;
#pragma omp parallel num_threads(1)
    {
        omp_set_num_threads(1);
        detail::exact_assign_topn(
            query.data(),
            centroids.data(),
            1,
            kDim,
            kClusters,
            kProbes,
            false,
            norms.data(),
            actual.data(),
            scratch
        );
    }
    EXPECT_EQ(actual[0].id, kClusters - 1);
    EXPECT_EQ(actual[0].distance, 0.25F);
    EXPECT_EQ(actual[1].id, 257U);
    EXPECT_EQ(actual[1].distance, 0.5F);
}

TEST(IvfFlatRoutingTest, BatchSearchStreamsBlocksAndPreservesRemovedPadding) {
    constexpr size_t kCount = 577, kDim = 65, kClusters = 513, kQueries = 513;
    std::mt19937 rng(218);
    std::uniform_real_distribution<float> random(-1, 1);
    std::vector<float> data(kCount * kDim), centroids(kClusters * kDim);
    std::vector<float> queries(kQueries * kDim);
    std::generate(data.begin(), data.end(), [&] { return random(rng); });
    std::generate(centroids.begin(), centroids.end(), [&] { return random(rng); });
    std::generate(queries.begin(), queries.end(), [&] { return random(rng); });
    std::fill_n(queries.data(), kDim, 0.0F);
    for (size_t i = 1; i < kClusters; i += 2) {
        std::copy_n(centroids.data(), kDim, centroids.data() + i * kDim);
    }
    std::vector<PID> labels(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        labels[i] = static_cast<PID>(i % (kClusters - 1));
    }
    for (auto metric : {METRIC_L2, METRIC_IP, METRIC_COSINE}) {
        for (size_t bits : {1U, 5U, 32U}) {
            IVF index(kCount, kDim, kClusters, bits, metric);
            index.construct(data.data(), centroids.data(), labels.data(), false, 1);
            std::vector<PID> removed{0, 17, 255, 511};
            ASSERT_EQ(index.remove(removed.data(), removed.size()), removed.size());
            for (size_t nprobe : {1U, 17U, 512U, 999U}) {
                const size_t k = nprobe == 17 ? kCount + 3 : 11;
                std::vector<PID> expected(kQueries * k), actual(kQueries * k);
                std::vector<float> expected_distances(kQueries * k),
                    distances(kQueries * k);
                for (bool hacc : {false, true}) {
                    for (size_t row = 0; row < kQueries; ++row) {
                        index.search(
                            queries.data() + row * kDim,
                            k,
                            nprobe,
                            expected.data() + row * k,
                            expected_distances.data() + row * k,
                            hacc
                        );
                    }
                    for (size_t threads : {1U, 4U}) {
                        const int previous_threads = omp_get_max_threads();
                        index.search_batch(
                            queries.data(),
                            kQueries,
                            k,
                            nprobe,
                            actual.data(),
                            distances.data(),
                            hacc,
                            threads
                        );
                        EXPECT_EQ(omp_get_max_threads(), previous_threads);
                        EXPECT_EQ(actual, expected);
                        EXPECT_EQ(distances, expected_distances);
                        index.search_batch(
                            queries.data(),
                            kQueries,
                            k,
                            nprobe,
                            actual.data(),
                            nullptr,
                            hacc,
                            threads
                        );
                        EXPECT_EQ(actual, expected);
                    }
                }
            }
            removed.resize(kCount);
            std::iota(removed.begin(), removed.end(), PID{0});
            index.remove(removed.data(), removed.size());
            std::vector<PID> actual(kQueries);
            std::vector<float> distances(kQueries);
            index.search_batch(
                queries.data(), kQueries, 1, 17, actual.data(), distances.data(), false, 4
            );
            EXPECT_TRUE(std::all_of(actual.begin(), actual.end(), [](PID id) {
                return id == kPidMax;
            }));
            EXPECT_TRUE(std::all_of(distances.begin(), distances.end(), [](float distance) {
                return std::isinf(distance);
            }));
        }
    }
}

}  // namespace
}  // namespace rabitqlib::ivf
