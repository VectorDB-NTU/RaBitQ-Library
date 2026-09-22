#include "rabitqlib/index/ivf/initializer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace rabitqlib::ivf {
namespace {

TEST(HNSWInitializerTest, PersistsUtf8SidecarPaths) {
    const auto directory = std::filesystem::temp_directory_path() /
                           std::filesystem::u8path(u8"rabitq_\u6d4b\u8bd5_\U0001f680");
    std::filesystem::create_directories(directory);
    const auto path = (directory / std::filesystem::u8path(u8"\u7d22\u5f15")).u8string();
    const std::array<float, 4> centroids{0.0F, 0.0F, 3.0F, 4.0F};
    HNSWInitializer initializer(2, 2);
    initializer.add_vectors(centroids.data(), 1);
    std::ofstream ignored_output;
    initializer.save(ignored_output, path.c_str());
    EXPECT_TRUE(std::filesystem::is_regular_file(std::filesystem::u8path(path + ".hnsw")));
    HNSWInitializer loaded(2, 2);
    std::ifstream ignored_input;
    loaded.load(ignored_input, path.c_str());
    std::vector<AnnCandidate<float>> candidates(1);
    loaded.centroids_distances(centroids.data() + 2, 1, candidates);
    EXPECT_EQ(candidates[0].id, 1U);
    EXPECT_FLOAT_EQ(candidates[0].distance, 0.0F);
    std::filesystem::remove(std::filesystem::u8path(path + ".hnsw"));
    std::filesystem::remove(directory);
}

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

TEST(HNSWInitializerTest, RoutesByInnerProductWithoutTakingSquareRoot) {
    constexpr size_t kDim = 17;
    constexpr size_t kCount = 32;
    std::vector<float> centroids(kCount * kDim, 0.0F);
    centroids[0] = 0.9F;
    centroids[kDim] = 100.0F;
    HNSWInitializer initializer(kDim, kCount, METRIC_IP);
    initializer.add_vectors(centroids.data(), 4);
    std::array<float, kDim> query{};
    query[0] = 1.0F;
    std::vector<AnnCandidate<float>> candidates(1);
    initializer.centroids_distances(query.data(), 1, candidates);
    EXPECT_EQ(candidates[0].id, 1U);
    EXPECT_FLOAT_EQ(candidates[0].distance, -99.0F);
}

TEST(HNSWInitializerTest, ParallelInsertionPreservesCentroidLabelsAfterReload) {
    constexpr size_t kDim = 17;
    constexpr size_t kCount = 1024;
    std::vector<float> centroids(kCount * kDim);
    for (size_t label = 0; label < kCount; ++label) {
        for (size_t dim = 0; dim < kDim; ++dim) {
            centroids[label * kDim + dim] = static_cast<float>(label * kDim + dim);
        }
    }
    const std::string path = ::testing::TempDir() + "rabitq_initializer_labels";
    HNSWInitializer initializer(kDim, kCount);
    initializer.add_vectors(centroids.data(), 4);
    std::ofstream ignored_output;
    initializer.save(ignored_output, path.c_str());

    HNSWInitializer loaded(kDim, kCount);
    std::ifstream ignored_input;
    loaded.load(ignored_input, path.c_str());
    for (size_t label = 0; label < kCount; ++label) {
        for (size_t dim = 0; dim < kDim; ++dim) {
            EXPECT_FLOAT_EQ(
                initializer.centroid(static_cast<PID>(label))[dim],
                centroids[label * kDim + dim]
            );
            EXPECT_FLOAT_EQ(
                loaded.centroid(static_cast<PID>(label))[dim], centroids[label * kDim + dim]
            );
        }
    }
    std::remove((path + ".hnsw").c_str());
}

TEST(HNSWInitializerTest, ReloadsInnerProductGraphForRouting) {
    constexpr size_t kDim = 17;
    constexpr size_t kCount = 32;
    std::vector<float> centroids(kCount * kDim, 0.0F);
    centroids[0] = 0.9F;
    centroids[kDim] = 100.0F;
    const std::string path = ::testing::TempDir() + "rabitq_initializer_ip";
    HNSWInitializer initializer(kDim, kCount, METRIC_IP);
    initializer.add_vectors(centroids.data(), 4);
    std::ofstream ignored_output;
    initializer.save(ignored_output, path.c_str());

    HNSWInitializer loaded(kDim, kCount, METRIC_IP);
    std::ifstream ignored_input;
    loaded.load(ignored_input, path.c_str());
    std::array<float, kDim> query{};
    query[0] = 1.0F;
    std::vector<AnnCandidate<float>> candidates(1);
    loaded.centroids_distances(query.data(), 1, candidates);
    EXPECT_EQ(candidates[0].id, 1U);
    EXPECT_FLOAT_EQ(candidates[0].distance, -99.0F);
    std::remove((path + ".hnsw").c_str());
}

TEST(HNSWInitializerTest, ConcurrentSearchesKeepRoutingStateSynchronized) {
    constexpr size_t kDim = 17;
    constexpr size_t kCount = 32;
    std::vector<float> centroids(kCount * kDim, 0.0F);
    for (size_t label = 0; label < kCount; ++label) {
        centroids[label * kDim] = static_cast<float>(label);
    }
    HNSWInitializer initializer(kDim, kCount);
    initializer.add_vectors(centroids.data(), 4);
    std::array<float, kDim> query{};
    query[0] = 15.0F;
    std::atomic<bool> invalid_result{false};
    std::vector<std::thread> threads;
    for (size_t worker = 0; worker < 4; ++worker) {
        threads.emplace_back([&, worker] {
            for (size_t iteration = 0; iteration < 25; ++iteration) {
                const size_t nprobe = 1 + (worker + iteration) % 4;
                std::vector<AnnCandidate<float>> candidates(nprobe);
                initializer.centroids_distances(query.data(), nprobe, candidates);
                for (const auto& candidate : candidates) {
                    if (candidate.id >= kCount || !std::isfinite(candidate.distance)) {
                        invalid_result = true;
                    }
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_FALSE(invalid_result);
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
