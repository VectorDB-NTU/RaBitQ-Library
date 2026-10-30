#include "rabitqlib/index/ivf/initializer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <ios>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/detail/parallel_for.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"
#include "temp_file.hpp"

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

TEST(HNSWInitializerTest, ReportsSidecarOpenFailureAndCanRetry) {
    const std::string path = ::testing::TempDir() + "rabitq_hnsw_save_failure";
    const std::filesystem::path sidecar(path + ".hnsw");
    std::filesystem::create_directory(sidecar);
    const std::array<float, 4> centroids{0.0F, 0.0F, 3.0F, 4.0F};
    HNSWInitializer initializer(2, 2);
    initializer.add_vectors(centroids.data(), 1);
    std::ofstream ignored_output;
    try {
        initializer.save(ignored_output, path.c_str());
        FAIL() << "Saving the HNSW sidecar to a directory must fail";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "Cannot open HNSW index file for writing");
    }
    std::filesystem::remove(sidecar);

    initializer.save(ignored_output, path.c_str());
    HNSWInitializer loaded(2, 2);
    std::ifstream ignored_input;
    loaded.load(ignored_input, path.c_str());
    std::vector<AnnCandidate<float>> candidates(1);
    loaded.centroids_distances(centroids.data() + 2, 1, candidates);
    EXPECT_EQ(candidates[0].id, 1U);
    EXPECT_FLOAT_EQ(candidates[0].distance, 0.0F);
    std::filesystem::remove(sidecar);
}

TEST(HNSWInitializerTest, ReportsBufferedWriteFailure) {
    if (!std::filesystem::exists("/dev/full")) {
        GTEST_SKIP() << "Requires a device that rejects writes";
    }
    CentroidL2Space space(2);
    hnswlib::HierarchicalNSW<float> graph(&space, 2);
    const std::array<float, 2> point{3.0F, 4.0F};
    graph.addPoint(point.data(), 0);
    // This small index fits in the stream buffer; failure can occur at close().
    EXPECT_THROW(graph.saveIndex("/dev/full"), std::ios_base::failure);
    const auto result = graph.searchKnn(point.data(), 1);
    ASSERT_EQ(result.size(), 1U);
    EXPECT_EQ(result.top().second, 0U);
    EXPECT_FLOAT_EQ(result.top().first, 0.0F);
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

TEST(HNSWInitializerTest, ParallelInsertionPreservesRandomLevelDraws) {
    constexpr size_t kDim = 17;
    constexpr size_t kCount = 2048;
    constexpr size_t kSeed = 100;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> centroids(kCount * kDim);
    for (auto& value : centroids) {
        value = distribution(rng);
    }
    CentroidL2Space space(kDim);
    hnswlib::HierarchicalNSW<float> reference(&space, 1, 16, 100, kSeed);
    std::vector<int> expected_levels(kCount);
    for (auto& level : expected_levels) {
        level = reference.getRandomLevel(reference.mult_);
    }
    for (size_t threads : {1U, 4U, 8U}) {
        SCOPED_TRACE(threads);
        hnswlib::HierarchicalNSW<float> graph(&space, kCount, 16, 100, kSeed);
        index_detail::parallel_for(0, kCount, threads, [&](size_t row, size_t) {
            graph.addPoint(centroids.data() + row * kDim, row);
        });
        EXPECT_EQ(graph.cur_element_count.load(), kCount);
        // Registration assigns internal IDs and draws their levels in the same order.
        EXPECT_EQ(graph.element_levels_, expected_levels);
        EXPECT_EQ(graph.level_generator_, reference.level_generator_);
    }
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
    const rabitq_test::TempFile path_file("rabitq_initializer_labels");
    const std::string& path = path_file.path();
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
}

TEST(HNSWInitializerTest, ReloadsInnerProductGraphForRouting) {
    constexpr size_t kDim = 17;
    constexpr size_t kCount = 32;
    std::vector<float> centroids(kCount * kDim, 0.0F);
    centroids[0] = 0.9F;
    centroids[kDim] = 100.0F;
    const rabitq_test::TempFile path_file("rabitq_initializer_ip");
    const std::string& path = path_file.path();
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
}

TEST(HNSWInitializerTest, PerCallSearchWindowsPreserveDefaultAndLegacyResults) {
    constexpr size_t kDim = 17;
    constexpr size_t kCount = 1024;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> centroids(kCount * kDim);
    for (auto& value : centroids) {
        value = distribution(rng);
    }
    CentroidL2Space space(kDim);
    hnswlib::HierarchicalNSW<float> graph(&space, kCount, 16, 100);
    for (size_t i = 0; i < kCount; ++i) {
        graph.addPoint(centroids.data() + i * kDim, i);
    }
    struct EvenLabels : hnswlib::BaseFilterFunctor {
        bool operator()(hnswlib::labeltype label) override { return label % 2 == 0; }
    } even_labels;
    std::array<float, kDim> query{};
    for (auto* filter :
         {static_cast<hnswlib::BaseFilterFunctor*>(nullptr),
          static_cast<hnswlib::BaseFilterFunctor*>(&even_labels)}) {
        for (size_t ef : {1U, 64U, 768U, 1000U}) {
            graph.setEf(ef);
            auto expected = graph.searchKnn(query.data(), 10, filter);
            graph.setEf(11);
            auto actual = graph.searchKnnWithEf(query.data(), 10, ef, filter);
            EXPECT_EQ(graph.ef_, 11U);
            ASSERT_EQ(actual.size(), expected.size());
            while (!expected.empty()) {
                EXPECT_EQ(actual.top(), expected.top());
                actual.pop();
                expected.pop();
            }
        }
    }
}

TEST(HNSWInitializerTest, ConcurrentSearchesMatchSerialWithDifferentWindows) {
    constexpr size_t kDim = 17;
    constexpr size_t kCount = 1024;
    const std::array<size_t, 4> probes{1, 16, 400, 500};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> centroids(kCount * kDim);
    for (auto& value : centroids) {
        value = distribution(rng);
    }
    for (const auto metric : {METRIC_L2, METRIC_IP}) {
        HNSWInitializer initializer(kDim, kCount, metric);
        initializer.add_vectors(centroids.data(), 1);
        std::array<std::vector<AnnCandidate<float>>, 4> expected;
        for (size_t i = 0; i < probes.size(); ++i) {
            expected[i].resize(probes[i]);
            initializer.centroids_distances(
                centroids.data() + i * kDim, probes[i], expected[i]
            );
        }
        std::atomic<bool> invalid_result{false};
        std::vector<std::thread> threads;
        threads.reserve(4);
        for (size_t worker = 0; worker < 4; ++worker) {
            threads.emplace_back([&, worker] {
                for (size_t iteration = 0; iteration < 8; ++iteration) {
                    const size_t test = (worker + iteration) % probes.size();
                    std::vector<AnnCandidate<float>> candidates(probes[test]);
                    initializer.centroids_distances(
                        centroids.data() + test * kDim, probes[test], candidates
                    );
                    for (size_t i = 0; i < candidates.size(); ++i) {
                        if (candidates[i].id != expected[test][i].id ||
                            candidates[i].distance != expected[test][i].distance) {
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
}

TEST(ParallelForTest, JoinsWorkersBeforeRethrowingFunctionException) {
    if (rabitqlib::resolve_num_threads(2) < 2) {
        GTEST_SKIP() << "Requires two available workers";
    }
    std::promise<void> worker_started, release_worker;
    auto started = worker_started.get_future().share();
    auto release = release_worker.get_future();
    std::atomic<bool> worker_finished{false};
    auto result = std::async(std::launch::async, [&] {
        try {
            rabitqlib::index_detail::parallel_for(0, 2, 2, [&](size_t id, size_t) {
                if (id == 0) {
                    if (started.wait_for(std::chrono::seconds(5)) !=
                        std::future_status::ready) {
                        throw std::runtime_error("Second worker did not start");
                    }
                    throw std::runtime_error("parallel failure");
                }
                worker_started.set_value();
                release.wait();
                worker_finished = true;
            });
        } catch (const std::runtime_error& error) {
            EXPECT_STREQ(error.what(), "parallel failure");
            return worker_finished.load();
        }
        ADD_FAILURE() << "Worker exception was not propagated";
        return false;
    });
    EXPECT_EQ(started.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    // Keep one callback outstanding while the other throws. The caller must not
    // observe that exception until the outstanding callback has completed.
    EXPECT_EQ(result.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    release_worker.set_value();
    EXPECT_TRUE(result.get());
}

}  // namespace
}  // namespace rabitqlib::ivf
