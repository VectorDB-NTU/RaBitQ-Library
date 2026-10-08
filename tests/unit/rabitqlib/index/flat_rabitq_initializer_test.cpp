#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <iterator>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"

namespace rabitqlib::ivf {
namespace {
TEST(IvfInitializerTest, ExplicitChoiceOverridesAutomaticThreshold) {
    const std::array<std::pair<size_t, InitializerType>, 6> cases{{
        {4999, InitializerType::Flat},
        {5000, InitializerType::FlatRaBitQ},
        {19999, InitializerType::FlatRaBitQ},
        {20000, InitializerType::FlatRaBitQ},
        {59999, InitializerType::FlatRaBitQ},
        {60000, InitializerType::HNSW},
    }};
    for (const auto& [clusters, expected] : cases) {
        IVF automatic(1, 64, clusters, 1);
        EXPECT_EQ(automatic.initializer_type(), expected);
        for (auto type :
             {InitializerType::Flat, InitializerType::FlatRaBitQ, InitializerType::HNSW}) {
            IVF explicit_type(
                1, 64, clusters, 1, METRIC_L2, RotatorType::FhtKacRotator, type
            );
            EXPECT_EQ(explicit_type.initializer_type(), type);
        }
    }
    EXPECT_THROW(
        (
            IVF(1,
                64,
                1,
                1,
                METRIC_L2,
                RotatorType::FhtKacRotator,
                static_cast<InitializerType>(99))
        ),
        std::invalid_argument
    );
}

TEST(IvfInitializerTest, MatrixRotationAndExplicitRoutingRoundTrip) {
    constexpr size_t kDim = 17, kCount = 3;
    std::vector<float> data(kDim * kCount, 0.0F);
    data[0] = 1;
    data[kDim + 1] = 1;
    data[kDim * 2 + 2] = 1;
    const std::array<PID, kCount> labels{0, 1, 2};
    const std::string path = ::testing::TempDir() + "ivf_matrix_flat_rabitq.index";
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        IVF index(
            kCount,
            kDim,
            kCount,
            32,
            metric,
            RotatorType::MatrixRotator,
            InitializerType::FlatRaBitQ
        );
        index.construct(data.data(), data.data(), labels.data(), false, 1);
        index.save(path.c_str());
        IVF loaded;
        loaded.load(path.c_str());
        EXPECT_EQ(loaded.initializer_type(), InitializerType::FlatRaBitQ);
        for (size_t i = 0; i < kCount; ++i) {
            PID id = kPidMax;
            float distance = 0;
            loaded.search(data.data() + i * kDim, 1, 1, &id, &distance);
            EXPECT_EQ(id, i);
            EXPECT_NEAR(distance, 0, 1e-5);
        }
    }
    std::remove(path.c_str());
}

TEST(IvfInitializerTest, ReloadChangesRoutingAndFailedLoadPreservesExistingIndex) {
    constexpr size_t kDim = 65, kCount = 33, kTopk = 5;
    std::mt19937 generator(9923);
    std::normal_distribution<float> normal;
    std::vector<float> data(kCount * kDim);
    std::vector<PID> labels(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        labels[i] = static_cast<PID>(i);
        for (size_t j = 0; j < kDim; ++j)
            data[i * kDim + j] = normal(generator);
    }
    const std::string path = ::testing::TempDir() + "ivf_initializer_reload.index";
    IVF target;
    for (auto type :
         {InitializerType::Flat,
          InitializerType::FlatRaBitQ,
          InitializerType::HNSW,
          InitializerType::Flat}) {
        IVF source(kCount, kDim, kCount, 5, METRIC_L2, RotatorType::FhtKacRotator, type);
        source.construct(data.data(), data.data(), labels.data(), false, 1);
        std::array<PID, kTopk> expected_ids{}, actual_ids{};
        std::array<float, kTopk> expected_distances{}, actual_distances{};
        source.search(
            data.data(), kTopk, 7, expected_ids.data(), expected_distances.data()
        );
        source.save(path.c_str());
        target.load(path.c_str());
        EXPECT_EQ(target.initializer_type(), type);
        target.search(data.data(), kTopk, 7, actual_ids.data(), actual_distances.data());
        EXPECT_EQ(actual_ids, expected_ids);
        EXPECT_EQ(actual_distances, expected_distances);
        {
            std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
            // A persisted v2 file must contain a concrete initializer, never Auto.
            file.seekp(24);
            const uint32_t invalid = 0;
            file.write(reinterpret_cast<const char*>(&invalid), sizeof(invalid));
        }
        EXPECT_THROW(target.load(path.c_str()), std::runtime_error);
        EXPECT_EQ(target.initializer_type(), type);
        target.search(data.data(), kTopk, 7, actual_ids.data(), actual_distances.data());
        EXPECT_EQ(actual_ids, expected_ids);
        EXPECT_EQ(actual_distances, expected_distances);
    }
    std::remove(path.c_str());
    std::remove((path + ".hnsw").c_str());
}

TEST(FlatRaBitQInitializerTest, TailBatchesMetricsAndExactReturnedDistances) {
    std::mt19937 generator(141);
    std::normal_distribution<float> normal;
    for (size_t dim : {32U, 64U, 96U, 1056U}) {
        for (size_t count : {1U, 31U, 32U, 33U, 65U}) {
            std::vector<float> centroids(count * dim);
            std::generate(centroids.begin(), centroids.end(), [&] {
                return normal(generator);
            });
            // Unit vectors make self-query the winner under both metrics.
            for (size_t i = 0; i < count; ++i) {
                const float norm = std::sqrt(l2norm_sqr(centroids.data() + i * dim, dim));
                for (size_t j = 0; j < dim; ++j)
                    centroids[i * dim + j] /= norm;
            }
            for (auto metric : {METRIC_L2, METRIC_IP}) {
                FlatRaBitQInitializer index(dim, count, metric);
                index.add_vectors(centroids.data(), 2);
                const float* query = centroids.data() + (count - 1) * dim;
                for (size_t nprobe : {0U, 1U, 7U, 99U}) {
                    std::vector<AnnCandidate<float>> result;
                    index.centroids_distances(query, nprobe, result);
                    ASSERT_EQ(result.size(), std::min(nprobe, count));
                    if (result.empty())
                        continue;
                    EXPECT_EQ(result[0].id, count - 1);
                    EXPECT_TRUE(std::is_sorted(result.begin(), result.end()));
                    std::vector<bool> seen(count);
                    for (const auto& candidate : result) {
                        ASSERT_LT(candidate.id, count);
                        EXPECT_FALSE(seen[candidate.id]);
                        seen[candidate.id] = true;
                        const float exact =
                            metric == METRIC_L2
                                ? std::sqrt(euclidean_sqr(
                                      query, index.centroid(candidate.id), dim
                                  ))
                                : dot_product_dis(query, index.centroid(candidate.id), dim);
                        EXPECT_FLOAT_EQ(candidate.distance, exact);
                    }
                }
            }
        }
    }
}

TEST(FlatRaBitQInitializerTest, ZeroResidualsAndInnerProductMagnitude) {
    constexpr size_t kDim = 96, kCount = 33;
    std::vector<float> centroids(kDim * kCount, 0.0F), query(kDim, 0.0F);
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        FlatRaBitQInitializer index(kDim, kCount, metric);
        index.add_vectors(centroids.data(), 1);
        std::vector<AnnCandidate<float>> result;
        index.centroids_distances(query.data(), 7, result);
        ASSERT_EQ(result.size(), 7U);
        for (const auto& item : result)
            EXPECT_FLOAT_EQ(item.distance, metric == METRIC_L2 ? 0 : 1);
    }
    for (size_t i = 0; i < kCount; ++i)
        centroids[i * kDim] = 0.5F;
    centroids[17 * kDim] = 100.0F;
    query[0] = 1.0F;
    FlatRaBitQInitializer index(kDim, kCount, METRIC_IP);
    index.add_vectors(centroids.data(), 1);
    std::vector<AnnCandidate<float>> result;
    index.centroids_distances(query.data(), 1, result);
    ASSERT_EQ(result.size(), 1U);
    EXPECT_EQ(result[0].id, 17U);
    EXPECT_FLOAT_EQ(result[0].distance, -99.0F);
}

TEST(FlatRaBitQInitializerTest, PersistsCodesAndSupportsConcurrentSearch) {
    constexpr size_t kDim = 96, kCount = 97;
    std::mt19937 generator(731);
    std::normal_distribution<float> normal;
    std::vector<float> centroids(kDim * kCount), queries(8 * kDim);
    std::generate(centroids.begin(), centroids.end(), [&] { return normal(generator); });
    std::generate(queries.begin(), queries.end(), [&] { return normal(generator); });
    const std::string path = ::testing::TempDir() + "flat_rabitq_initializer.bin";
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        FlatRaBitQInitializer index(kDim, kCount, metric), loaded(kDim, kCount, metric);
        index.add_vectors(centroids.data(), 2);
        {
            std::ofstream out(path, std::ios::binary);
            index.save(out, nullptr);
        }
        {
            std::ifstream input(path, std::ios::binary);
            loaded.load(input, nullptr);
        }
        for (size_t i = 0; i < kCount; ++i) {
            EXPECT_TRUE(
                std::equal(index.centroid(i), index.centroid(i) + kDim, loaded.centroid(i))
            );
        }
        std::vector<std::vector<AnnCandidate<float>>> expected(8), actual(8);
        std::vector<std::thread> threads;
        for (size_t i = 0; i < 8; ++i) {
            index.centroids_distances(queries.data() + i * kDim, 7, expected[i]);
            threads.emplace_back([&, i] {
                loaded.centroids_distances(queries.data() + i * kDim, 7, actual[i]);
            });
        }
        for (auto& thread : threads)
            thread.join();
        for (size_t i = 0; i < 8; ++i) {
            ASSERT_EQ(actual[i].size(), expected[i].size());
            for (size_t j = 0; j < actual[i].size(); ++j) {
                EXPECT_EQ(actual[i][j].id, expected[i][j].id);
                EXPECT_FLOAT_EQ(actual[i][j].distance, expected[i][j].distance);
            }
        }
        std::ifstream before(path, std::ios::binary);
        const std::string bytes{std::istreambuf_iterator<char>(before), {}};
        before.close();
        {
            std::ofstream out(path, std::ios::binary);
            loaded.save(out, nullptr);
        }
        std::ifstream after(path, std::ios::binary);
        EXPECT_EQ((std::string{std::istreambuf_iterator<char>(after), {}}), bytes);
    }
    std::remove(path.c_str());
}
}  // namespace
}  // namespace rabitqlib::ivf
