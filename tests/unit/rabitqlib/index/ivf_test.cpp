#include "rabitqlib/index/ivf/ivf.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ivf_search.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/types.hpp"
#include "temp_file.hpp"

namespace rabitqlib::ivf {

// Outside the anonymous namespace so it matches the friend declaration in ivf.hpp.
struct IvfTestAccess {
    static Cluster cluster_view(const IVF& index, size_t i) { return index.clusters_[i]; }

    static bool has_spare_capacity(const IVF& index, size_t i) {
        const auto& cs = index.cluster_storage_[i];
        return cs.batch_storage.capacity() > cs.batch_storage.size() ||
               cs.ex_storage.capacity() > cs.ex_storage.size() ||
               cs.raw_storage.capacity() > cs.raw_storage.size() ||
               cs.id_storage.capacity() > cs.id_storage.size();
    }

    static std::unique_ptr<Rotator<float>>& rotator(IVF& index) { return index.rotator_; }
};

namespace {

TEST(IvfSearchTest, BatchSearchMatchesIndividualQueries) {
    constexpr size_t kCount = 97, kDim = 65, kNumClusters = 4, kNumQueries = 13, kTopK = 7;
    std::mt19937 rng(73);
    std::uniform_real_distribution<float> random(-1, 1);
    std::vector<float> data(kCount * kDim), centroids(kNumClusters * kDim);
    std::vector<float> queries(kNumQueries * kDim);
    std::vector<PID> labels(kCount);
    std::generate(data.begin(), data.end(), [&] { return random(rng); });
    std::generate(centroids.begin(), centroids.end(), [&] { return random(rng); });
    std::generate(queries.begin(), queries.end(), [&] { return random(rng); });
    for (size_t i = 0; i < kCount; ++i) {
        labels[i] = static_cast<PID>(i % (kNumClusters - 1));
    }
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (size_t bits : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 32U}) {
            IVF index(kCount, kDim, kNumClusters, bits, metric);
            EXPECT_EQ(index.padded_dim(), 96U);
            index.construct(data.data(), centroids.data(), labels.data(), false, 1);
            std::array<PID, kNumQueries * kTopK> expected{}, actual{}, ids_only{};
            std::array<float, kNumQueries * kTopK> expected_distances{}, actual_distances{};
            for (size_t nprobe : {1U, 99U}) {
                for (std::optional<bool> hacc :
                     {std::optional<bool>{},
                      std::optional<bool>(false),
                      std::optional<bool>(true)}) {
                    for (size_t i = 0; i < kNumQueries; ++i) {
                        index.search(
                            queries.data() + i * kDim,
                            kTopK,
                            nprobe,
                            expected.data() + i * kTopK,
                            expected_distances.data() + i * kTopK,
                            hacc.value_or(bits >= 4 && bits <= 9)
                        );
                    }
                    for (size_t threads : {1U, 4U}) {
                        index.search_batch(
                            queries.data(),
                            kNumQueries,
                            kTopK,
                            nprobe,
                            actual.data(),
                            actual_distances.data(),
                            hacc,
                            threads
                        );
                        index.search_batch(
                            queries.data(),
                            kNumQueries,
                            kTopK,
                            nprobe,
                            ids_only.data(),
                            nullptr,
                            hacc,
                            threads
                        );
                        EXPECT_EQ(actual, expected);
                        EXPECT_EQ(ids_only, expected);
                        EXPECT_EQ(actual_distances, expected_distances);
                    }
                }
            }
            EXPECT_NO_THROW(index.search_batch(nullptr, 0, kTopK, 1, nullptr));
            EXPECT_THROW(
                index.search_batch(
                    queries.data(),
                    std::numeric_limits<size_t>::max(),
                    kTopK,
                    1,
                    actual.data()
                ),
                std::length_error
            );
            EXPECT_THROW(
                index.search_batch(nullptr, 1, kTopK, 1, actual.data()),
                std::invalid_argument
            );
            EXPECT_THROW(
                index.search_batch(queries.data(), 1, 0, 1, actual.data()),
                std::invalid_argument
            );
            EXPECT_THROW(
                index.search_batch(queries.data(), 1, kTopK, 0, actual.data()),
                std::invalid_argument
            );
            EXPECT_THROW(
                index.search_batch(queries.data(), 1, kTopK, 1, nullptr),
                std::invalid_argument
            );
            std::vector<PID> removed(kCount);
            std::iota(removed.begin(), removed.end(), PID{0});
            index.remove(removed.data(), kCount);
            index.search_batch(
                queries.data(),
                kNumQueries,
                kTopK,
                4,
                actual.data(),
                actual_distances.data(),
                std::nullopt,
                4
            );
            for (size_t i = 0; i < actual.size(); ++i) {
                EXPECT_EQ(actual[i], kPidMax);
                EXPECT_EQ(actual_distances[i], std::numeric_limits<float>::infinity());
            }
        }
    }
    IVF empty;
    PID id = 0;
    EXPECT_THROW(empty.search_batch(queries.data(), 1, 1, 1, &id), std::logic_error);
}

TEST(IvfSearchTest, BatchCandidatesPreserveTiesAndTailCount) {
    buffer::SearchBuffer<float> knns(3);
    const std::array<PID, 6> ids{0, 1, 2, 3, 4, 5};
    const std::array<float, 6> distances{3, 1, 2, 1, 4, -100};
    detail::insert_candidates(knns, ids.data(), distances.data(), 3);
    detail::insert_candidates(knns, ids.data() + 3, distances.data() + 3, 2);
    detail::insert_candidates(knns, ids.data(), distances.data(), 0);
    std::array<PID, 3> results{};
    std::array<float, 3> result_distances{};
    knns.copy_results(results.data(), result_distances.data());
    EXPECT_EQ(results, (std::array<PID, 3>{3, 1, 2}));
    EXPECT_EQ(result_distances, (std::array<float, 3>{1, 1, 2}));
}

TEST(IvfConfigurationTest, RejectsUnsupportedMetric) {
    EXPECT_THROW(
        (IVF(8, 64, 1, 1, static_cast<MetricType>(255), RotatorType::MatrixRotator)),
        std::invalid_argument
    );
}

TEST(IvfConfigurationTest, RejectsUnsupportedCountsAndNullConstructionInputs) {
    constexpr size_t kDim = 64;
    EXPECT_THROW((IVF(0, kDim, 1, 1)), std::invalid_argument);
    EXPECT_THROW(
        (IVF(buffer::kSearchBufferMaxPointCount + 1, kDim, 1, 1)), std::invalid_argument
    );
    EXPECT_THROW((IVF(1, kDim, 0, 1)), std::invalid_argument);
    EXPECT_THROW(
        (IVF(1, kDim, buffer::kSearchBufferMaxPointCount + 1, 1)), std::invalid_argument
    );

    IVF index(1, kDim, 1, 1);
    std::array<float, kDim> vector{};
    const PID cluster = 0;
    EXPECT_THROW(
        index.construct(nullptr, vector.data(), &cluster, false, 1), std::invalid_argument
    );
    EXPECT_THROW(
        index.construct(vector.data(), nullptr, &cluster, false, 1), std::invalid_argument
    );
    EXPECT_THROW(
        index.construct(vector.data(), vector.data(), nullptr, false, 1),
        std::invalid_argument
    );

    IVF unconfigured;
    EXPECT_THROW(
        unconfigured.construct(vector.data(), vector.data(), &cluster, false, 1),
        std::logic_error
    );
    const std::string path = ::testing::TempDir() + "rabitq_unconfigured_ivf.index";
    EXPECT_THROW(unconfigured.save(path.c_str()), std::logic_error);
}

TEST(IvfSearchTest, RawRerankingUsesOriginalCoordinates) {
    constexpr size_t kCount = 33;
    constexpr size_t kDim = 65;
    std::vector<float> data(kCount * kDim);
    std::vector<float> centroid(kDim, 0);
    std::vector<float> query(kDim, 0.25F);
    std::vector<PID> clusters(kCount, 0);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 71) / 71.0F;
    }
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (auto rotator : {RotatorType::FhtKacRotator, RotatorType::MatrixRotator}) {
            IVF index(kCount, kDim, 1, 32, metric, rotator);
            EXPECT_EQ(index.nbits(), 32);
            index.construct(data.data(), centroid.data(), clusters.data(), false, 1);
            for (bool hacc : {false, true}) {
                std::vector<PID> ids(kCount);
                std::vector<float> distances(kCount);
                index.search(query.data(), kCount, 1, ids.data(), distances.data(), hacc);
                for (size_t i = 0; i < kCount; ++i) {
                    ASSERT_LT(ids[i], kCount);
                    float expected = metric == METRIC_L2 ? 0.0F : 1.0F;
                    for (size_t j = 0; j < kDim; ++j) {
                        float value = data[ids[i] * kDim + j];
                        float delta = value - query[j];
                        expected += metric == METRIC_L2 ? delta * delta : -value * query[j];
                    }
                    EXPECT_NEAR(distances[i], expected, 2e-5F);
                }
                EXPECT_TRUE(std::is_sorted(distances.begin(), distances.end()));
                std::vector<PID> ids_only(kCount);
                index.search(query.data(), kCount, 1, ids_only.data(), hacc);
                EXPECT_EQ(ids, ids_only);
            }
        }
    }
}

TEST(IvfSearchTest, AutomaticHighAccuracyMatchesBitPolicy) {
    constexpr size_t kCount = 33;
    constexpr size_t kDim = 65;
    std::vector<float> data(kCount * kDim);
    std::vector<float> centroid(kDim, 0);
    std::vector<float> query(kDim);
    std::vector<PID> clusters(kCount, 0);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 71) / 71.0F;
    }
    for (size_t i = 0; i < kDim; ++i) {
        query[i] = static_cast<float>(i % 13) / 13.0F;
    }
    for (size_t bits : {1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 32UL}) {
        IVF index(kCount, kDim, 1, bits);
        index.construct(data.data(), centroid.data(), clusters.data(), false, 1);
        std::array<PID, 7> ids{}, expected_ids{}, ids_only{};
        std::array<float, 7> distances{}, expected_distances{};
        index.search(query.data(), ids.size(), 1, ids.data(), distances.data());
        index.search(query.data(), ids.size(), 1, ids_only.data());
        index.search(
            query.data(),
            ids.size(),
            1,
            expected_ids.data(),
            expected_distances.data(),
            bits >= 4 && bits <= 9
        );
        EXPECT_EQ(ids, expected_ids);
        EXPECT_EQ(ids_only, expected_ids);
        EXPECT_EQ(distances, expected_distances);
    }
}

TEST(IvfSearchTest, RejectsUnbuiltStateAndInvalidArguments) {
    constexpr size_t kDim = 64;
    std::array<float, kDim> query{};
    PID result = 0;

    IVF empty;
    EXPECT_THROW(empty.search(query.data(), 1, 1, &result), std::logic_error);

    IVF index(1, kDim, 1, 1);
    EXPECT_THROW(index.search(query.data(), 1, 1, &result), std::logic_error);

    std::array<float, kDim> data;
    data.fill(1.0F);
    std::array<float, kDim> centroid{};
    const PID cluster = 0;
    index.construct(data.data(), centroid.data(), &cluster, false, 1);

    EXPECT_THROW(index.search(query.data(), 0, 1, &result), std::invalid_argument);
    EXPECT_THROW(index.search(query.data(), 2, 1, &result), std::invalid_argument);
    EXPECT_THROW(
        index.search(query.data(), std::numeric_limits<size_t>::max(), 1, &result),
        std::invalid_argument
    );
    EXPECT_THROW(index.search(query.data(), 1, 0, &result), std::invalid_argument);
    EXPECT_THROW(index.search(nullptr, 1, 1, &result), std::invalid_argument);
    EXPECT_THROW(index.search(query.data(), 1, 1, nullptr), std::invalid_argument);
}

TEST(IvfSearchTest, FillsSentinelsWhenProbedClustersCannotFillK) {
    constexpr size_t kCount = 2;
    constexpr size_t kDim = 64;
    std::array<float, kCount * kDim> data{};
    std::array<float, kCount * kDim> centroids{};
    std::array<float, kDim> query{};
    std::array<PID, kCount> clusters{1, 1};
    std::fill(data.begin(), data.end(), 1.0F);
    std::fill(centroids.begin() + static_cast<ptrdiff_t>(kDim), centroids.end(), 1.0F);

    IVF index(kCount, kDim, kCount, 32, MetricType::METRIC_L2, RotatorType::MatrixRotator);
    index.construct(data.data(), centroids.data(), clusters.data(), false, 1);

    PID result = kPidMax;
    float distance = -123.0F;
    index.search(query.data(), 1, 1, &result, &distance);
    EXPECT_EQ(result, kPidMax);
    EXPECT_EQ(distance, std::numeric_limits<float>::infinity());
}

TEST(IvfSearchTest, RoutesInnerProductQueriesByInnerProduct) {
    constexpr size_t kCount = 2;
    constexpr size_t kDim = 64;
    std::vector<float> data(kCount * kDim, 0.0F);
    std::vector<float> centroids(kCount * kDim, 0.0F);
    std::array<PID, kCount> clusters{0, 1};
    std::array<float, kDim> query{};
    data[0] = centroids[0] = 0.9F;
    data[kDim] = centroids[kDim] = 100.0F;
    query[0] = 1.0F;

    IVF index(kCount, kDim, kCount, 32, METRIC_IP);
    index.construct(data.data(), centroids.data(), clusters.data(), false, 1);

    PID result = kPidMax;
    float distance = 0.0F;
    index.search(query.data(), 1, 1, &result, &distance);
    EXPECT_EQ(result, 1U);
    EXPECT_NEAR(distance, -99.0F, 1e-5F);
}

TEST(IvfSearchTest, RoutesInnerProductQueriesWithHNSWCentroids) {
    constexpr size_t kDim = 64;
    constexpr size_t kCount = 128;
    std::array<float, kDim> data{};
    std::vector<float> centroids(kCount * kDim, 0.0F);
    std::array<float, kDim> query{};
    for (size_t i = 0; i < kCount; ++i) {
        centroids[i * kDim] = 0.5F;
    }
    data[0] = centroids[0] = 100.0F;
    query[0] = 1.0F;
    const PID cluster = 0;

    IVF index(
        1, kDim, kCount, 32, METRIC_IP, RotatorType::FhtKacRotator, InitializerType::HNSW
    );
    index.construct(data.data(), centroids.data(), &cluster, false, 4);
    PID result = kPidMax;
    float distance = std::numeric_limits<float>::infinity();
    index.search(query.data(), 1, 1, &result, &distance);
    EXPECT_EQ(result, 0U);
    EXPECT_NEAR(distance, -99.0F, 1e-5F);
}

TEST(IvfSearchTest, InnerProductRawRerankingUsesResidualNormForPruning) {
    constexpr size_t kCount = 65;
    constexpr size_t kDim = 64;
    constexpr size_t kTopK = 1;
    std::vector<float> data(kCount * kDim);
    std::vector<float> centroids(2 * kDim, 0.0F);
    std::vector<PID> clusters(kCount, 1);
    std::array<float, kDim> query{};
    query[0] = 1.0F;
    centroids[0] = 100.0F;
    centroids[kDim] = 50.0F;

    for (size_t id = 0; id < kCount; ++id) {
        clusters[id] = id < 32 ? 0 : 1;
        data[id * kDim] = id < 32 ? 10.0F + static_cast<float>(id) * 0.01F : 0.0F;
        for (size_t d = 1; d < kDim; ++d) {
            data[id * kDim + d] =
                std::sin(static_cast<float>((id + 1) * (d + 3)) * 0.17F) * 20.0F;
        }
    }
    data[32 * kDim] = 1000.0F;

    IVF index(kCount, kDim, 2, 32, METRIC_IP);
    index.construct(data.data(), centroids.data(), clusters.data(), false, 1);
    for (bool use_hacc : {false, true}) {
        std::array<PID, kTopK> ids{};
        std::array<float, kTopK> distances{};
        index.search(query.data(), kTopK, 2, ids.data(), distances.data(), use_hacc);
        EXPECT_EQ(ids[0], 32U);
        EXPECT_FLOAT_EQ(distances[0], -999.0F);
    }
}

TEST(IvfPersistenceTest, ReloadsRawAndQuantizedStorage) {
    constexpr size_t kCount = 33;
    constexpr size_t kDim = 65;
    std::vector<float> data(kCount * kDim, 1.0F);
    std::vector<float> centroid(kDim, 0.0F);
    std::vector<PID> clusters(kCount, 0);
    const rabitq_test::TempFile path_file("rabitq_ivf_storage.index");
    const std::string& path = path_file.path();
    IVF loaded;
    for (auto rotator : {RotatorType::FhtKacRotator, RotatorType::MatrixRotator}) {
        for (size_t bits : {32UL, 1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL}) {
            IVF index(kCount, kDim, 1, bits, METRIC_L2, rotator);
            index.construct(data.data(), centroid.data(), clusters.data(), false, 1);
            index.save(path.c_str());
            loaded.load(path.c_str());
            EXPECT_EQ(loaded.nbits(), bits);
            EXPECT_EQ(loaded.padded_dim(), index.padded_dim());
            EXPECT_EQ(loaded.rotator_type(), rotator);
            std::vector<PID> ids(kCount);
            std::vector<PID> loaded_ids(kCount);
            std::vector<float> distances(kCount);
            std::vector<float> loaded_distances(kCount);
            index.search(centroid.data(), kCount, 1, ids.data(), distances.data(), true);
            loaded.search(
                centroid.data(), kCount, 1, loaded_ids.data(), loaded_distances.data(), true
            );
            EXPECT_EQ(ids, loaded_ids);
            EXPECT_EQ(distances, loaded_distances);
        }
    }
}

TEST(IvfPersistenceTest, FailedLoadPreservesExistingIndex) {
    constexpr size_t kTargetCount = 2;
    constexpr size_t kTargetDim = 64;
    std::vector<float> target_data(kTargetCount * kTargetDim, 1.0F);
    std::vector<float> target_centroid(kTargetDim, 0.0F);
    std::array<PID, kTargetCount> target_clusters{0, 0};
    std::fill(
        target_data.begin() + static_cast<ptrdiff_t>(kTargetDim), target_data.end(), 2.0F
    );
    const rabitq_test::TempFile path_file("rabitq_ivf_truncated.index");
    const std::string& path = path_file.path();

    IVF target(kTargetCount, kTargetDim, 1, 32, METRIC_IP, RotatorType::MatrixRotator);
    target.construct(
        target_data.data(), target_centroid.data(), target_clusters.data(), false, 1
    );
    std::array<PID, kTargetCount> expected_ids{};
    std::array<float, kTargetCount> expected_distances{};
    target.search(
        target_centroid.data(),
        kTargetCount,
        1,
        expected_ids.data(),
        expected_distances.data()
    );

    constexpr size_t kSourceCount = 33;
    constexpr size_t kSourceDim = 65;
    std::vector<float> source_data(kSourceCount * kSourceDim, 0.25F);
    std::vector<float> source_centroid(kSourceDim, 0.5F);
    std::vector<PID> source_clusters(kSourceCount, 0);
    IVF source(kSourceCount, kSourceDim, 1, 4, METRIC_L2, RotatorType::FhtKacRotator);
    source.construct(
        source_data.data(), source_centroid.data(), source_clusters.data(), false, 1
    );
    source.save(path.c_str());

    std::ifstream input(path, std::ios::binary);
    const std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()
    );
    ASSERT_GT(bytes.size(), 1U);
    input.close();
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size() - 1));
    output.close();

    EXPECT_ANY_THROW(target.load(path.c_str()));
    EXPECT_EQ(target.max_elements(), kTargetCount);
    EXPECT_EQ(target.dimension(), kTargetDim);
    EXPECT_EQ(target.num_clusters(), 1U);
    EXPECT_EQ(target.nbits(), 32U);
    EXPECT_EQ(target.metric_type(), METRIC_IP);
    EXPECT_EQ(target.rotator_type(), RotatorType::MatrixRotator);

    std::array<PID, kTargetCount> actual_ids{};
    std::array<float, kTargetCount> actual_distances{};
    target.search(
        target_centroid.data(), kTargetCount, 1, actual_ids.data(), actual_distances.data()
    );
    EXPECT_EQ(actual_ids, expected_ids);
    EXPECT_EQ(actual_distances, expected_distances);
}

TEST(IvfPersistenceTest, RejectsPointIdsUsingTheSearchBufferMarker) {
    constexpr size_t kCount = 33;
    constexpr size_t kDim = 64;
    std::vector<float> data(kCount * kDim, 0.25F);
    std::vector<float> centroid(kDim, 0.0F);
    std::vector<PID> clusters(kCount, 0);
    const rabitq_test::TempFile path_file("rabitq_ivf_bad_id.index");
    const std::string& path = path_file.path();

    IVF source(kCount, kDim, 1, 4);
    source.construct(data.data(), centroid.data(), clusters.data(), false, 1);
    source.save(path.c_str());

    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(file.is_open());
    file.seekp(-static_cast<std::streamoff>(sizeof(PID)), std::ios::end);
    const PID marked_id = buffer::kSearchBufferCheckedMask;
    file.write(reinterpret_cast<const char*>(&marked_id), sizeof(marked_id));
    file.close();

    IVF loaded;
    EXPECT_THROW(loaded.load(path.c_str()), std::runtime_error);
}

// ── add / remove ────────────────────────────────────────────────────────────────────────

constexpr size_t kDynDim = 65;  // Exercise padding and partially occupied FastScan batches.
constexpr size_t kDynNumClusters = 4;

std::vector<float> random_rows(size_t rows, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> normal;
    std::vector<float> values(rows * kDynDim);
    for (auto& value : values) {
        value = normal(gen);
    }
    return values;
}

// Nearest centroid in the original space: smallest L2 distance, or largest inner product.
std::vector<PID> nearest_centroids(
    const std::vector<float>& centroids, const float* rows, size_t count, MetricType metric
) {
    std::vector<PID> assigned(count);
    for (size_t i = 0; i < count; ++i) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t c = 0; c < kDynNumClusters; ++c) {
            double dist = 0;
            for (size_t d = 0; d < kDynDim; ++d) {
                const double x = rows[i * kDynDim + d];
                const double y = centroids[c * kDynDim + d];
                dist += metric == METRIC_L2 ? (x - y) * (x - y) : -x * y;
            }
            if (dist < best) {
                best = dist;
                assigned[i] = static_cast<PID>(c);
            }
        }
    }
    return assigned;
}

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string saved_bytes(const IVF& index, const std::string& path) {
    index.save(path.c_str());
    return read_file(path);
}

// Exact distance as the index defines it: squared L2, or 1 - inner product.
double exact_distance(const float* a, const float* b, MetricType metric) {
    double dist = metric == METRIC_L2 ? 0.0 : 1.0;
    for (size_t d = 0; d < kDynDim; ++d) {
        const double x = a[d];
        const double y = b[d];
        dist += metric == METRIC_L2 ? (x - y) * (x - y) : -x * y;
    }
    return dist;
}

struct Hits {
    std::vector<PID> ids;
    std::vector<float> distances;
};

// Ranks every point: k is the point count and every cluster is probed.
Hits search_everything(
    const IVF& index, const float* query, std::optional<bool> hacc = std::nullopt
) {
    const size_t k = index.max_elements();
    Hits hits{std::vector<PID>(k), std::vector<float>(k)};
    if (hacc.has_value()) {
        index.search(
            query, k, kDynNumClusters, hits.ids.data(), hits.distances.data(), *hacc
        );
    } else {
        index.search(query, k, kDynNumClusters, hits.ids.data(), hits.distances.data());
    }
    return hits;
}

TEST(IvfLifecycleTest, BatchTailSurvivesAddRemoveAndReload) {
    constexpr size_t kOld = 31, kCount = 33;
    const auto points = random_rows(kCount, 753);
    const auto query = random_rows(1, 754);
    const std::vector<float> centroid(kDynDim, 0.0F);
    const std::vector<PID> assigned(kCount, 0);
    const std::array<PID, 2> removed{0, kCount - 1};
    const rabitq_test::TempFile path_file("rabitq_ivf_lifecycle.index");
    const std::string& path = path_file.path();
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (size_t bits : {1U, 4U, 9U, 32U}) {
            for (bool hacc : {false, true}) {
                SCOPED_TRACE(::testing::Message() << metric << "/" << bits << "/" << hacc);
                IVF index(kOld, kDynDim, 1, bits, metric);
                index.construct(points.data(), centroid.data(), assigned.data(), false, 1);
                index.add(
                    points.data() + kOld * kDynDim,
                    kCount - kOld,
                    assigned.data() + kOld,
                    false,
                    1
                );
                ASSERT_EQ(index.num_points(), kCount);
                const auto search = [&](const IVF& source) {
                    Hits hits{
                        std::vector<PID>(source.num_points()),
                        std::vector<float>(source.num_points())};
                    source.search(
                        query.data(),
                        hits.ids.size(),
                        1,
                        hits.ids.data(),
                        hits.distances.data(),
                        hacc
                    );
                    return hits;
                };
                const auto before = search(index);
                auto ids = before.ids;
                std::sort(ids.begin(), ids.end());
                for (size_t row = 0; row < kCount; ++row) {
                    ASSERT_EQ(ids[row], row);
                    EXPECT_TRUE(std::isfinite(before.distances[row]));
                    if (bits == 32) {
                        EXPECT_NEAR(
                            before.distances[row],
                            exact_distance(
                                query.data(),
                                points.data() + before.ids[row] * kDynDim,
                                metric
                            ),
                            1e-3
                        );
                    }
                }
                ASSERT_EQ(index.remove(removed.data(), removed.size()), removed.size());
                ASSERT_EQ(index.num_points(), kCount - removed.size());
                const auto after = search(index);
                size_t next = 0;
                for (size_t row = 0; row < kCount; ++row) {
                    if (before.ids[row] == removed[0] || before.ids[row] == removed[1]) {
                        continue;
                    }
                    ASSERT_LT(next, after.ids.size());
                    EXPECT_EQ(after.ids[next], before.ids[row]);
                    EXPECT_EQ(after.distances[next], before.distances[row]);
                    ++next;
                }
                EXPECT_EQ(next, after.ids.size());
                index.save(path.c_str());
                IVF loaded;
                loaded.load(path.c_str());
                EXPECT_EQ(loaded.metric_type(), metric);
                EXPECT_EQ(loaded.nbits(), bits);
                const auto restored = search(loaded);
                EXPECT_EQ(restored.ids, after.ids);
                EXPECT_EQ(restored.distances, after.distances);
            }
        }
    }
}

TEST(IvfAddTest, EveryWayOfAddingTheSamePointsGivesByteIdenticalIndexes) {
    constexpr size_t kOld = 200;
    constexpr size_t kNew = 200;
    const auto points = random_rows(kOld + kNew, 11);
    const auto centroids = random_rows(kDynNumClusters, 12);
    const float* extra = points.data() + (kOld * kDynDim);
    const rabitq_test::TempFile base_path_file("rabitq_ivf_add_base.index");
    const std::string& base_path = base_path_file.path();
    const rabitq_test::TempFile path_file("rabitq_ivf_add.index");
    const std::string& path = path_file.path();

    for (auto metric : {METRIC_L2, METRIC_IP}) {
        const auto assigned =
            nearest_centroids(centroids, points.data(), kOld + kNew, metric);
        for (size_t bits : {1UL, 4UL, 9UL, 32UL}) {
            SCOPED_TRACE(
                ::testing::Message()
                << "metric=" << (metric == METRIC_L2 ? "l2" : "ip") << " bits=" << bits
            );
            IVF base(kOld, kDynDim, kDynNumClusters, bits, metric);
            base.construct(points.data(), centroids.data(), assigned.data(), false, 1);
            base.save(base_path.c_str());

            // Every index below starts from the same file, so they share one rotator.
            IVF single;
            single.load(base_path.c_str());
            single.add(extra, kNew, assigned.data() + kOld, false, 1);
            ASSERT_EQ(single.max_elements(), kOld + kNew);
            const std::string expected = saved_bytes(single, path);

            IVF routed;
            routed.load(base_path.c_str());
            routed.add(extra, kNew, nullptr, false, 1);
            EXPECT_TRUE(saved_bytes(routed, path) == expected) << "routed";

            // Chunk sizes put the tail batch of each cluster on and off a batch boundary.
            IVF chunked;
            chunked.load(base_path.c_str());
            size_t done = 0;
            for (size_t chunk : std::array<size_t, 6>{1, 31, 32, 7, 1, kNew - 72}) {
                chunked.add(
                    extra + (done * kDynDim), chunk, assigned.data() + kOld + done, false, 1
                );
                done += chunk;
            }
            ASSERT_EQ(done, kNew);
            EXPECT_TRUE(saved_bytes(chunked, path) == expected) << "chunked";

            // Even clusters are reserved and grow in place; odd ones are rebuilt each time.
            IVF reserved;
            reserved.load(base_path.c_str());
            std::vector<size_t> final_sizes(kDynNumClusters, 0);
            for (PID cluster : assigned) {
                ++final_sizes[cluster];
            }
            for (size_t i = 0; i < kDynNumClusters; i += 2) {
                reserved.reserve(static_cast<PID>(i), final_sizes[i]);
            }
            const Cluster reserved_view = IvfTestAccess::cluster_view(reserved, 0);
            done = 0;
            for (size_t chunk : std::array<size_t, 6>{1, 31, 32, 7, 1, kNew - 72}) {
                reserved.add(
                    extra + (done * kDynDim), chunk, assigned.data() + kOld + done, false, 1
                );
                done += chunk;
            }
            EXPECT_TRUE(saved_bytes(reserved, path) == expected) << "reserved";
            EXPECT_EQ(
                IvfTestAccess::cluster_view(reserved, 0).batch_data(),
                reserved_view.batch_data()
            );

            IVF threaded;
            threaded.load(base_path.c_str());
            threaded.add(extra, kNew, nullptr, false, 4);
            EXPECT_TRUE(saved_bytes(threaded, path) == expected) << "threaded";
        }
    }
}

TEST(IvfAddTest, AddingToOneClusterLeavesOtherClustersUntouched) {
    // Cluster sizes 0, 1, 32, 97 cover empty, single, full-batch and tail clusters.
    constexpr size_t kOld = 130;
    const auto points = random_rows(kOld, 741);
    const auto centroids = random_rows(kDynNumClusters, 742);
    std::vector<PID> assigned(kOld, 3);
    assigned[0] = 1;
    std::fill(assigned.begin() + 1, assigned.begin() + 33, PID{2});

    IVF index(kOld, kDynDim, kDynNumClusters, 4, METRIC_L2);
    index.construct(points.data(), centroids.data(), assigned.data(), false, 1);

    std::vector<Cluster> before;
    before.reserve(kDynNumClusters);
    for (size_t i = 0; i < kDynNumClusters; ++i) {
        before.push_back(IvfTestAccess::cluster_view(index, i));
    }

    // Route every new point to cluster 0 (the initially-empty one) explicitly.
    constexpr size_t kNew = 5;
    const auto extra = random_rows(kNew, 743);
    const std::vector<PID> new_cluster_ids(kNew, 0);
    index.add(extra.data(), kNew, new_cluster_ids.data(), false, 1);

    for (size_t i = 0; i < kDynNumClusters; ++i) {
        SCOPED_TRACE(::testing::Message() << "cluster=" << i);
        const Cluster after = IvfTestAccess::cluster_view(index, i);
        if (i == 0) {
            EXPECT_EQ(after.num(), before[i].num() + kNew);
        } else {
            // Untouched clusters keep their storage: no reallocation and no copy.
            EXPECT_EQ(after.num(), before[i].num());
            EXPECT_EQ(after.batch_data(), before[i].batch_data());
            EXPECT_EQ(after.ex_data(), before[i].ex_data());
            EXPECT_EQ(after.ids(), before[i].ids());
        }
    }
}

TEST(IvfAddTest, SparseInterleavedClustersPreserveRowOrderAndMatchRebuild) {
    constexpr size_t kOld = 3, kCount = 9, kNumClusters = 257;
    const auto points = random_rows(kCount, 744);
    const auto centroids = random_rows(kNumClusters, 745);
    const std::array<PID, kCount> assigned{0, 128, 256, 256, 0, 256, 128, 0, 256};
    const rabitq_test::TempFile path_file("rabitq_ivf_sparse_add.index");
    const std::string& path = path_file.path();
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (size_t bits : {1U, 4U, 32U}) {
            IVF index(kOld, kDynDim, kNumClusters, bits, metric);
            index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
            index.reserve(256, 32);
            const Cluster reserved = IvfTestAccess::cluster_view(index, 256);
            index.add(
                points.data() + kOld * kDynDim,
                kCount - kOld,
                assigned.data() + kOld,
                false,
                4
            );
            EXPECT_EQ(IvfTestAccess::cluster_view(index, 256).ids(), reserved.ids());
            for (size_t i = 0; i < kNumClusters; ++i) {
                std::vector<PID> expected;
                for (size_t row = 0; row < kCount; ++row) {
                    if (assigned[row] == i) {
                        expected.push_back(static_cast<PID>(row));
                    }
                }
                const Cluster cluster = IvfTestAccess::cluster_view(index, i);
                ASSERT_EQ(cluster.num(), expected.size());
                for (size_t row = 0; row < expected.size(); ++row) {
                    EXPECT_EQ(cluster.ids()[row], expected[row]);
                }
            }
            const auto expected = saved_bytes(index, path);
            IVF rebuilt;
            rebuilt.load(path.c_str());
            rebuilt.construct(points.data(), centroids.data(), assigned.data(), false, 1);
            EXPECT_EQ(saved_bytes(rebuilt, path), expected);
        }
    }
}

TEST(IvfAddTest, UnreservedClustersGrowToExactlyTheirNewSize) {
    constexpr size_t kOld = 40;
    const auto points = random_rows(kOld + 1, 851);
    const auto centroids = random_rows(kDynNumClusters, 852);
    const std::vector<PID> assigned(kOld, 0);

    for (size_t bits : {1UL, 4UL, 32UL}) {
        SCOPED_TRACE(::testing::Message() << "bits=" << bits);
        IVF index(kOld, kDynDim, kDynNumClusters, bits, METRIC_L2);
        index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
        const PID route_to_zero = 0;
        index.add(points.data() + (kOld * kDynDim), 1, &route_to_zero, false, 1);
        EXPECT_EQ(IvfTestAccess::cluster_view(index, 0).num(), kOld + 1);
        EXPECT_FALSE(IvfTestAccess::has_spare_capacity(index, 0));
    }
}

TEST(IvfReserveTest, ReservedClusterGrowsInPlaceUntilItsCapacityIsUsed) {
    constexpr size_t kOld = 40;
    constexpr size_t kReserved = kOld + 2;
    const auto points = random_rows(kReserved + 1, 861);
    const auto centroids = random_rows(kDynNumClusters, 862);
    const std::vector<PID> assigned(kOld, 0);
    const PID route_to_zero = 0;

    for (size_t bits : {1UL, 4UL, 32UL}) {
        SCOPED_TRACE(::testing::Message() << "bits=" << bits);
        IVF index(kOld, kDynDim, kDynNumClusters, bits, METRIC_L2);
        index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
        index.reserve(0, kReserved);
        const Cluster reserved = IvfTestAccess::cluster_view(index, 0);
        EXPECT_EQ(reserved.num(), kOld);

        for (size_t row = kOld; row < kReserved; ++row) {
            index.add(points.data() + (row * kDynDim), 1, &route_to_zero, false, 1);
            const Cluster grown = IvfTestAccess::cluster_view(index, 0);
            EXPECT_EQ(grown.num(), row + 1);
            EXPECT_EQ(grown.batch_data(), reserved.batch_data());
            EXPECT_EQ(grown.ex_data(), reserved.ex_data());
            EXPECT_EQ(grown.ids(), reserved.ids());
        }

        // The old storage is alive while its replacement is allocated, so they differ.
        index.add(points.data() + (kReserved * kDynDim), 1, &route_to_zero, false, 1);
        EXPECT_NE(
            IvfTestAccess::cluster_view(index, 0).batch_data(), reserved.batch_data()
        );
        EXPECT_FALSE(IvfTestAccess::has_spare_capacity(index, 0));
    }
}

TEST(IvfReserveTest, RejectsBadInputAndLeavesSavedBytesUnchanged) {
    constexpr size_t kOld = 40;
    const auto points = random_rows(kOld, 871);
    const auto centroids = random_rows(kDynNumClusters, 872);
    const std::vector<PID> assigned(kOld, 0);
    const rabitq_test::TempFile path_file("rabitq_ivf_reserve.index");
    const std::string& path = path_file.path();
    const auto expect_message = [](const auto& call, const char* message) {
        try {
            call();
            ADD_FAILURE() << "expected: " << message;
        } catch (const std::exception& error) { EXPECT_STREQ(error.what(), message); }
    };

    IVF unbuilt(kOld, kDynDim, kDynNumClusters, 4);
    EXPECT_THROW(unbuilt.reserve(0, 1), std::logic_error);
    expect_message(
        [&] { unbuilt.reserve(0, 1); },
        "IVF index must be constructed or loaded before reserve"
    );

    IVF index(kOld, kDynDim, kDynNumClusters, 4);
    index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
    const std::string expected = saved_bytes(index, path);
    const Cluster constructed = IvfTestAccess::cluster_view(index, 0);

    EXPECT_THROW(
        index.reserve(static_cast<PID>(kDynNumClusters), 1), std::invalid_argument
    );
    expect_message(
        [&] { index.reserve(static_cast<PID>(kDynNumClusters), 1); },
        "IVF cluster ID is out of range"
    );
    EXPECT_THROW(
        index.reserve(0, buffer::kSearchBufferMaxPointCount + 1), std::invalid_argument
    );
    expect_message(
        [&] { index.reserve(0, buffer::kSearchBufferMaxPointCount + 1); },
        "IVF point count exceeds the supported ID range"
    );

    // Asking for less than the cluster already holds does nothing.
    index.reserve(0, kOld / 2);
    EXPECT_EQ(IvfTestAccess::cluster_view(index, 0).batch_data(), constructed.batch_data());
    EXPECT_FALSE(IvfTestAccess::has_spare_capacity(index, 0));

    // Spare capacity changes neither the file nor what a loaded index holds.
    index.reserve(0, kOld * 2);
    index.reserve(1, kOld);
    EXPECT_TRUE(IvfTestAccess::has_spare_capacity(index, 0));
    EXPECT_TRUE(saved_bytes(index, path) == expected);
    IVF loaded;
    loaded.load(path.c_str());
    for (size_t i = 0; i < kDynNumClusters; ++i) {
        EXPECT_FALSE(IvfTestAccess::has_spare_capacity(loaded, i));
    }
}

// Delegates to a real rotator, but the call numbered `fail_at` throws.
class FailingRotator : public Rotator<float> {
   public:
    FailingRotator(std::unique_ptr<Rotator<float>> inner, size_t fail_at)
        : Rotator<float>(0, inner->size()), inner_(std::move(inner)), fail_at_(fail_at) {}

    void rotate(const float* data, float* rotated_data) const override {
        if (calls_++ == fail_at_) {
            throw std::runtime_error("injected rotation failure");
        }
        inner_->rotate(data, rotated_data);
    }
    void load(std::ifstream& input) override { inner_->load(input); }
    void save(std::ofstream& output) const override { inner_->save(output); }
    void load(const char* data) override { inner_->load(data); }
    void save(char* data) const override { inner_->save(data); }
    [[nodiscard]] size_t dump_bytes() const override { return inner_->dump_bytes(); }

    std::unique_ptr<Rotator<float>> release() { return std::move(inner_); }

   private:
    std::unique_ptr<Rotator<float>> inner_;
    size_t fail_at_;
    mutable size_t calls_ = 0;
};

TEST(IvfAddTest, FailedAddLeavesClustersGrowingInPlaceUnchanged) {
    // Cluster 0 has a partial tail batch and grows in place; cluster 1 fails afterwards.
    constexpr size_t kOld = 70;
    constexpr size_t kInPlace = 3;
    const auto points = random_rows(kOld + 6, 881);
    const auto centroids = random_rows(kDynNumClusters, 882);
    std::vector<PID> assigned(kOld, 1);
    std::fill(assigned.begin(), assigned.begin() + 40, PID{0});
    const std::array<PID, 6> new_clusters{0, 0, 0, 1, 1, 1};
    const float* extra = points.data() + (kOld * kDynDim);
    const rabitq_test::TempFile base_path_file("rabitq_ivf_add_fail_base.index");
    const std::string& base_path = base_path_file.path();
    const rabitq_test::TempFile path_file("rabitq_ivf_add_fail.index");
    const std::string& path = path_file.path();

    for (size_t bits : {1UL, 4UL, 32UL}) {
        SCOPED_TRACE(::testing::Message() << "bits=" << bits);
        IVF index(kOld, kDynDim, kDynNumClusters, bits, METRIC_L2);
        index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
        index.save(base_path.c_str());
        index.reserve(0, 64);
        const std::string expected = saved_bytes(index, path);
        const Cluster before = IvfTestAccess::cluster_view(index, 0);

        auto& rotator = IvfTestAccess::rotator(index);
        rotator = std::make_unique<FailingRotator>(std::move(rotator), kInPlace);
        EXPECT_THROW(
            index.add(extra, new_clusters.size(), new_clusters.data(), false, 1),
            std::runtime_error
        );
        rotator = static_cast<FailingRotator&>(*rotator).release();

        EXPECT_EQ(index.max_elements(), kOld);
        EXPECT_EQ(IvfTestAccess::cluster_view(index, 0).num(), before.num());
        EXPECT_EQ(IvfTestAccess::cluster_view(index, 0).batch_data(), before.batch_data());
        EXPECT_TRUE(saved_bytes(index, path) == expected);

        // Retrying matches an index that never saw the failure.
        index.add(extra, new_clusters.size(), new_clusters.data(), false, 1);
        IVF reference;
        reference.load(base_path.c_str());
        reference.add(extra, new_clusters.size(), new_clusters.data(), false, 1);
        EXPECT_TRUE(saved_bytes(index, path) == saved_bytes(reference, path));
    }
}

TEST(IvfConstructionTest, RebuildMatchesIncrementalEncodingWithSkewedClustersAndTails) {
    // Cluster sizes 0, 1, 32, 97 cover empty rows and multiple physical tail batches.
    constexpr size_t kCount = 130;
    const auto points = random_rows(kCount, 731);
    const auto centroids = random_rows(kDynNumClusters, 732);
    std::vector<PID> assigned(kCount, 3);
    assigned[0] = 1;
    std::fill(assigned.begin() + 1, assigned.begin() + 33, PID{2});
    const rabitq_test::TempFile path_file("rabitq_ivf_rebuild.index");
    const std::string& path = path_file.path();
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (auto rotation : {RotatorType::FhtKacRotator, RotatorType::MatrixRotator}) {
            for (size_t bits : {1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 32UL}) {
                SCOPED_TRACE(
                    ::testing::Message() << "bits=" << bits << " metric=" << metric
                );
                IVF incremental(1, kDynDim, kDynNumClusters, bits, metric, rotation);
                incremental.construct(
                    points.data(), centroids.data(), assigned.data(), false, 1
                );
                incremental.add(
                    points.data() + kDynDim, kCount - 1, assigned.data() + 1, false, 1
                );
                const std::string expected = saved_bytes(incremental, path);
                IVF rebuilt;
                rebuilt.load(path.c_str());
                for (size_t threads : {1UL, 4UL}) {
                    rebuilt.construct(
                        points.data(), centroids.data(), assigned.data(), false, threads
                    );
                    EXPECT_EQ(saved_bytes(rebuilt, path), expected);
                    const auto actual_hits = search_everything(rebuilt, points.data());
                    const auto expected_hits =
                        search_everything(incremental, points.data());
                    EXPECT_EQ(actual_hits.ids, expected_hits.ids);
                    EXPECT_EQ(actual_hits.distances, expected_hits.distances);
                }
            }
        }
    }
}

TEST(IvfAddTest, KeepsExistingDistancesAndQuantizesNewPointsAccurately) {
    constexpr size_t kOld = 150;
    constexpr size_t kNew = 150;
    constexpr size_t kNumQueries = 8;
    const auto points = random_rows(kOld + kNew, 21);
    const auto centroids = random_rows(kDynNumClusters, 22);
    const auto queries = random_rows(kNumQueries, 23);

    for (auto metric : {METRIC_L2, METRIC_IP}) {
        const auto assigned =
            nearest_centroids(centroids, points.data(), kOld + kNew, metric);
        for (size_t bits : {1UL, 2UL, 4UL, 9UL, 32UL}) {
            SCOPED_TRACE(
                ::testing::Message()
                << "metric=" << (metric == METRIC_L2 ? "l2" : "ip") << " bits=" << bits
            );
            IVF index(kOld, kDynDim, kDynNumClusters, bits, metric);
            index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
            std::vector<Hits> before;
            for (size_t q = 0; q < kNumQueries; ++q) {
                before.push_back(search_everything(index, queries.data() + (q * kDynDim)));
            }

            index.add(points.data() + (kOld * kDynDim), kNew, nullptr, false, 1);

            double old_error = 0;
            double new_error = 0;
            for (size_t q = 0; q < kNumQueries; ++q) {
                const float* query = queries.data() + (q * kDynDim);
                const auto after = search_everything(index, query);
                std::vector<float> by_id(kOld + kNew, std::nanf(""));
                for (size_t r = 0; r < after.ids.size(); ++r) {
                    ASSERT_LT(after.ids[r], kOld + kNew) << "every point is still ranked";
                    by_id[after.ids[r]] = after.distances[r];
                }
                // A tail batch is unpacked and repacked; old points must not change at all.
                for (size_t r = 0; r < before[q].ids.size(); ++r) {
                    EXPECT_EQ(by_id[before[q].ids[r]], before[q].distances[r]);
                }
                for (size_t id = 0; id < kOld + kNew; ++id) {
                    const double error = std::abs(
                        by_id[id] -
                        exact_distance(query, points.data() + (id * kDynDim), metric)
                    );
                    (id < kOld ? old_error : new_error) += error;
                }
            }
            old_error /= kNumQueries * kOld;
            new_error /= kNumQueries * kNew;
            // Mis-encoded points would be off by the spread of the data, not by a factor.
            EXPECT_LT(new_error, 3 * old_error + 1e-4) << old_error << " vs " << new_error;
        }
    }
}

TEST(IvfAddTest, AddedRawVectorsAreRerankedExactly) {
    constexpr size_t kOld = 40;
    constexpr size_t kNew = 60;
    const auto points = random_rows(kOld + kNew, 31);
    const auto centroids = random_rows(kDynNumClusters, 32);
    const auto queries = random_rows(3, 33);
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        IVF index(kOld, kDynDim, kDynNumClusters, 32, metric);
        const auto assigned = nearest_centroids(centroids, points.data(), kOld, metric);
        index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
        index.add(points.data() + (kOld * kDynDim), kNew, nullptr, false, 1);
        for (size_t q = 0; q < 3; ++q) {
            const float* query = queries.data() + (q * kDynDim);
            const auto hits = search_everything(index, query);
            for (size_t r = 0; r < hits.ids.size(); ++r) {
                EXPECT_NEAR(
                    hits.distances[r],
                    exact_distance(query, points.data() + (hits.ids[r] * kDynDim), metric),
                    1e-4
                );
            }
            EXPECT_TRUE(std::is_sorted(hits.distances.begin(), hits.distances.end()));
        }
    }
}

TEST(IvfAddTest, FasterQuantizationIsSupportedAndDeterministic) {
    constexpr size_t kOld = 100;
    constexpr size_t kNew = 50;
    const auto points = random_rows(kOld + kNew, 41);
    const auto centroids = random_rows(kDynNumClusters, 42);
    const auto assigned =
        nearest_centroids(centroids, points.data(), kOld + kNew, METRIC_L2);
    const rabitq_test::TempFile base_path_file("rabitq_ivf_add_faster_base.index");
    const std::string& base_path = base_path_file.path();
    const rabitq_test::TempFile path_file("rabitq_ivf_add_faster.index");
    const std::string& path = path_file.path();

    IVF base(kOld, kDynDim, kDynNumClusters, 5);
    base.construct(points.data(), centroids.data(), assigned.data(), true, 1);
    base.save(base_path.c_str());
    IVF first;
    IVF second;
    first.load(base_path.c_str());
    second.load(base_path.c_str());
    first.add(points.data() + (kOld * kDynDim), kNew, assigned.data() + kOld, true, 1);
    second.add(points.data() + (kOld * kDynDim), kNew, assigned.data() + kOld, true, 2);
    EXPECT_TRUE(saved_bytes(first, path) == saved_bytes(second, path));
    const auto hits = search_everything(first, points.data() + (kOld * kDynDim));
    EXPECT_EQ(hits.ids.size(), kOld + kNew);
    EXPECT_TRUE(std::none_of(hits.distances.begin(), hits.distances.end(), [](float x) {
        return std::isnan(x);
    }));
}

TEST(IvfAddTest, RoutesNewPointsWithHNSWCentroids) {
    constexpr size_t kDim = 64;
    constexpr size_t kCount = 128;
    // Centroid i has first coordinate i + 1, so an inner product with e0 is largest for the
    // last centroid and the routing decision is unambiguous.
    std::vector<float> centroids(kCount * kDim, 0.0F);
    for (size_t i = 0; i < kCount; ++i) {
        centroids[i * kDim] = static_cast<float>(i + 1);
    }
    std::array<float, kDim> first{};
    first[0] = 1.0F;
    const PID cluster = 0;
    IVF index(
        1, kDim, kCount, 32, METRIC_IP, RotatorType::FhtKacRotator, InitializerType::HNSW
    );
    index.construct(first.data(), centroids.data(), &cluster, false, 4);

    std::array<float, kDim> second{};
    second[0] = 2.0F;
    index.add(second.data(), 1, nullptr, false, 2);
    ASSERT_EQ(index.max_elements(), 2U);

    // The query is routed by the same rule, so it reaches the cluster that received the
    // point.
    std::array<PID, 1> result{kPidMax};
    std::array<float, 1> distance{};
    index.search(second.data(), 1, 1, result.data(), distance.data());
    EXPECT_EQ(result[0], 1U);
    EXPECT_NEAR(distance[0], 1.0F - 4.0F, 1e-5F);
}

TEST(IvfAddTest, RejectsBadInputAndLeavesTheIndexUnchanged) {
    constexpr size_t kCount = 70;
    const auto points = random_rows(kCount + 1, 51);
    const auto centroids = random_rows(kDynNumClusters, 52);
    const auto assigned = nearest_centroids(centroids, points.data(), kCount, METRIC_L2);
    const rabitq_test::TempFile path_file("rabitq_ivf_add_reject.index");
    const std::string& path = path_file.path();

    IVF unbuilt(kCount, kDynDim, kDynNumClusters, 4);
    EXPECT_THROW(unbuilt.add(points.data(), 1), std::logic_error);
    IVF unconfigured;
    EXPECT_THROW(unconfigured.add(points.data(), 1), std::logic_error);

    IVF index(kCount, kDynDim, kDynNumClusters, 4);
    index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
    const std::string expected = saved_bytes(index, path);

    const std::array<PID, 2> bad_clusters{0, static_cast<PID>(kDynNumClusters)};
    EXPECT_THROW(index.add(points.data(), 2, bad_clusters.data()), std::invalid_argument);
    EXPECT_THROW(index.add(nullptr, 1), std::invalid_argument);
    // The ID check happens before the data is read.
    EXPECT_THROW(
        index.add(points.data(), buffer::kSearchBufferMaxPointCount), std::invalid_argument
    );
    EXPECT_NO_THROW(index.add(nullptr, 0));
    EXPECT_NO_THROW(index.add(points.data(), 0));
    EXPECT_EQ(index.max_elements(), kCount);
    EXPECT_TRUE(saved_bytes(index, path) == expected);
}

TEST(IvfRemoveTest, ExcludesRemovedPointsOnEveryScanPath) {
    constexpr size_t kCount = 200;
    constexpr size_t kNumQueries = 5;
    const auto points = random_rows(kCount, 61);
    const auto centroids = random_rows(kDynNumClusters, 62);
    const auto queries = random_rows(kNumQueries, 63);

    std::vector<PID> removed_ids;
    std::vector<bool> is_removed(kCount, false);
    for (PID id = 0; id < kCount; id += 3) {
        removed_ids.push_back(id);
        is_removed[id] = true;
    }
    removed_ids.push_back(kCount - 1);  // the last point sits in a partial batch
    is_removed[kCount - 1] = true;
    removed_ids.push_back(0);  // repeated IDs count once
    const size_t unique_removed =
        static_cast<size_t>(std::count(is_removed.begin(), is_removed.end(), true));

    for (auto metric : {METRIC_L2, METRIC_IP}) {
        const auto assigned = nearest_centroids(centroids, points.data(), kCount, metric);
        // 1-3 bits use standard FastScan, 4-9 bits and raw storage exercise other paths too
        for (size_t bits : {1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 32UL}) {
            for (bool hacc : {false, true}) {
                SCOPED_TRACE(
                    ::testing::Message() << "metric=" << (metric == METRIC_L2 ? "l2" : "ip")
                                         << " bits=" << bits << " hacc=" << hacc
                );
                IVF index(kCount, kDynDim, kDynNumClusters, bits, metric);
                index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
                std::vector<Hits> before;
                for (size_t q = 0; q < kNumQueries; ++q) {
                    before.push_back(
                        search_everything(index, queries.data() + (q * kDynDim), hacc)
                    );
                }

                EXPECT_EQ(
                    index.remove(removed_ids.data(), removed_ids.size()), unique_removed
                );
                EXPECT_EQ(index.remove(removed_ids.data(), removed_ids.size()), 0U);
                EXPECT_EQ(index.max_elements(), kCount);

                for (size_t q = 0; q < kNumQueries; ++q) {
                    const auto after =
                        search_everything(index, queries.data() + (q * kDynDim), hacc);
                    // Survivors keep their order and distances; the rest are sentinels.
                    size_t next = 0;
                    for (size_t r = 0; r < kCount; ++r) {
                        if (is_removed[before[q].ids[r]]) {
                            continue;
                        }
                        ASSERT_EQ(after.ids[next], before[q].ids[r]);
                        ASSERT_EQ(after.distances[next], before[q].distances[r]);
                        ++next;
                    }
                    ASSERT_EQ(next, kCount - unique_removed);
                    for (size_t r = next; r < kCount; ++r) {
                        EXPECT_EQ(after.ids[r], kPidMax);
                        EXPECT_EQ(
                            after.distances[r], std::numeric_limits<float>::infinity()
                        );
                    }

                    // A small k must still be filled from the survivors only.
                    std::array<PID, 10> top{};
                    std::array<float, 10> top_distances{};
                    index.search(
                        queries.data() + (q * kDynDim),
                        top.size(),
                        kDynNumClusters,
                        top.data(),
                        top_distances.data(),
                        hacc
                    );
                    for (size_t r = 0; r < top.size(); ++r) {
                        ASSERT_LT(top[r], kCount);
                        EXPECT_FALSE(is_removed[top[r]]);
                        EXPECT_TRUE(std::isfinite(top_distances[r]));
                    }
                }
            }
        }
    }
}

TEST(IvfRemoveTest, SurvivesSaveLoadAndLaterAdds) {
    constexpr size_t kOld = 100;
    constexpr size_t kNew = 60;
    constexpr size_t kNumQueries = 4;
    const auto points = random_rows(kOld + kNew, 71);
    const auto centroids = random_rows(kDynNumClusters, 72);
    const auto queries = random_rows(kNumQueries, 73);
    const std::vector<PID> removed{1, 33, 34, 98, 99};
    const rabitq_test::TempFile path_file("rabitq_ivf_remove.index");
    const std::string& path = path_file.path();

    // Counts the live results, checking that no removed point appears and that the unfilled
    // slots hold the sentinel with an infinite distance.
    auto count_live = [&removed](const Hits& hits) {
        size_t live = 0;
        for (size_t r = 0; r < hits.ids.size(); ++r) {
            if (hits.ids[r] == kPidMax) {
                EXPECT_EQ(hits.distances[r], std::numeric_limits<float>::infinity());
                continue;
            }
            ++live;
            EXPECT_EQ(std::count(removed.begin(), removed.end(), hits.ids[r]), 0);
            EXPECT_TRUE(std::isfinite(hits.distances[r]));
        }
        return live;
    };

    for (auto metric : {METRIC_L2, METRIC_IP}) {
        const auto assigned = nearest_centroids(centroids, points.data(), kOld, metric);
        for (size_t bits : {1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 32UL}) {
            for (bool hacc : {false, true}) {
                SCOPED_TRACE(
                    ::testing::Message() << "metric=" << (metric == METRIC_L2 ? "l2" : "ip")
                                         << " bits=" << bits << " hacc=" << hacc
                );
                IVF index(kOld, kDynDim, kDynNumClusters, bits, metric);
                index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
                ASSERT_EQ(index.remove(removed.data(), removed.size()), removed.size());
                index.save(path.c_str());

                IVF loaded;
                loaded.load(path.c_str());
                ASSERT_EQ(loaded.max_elements(), kOld);
                for (size_t q = 0; q < kNumQueries; ++q) {
                    const float* query = queries.data() + (q * kDynDim);
                    const auto expected = search_everything(index, query, hacc);
                    const auto actual = search_everything(loaded, query, hacc);
                    EXPECT_EQ(actual.ids, expected.ids);
                    EXPECT_EQ(actual.distances, expected.distances);
                    EXPECT_EQ(count_live(actual), kOld - removed.size());
                }
                EXPECT_EQ(loaded.remove(removed.data(), removed.size()), 0U);

                // Adding repacks the tail batches that hold removed points; they must stay
                // removed, and a reloaded index must grow exactly like the original.
                const float* added = points.data() + (kOld * kDynDim);
                index.add(added, kNew, nullptr, false, 1);
                loaded.add(added, kNew, nullptr, false, 1);
                ASSERT_EQ(loaded.max_elements(), kOld + kNew);
                for (size_t q = 0; q < kNumQueries; ++q) {
                    const float* query = queries.data() + (q * kDynDim);
                    const auto expected = search_everything(index, query, hacc);
                    const auto actual = search_everything(loaded, query, hacc);
                    EXPECT_EQ(actual.ids, expected.ids);
                    EXPECT_EQ(actual.distances, expected.distances);
                    EXPECT_EQ(count_live(actual), kOld + kNew - removed.size());
                }
            }
        }
    }
}

TEST(IvfRemoveTest, RejectsBadInputAndRemovesNothingOnFailure) {
    constexpr size_t kCount = 70;
    const auto points = random_rows(kCount, 81);
    const auto centroids = random_rows(kDynNumClusters, 82);
    const auto assigned = nearest_centroids(centroids, points.data(), kCount, METRIC_L2);
    const rabitq_test::TempFile path_file("rabitq_ivf_remove_reject.index");
    const std::string& path = path_file.path();

    const PID id = 0;
    IVF unbuilt(kCount, kDynDim, kDynNumClusters, 4);
    EXPECT_THROW(unbuilt.remove(&id, 1), std::logic_error);

    IVF index(kCount, kDynDim, kDynNumClusters, 4);
    index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
    const std::string expected = saved_bytes(index, path);

    const std::array<PID, 2> out_of_range{3, static_cast<PID>(kCount)};
    EXPECT_THROW(index.remove(out_of_range.data(), 2), std::invalid_argument);
    EXPECT_THROW(index.remove(nullptr, 1), std::invalid_argument);
    EXPECT_EQ(index.remove(nullptr, 0), 0U);
    EXPECT_TRUE(saved_bytes(index, path) == expected);

    // Removing everything leaves a searchable index that returns only sentinels.
    std::vector<PID> everything(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        everything[i] = static_cast<PID>(i);
    }
    EXPECT_EQ(index.remove(everything.data(), kCount), kCount);
    const auto hits = search_everything(index, points.data());
    for (size_t r = 0; r < kCount; ++r) {
        EXPECT_EQ(hits.ids[r], kPidMax);
    }
}

}  // namespace
}  // namespace rabitqlib::ivf

TEST(IvfSidecarValidationTest, RejectsMismatchedVectorStorage) {
    using namespace rabitqlib;
    for (const auto metric : {METRIC_L2, METRIC_IP}) {
        for (const size_t sidecar_dim : {64U, 128U}) {
            const rabitq_test::TempFile path_file("rabitq_sidecar_layout");
            const std::string& path = path_file.path();
            std::vector<float> data(96, 1.0F), center(128, 0.0F);
            PID cluster = 0;
            ivf::IVF original(
                1, 96, 1, 4, metric, RotatorType::FhtKacRotator, ivf::InitializerType::HNSW
            );
            original.construct(data.data(), center.data(), &cluster, false, 1);
            original.save(path.c_str());
            ivf::HNSWInitializer other(sidecar_dim, 1, metric);
            other.add_vectors(center.data(), 1);
            std::ofstream ignored;
            other.save(ignored, path.c_str());
            ivf::IVF loaded;
            try {
                loaded.load(path.c_str());
                FAIL() << "mismatched sidecar must be rejected";
            } catch (const std::runtime_error& error) {
                EXPECT_STREQ(
                    error.what(), "HNSW centroid layout does not match IVF dimension"
                );
            }
            original.save(path.c_str());
            ASSERT_NO_THROW(loaded.load(path.c_str()));
            PID id = 0;
            float distance = 0;
            ASSERT_NO_THROW(loaded.search(data.data(), 1, 1, &id, &distance));
        }
    }
}
