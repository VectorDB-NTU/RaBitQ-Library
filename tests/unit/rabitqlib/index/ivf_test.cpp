#include "rabitqlib/index/ivf/ivf.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/rotator.hpp"

namespace rabitqlib::ivf {
namespace {

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
    constexpr size_t kNum = 33;
    constexpr size_t kDim = 65;
    std::vector<float> data(kNum * kDim);
    std::vector<float> centroid(kDim, 0);
    std::vector<float> query(kDim, 0.25F);
    std::vector<PID> clusters(kNum, 0);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 71) / 71.0F;
    }
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (auto rotator : {RotatorType::FhtKacRotator, RotatorType::MatrixRotator}) {
            IVF index(kNum, kDim, 1, 32, metric, rotator);
            EXPECT_EQ(index.nbits(), 32);
            index.construct(data.data(), centroid.data(), clusters.data(), false, 1);
            for (bool hacc : {false, true}) {
                std::vector<PID> ids(kNum);
                std::vector<float> distances(kNum);
                index.search(query.data(), kNum, 1, ids.data(), distances.data(), hacc);
                for (size_t i = 0; i < kNum; ++i) {
                    ASSERT_LT(ids[i], kNum);
                    float expected = metric == METRIC_L2 ? 0.0F : 1.0F;
                    for (size_t j = 0; j < kDim; ++j) {
                        float value = data[ids[i] * kDim + j];
                        float delta = value - query[j];
                        expected += metric == METRIC_L2 ? delta * delta : -value * query[j];
                    }
                    EXPECT_NEAR(distances[i], expected, 2e-5F);
                }
                EXPECT_TRUE(std::is_sorted(distances.begin(), distances.end()));
                std::vector<PID> ids_only(kNum);
                index.search(query.data(), kNum, 1, ids_only.data(), hacc);
                EXPECT_EQ(ids, ids_only);
            }
        }
    }
}

TEST(IvfSearchTest, AutomaticHighAccuracyMatchesBitPolicy) {
    constexpr size_t kNum = 33;
    constexpr size_t kDim = 65;
    std::vector<float> data(kNum * kDim);
    std::vector<float> centroid(kDim, 0);
    std::vector<float> query(kDim);
    std::vector<PID> clusters(kNum, 0);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 71) / 71.0F;
    }
    for (size_t i = 0; i < kDim; ++i) {
        query[i] = static_cast<float>(i % 13) / 13.0F;
    }
    for (size_t bits : {1UL, 2UL, 3UL, 4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 32UL}) {
        IVF index(kNum, kDim, 1, bits);
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
    constexpr size_t kNum = 2;
    constexpr size_t kDim = 64;
    std::array<float, kNum * kDim> data{};
    std::array<float, kNum * kDim> centroids{};
    std::array<float, kDim> query{};
    std::array<PID, kNum> clusters{1, 1};
    std::fill(data.begin(), data.end(), 1.0F);
    std::fill(centroids.begin() + static_cast<ptrdiff_t>(kDim), centroids.end(), 1.0F);

    IVF index(kNum, kDim, kNum, 32, MetricType::METRIC_L2, RotatorType::MatrixRotator);
    index.construct(data.data(), centroids.data(), clusters.data(), false, 1);

    PID result = kPidMax;
    float distance = -123.0F;
    index.search(query.data(), 1, 1, &result, &distance);
    EXPECT_EQ(result, kPidMax);
    EXPECT_EQ(distance, std::numeric_limits<float>::infinity());
}

TEST(IvfSearchTest, RoutesInnerProductQueriesByInnerProduct) {
    constexpr size_t kNum = 2;
    constexpr size_t kDim = 64;
    std::vector<float> data(kNum * kDim, 0.0F);
    std::vector<float> centroids(kNum * kDim, 0.0F);
    std::array<PID, kNum> clusters{0, 1};
    std::array<float, kDim> query{};
    data[0] = centroids[0] = 0.9F;
    data[kDim] = centroids[kDim] = 100.0F;
    query[0] = 1.0F;

    IVF index(kNum, kDim, kNum, 32, METRIC_IP);
    index.construct(data.data(), centroids.data(), clusters.data(), false, 1);

    PID result = kPidMax;
    float distance = 0.0F;
    index.search(query.data(), 1, 1, &result, &distance);
    EXPECT_EQ(result, 1U);
    EXPECT_NEAR(distance, -99.0F, 1e-5F);
}

TEST(IvfSearchTest, RoutesInnerProductQueriesWithHNSWCentroids) {
    constexpr size_t kDim = 64;
    constexpr size_t kCount = 20000;
    std::array<float, kDim> data{};
    std::vector<float> centroids(kCount * kDim, 0.0F);
    std::array<float, kDim> query{};
    for (size_t i = 0; i < kCount; ++i) {
        centroids[i * kDim] = 0.5F;
    }
    data[0] = centroids[0] = 100.0F;
    query[0] = 1.0F;
    const PID cluster = 0;

    IVF index(1, kDim, kCount, 32, METRIC_IP);
    index.construct(data.data(), centroids.data(), &cluster, false, 4);
    PID result = kPidMax;
    float distance = std::numeric_limits<float>::infinity();
    index.search(query.data(), 1, 1, &result, &distance);
    EXPECT_EQ(result, 0U);
    EXPECT_NEAR(distance, -99.0F, 1e-5F);
}

TEST(IvfSearchTest, InnerProductRawRerankingUsesResidualNormForPruning) {
    constexpr size_t kNum = 65;
    constexpr size_t kDim = 64;
    constexpr size_t kTopK = 1;
    std::vector<float> data(kNum * kDim);
    std::vector<float> centroids(2 * kDim, 0.0F);
    std::vector<PID> clusters(kNum, 1);
    std::array<float, kDim> query{};
    query[0] = 1.0F;
    centroids[0] = 100.0F;
    centroids[kDim] = 50.0F;

    for (size_t id = 0; id < kNum; ++id) {
        clusters[id] = id < 32 ? 0 : 1;
        data[id * kDim] = id < 32 ? 10.0F + static_cast<float>(id) * 0.01F : 0.0F;
        for (size_t d = 1; d < kDim; ++d) {
            data[id * kDim + d] =
                std::sin(static_cast<float>((id + 1) * (d + 3)) * 0.17F) * 20.0F;
        }
    }
    data[32 * kDim] = 1000.0F;

    IVF index(kNum, kDim, 2, 32, METRIC_IP);
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
    constexpr size_t kNum = 33;
    constexpr size_t kDim = 65;
    std::vector<float> data(kNum * kDim, 1.0F);
    std::vector<float> centroid(kDim, 0.0F);
    std::vector<PID> clusters(kNum, 0);
    const std::string path = ::testing::TempDir() + "rabitq_ivf_storage.index";
    IVF loaded;
    for (auto rotator : {RotatorType::FhtKacRotator, RotatorType::MatrixRotator}) {
        for (size_t bits : {32UL, 4UL, 32UL, 1UL}) {
            IVF index(kNum, kDim, 1, bits, METRIC_L2, rotator);
            index.construct(data.data(), centroid.data(), clusters.data(), false, 1);
            index.save(path.c_str());
            loaded.load(path.c_str());
            EXPECT_EQ(loaded.nbits(), bits);
            EXPECT_EQ(loaded.rotator_type(), rotator);
            std::vector<PID> ids(kNum);
            std::vector<PID> loaded_ids(kNum);
            std::vector<float> distances(kNum);
            std::vector<float> loaded_distances(kNum);
            index.search(centroid.data(), kNum, 1, ids.data(), distances.data(), true);
            loaded.search(
                centroid.data(), kNum, 1, loaded_ids.data(), loaded_distances.data(), true
            );
            EXPECT_EQ(ids, loaded_ids);
            EXPECT_EQ(distances, loaded_distances);
        }
    }
    std::remove(path.c_str());
}

TEST(IvfPersistenceTest, FailedLoadPreservesExistingIndex) {
    constexpr size_t kTargetNum = 2;
    constexpr size_t kTargetDim = 64;
    std::vector<float> target_data(kTargetNum * kTargetDim, 1.0F);
    std::vector<float> target_centroid(kTargetDim, 0.0F);
    std::array<PID, kTargetNum> target_clusters{0, 0};
    std::fill(
        target_data.begin() + static_cast<ptrdiff_t>(kTargetDim), target_data.end(), 2.0F
    );
    const std::string path = ::testing::TempDir() + "rabitq_ivf_truncated.index";

    IVF target(kTargetNum, kTargetDim, 1, 32, METRIC_IP, RotatorType::MatrixRotator);
    target.construct(
        target_data.data(), target_centroid.data(), target_clusters.data(), false, 1
    );
    std::array<PID, kTargetNum> expected_ids{};
    std::array<float, kTargetNum> expected_distances{};
    target.search(
        target_centroid.data(),
        kTargetNum,
        1,
        expected_ids.data(),
        expected_distances.data()
    );

    constexpr size_t kSourceNum = 33;
    constexpr size_t kSourceDim = 65;
    std::vector<float> source_data(kSourceNum * kSourceDim, 0.25F);
    std::vector<float> source_centroid(kSourceDim, 0.5F);
    std::vector<PID> source_clusters(kSourceNum, 0);
    IVF source(kSourceNum, kSourceDim, 1, 4, METRIC_L2, RotatorType::FhtKacRotator);
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
    EXPECT_EQ(target.max_elements(), kTargetNum);
    EXPECT_EQ(target.dimension(), kTargetDim);
    EXPECT_EQ(target.num_clusters(), 1U);
    EXPECT_EQ(target.nbits(), 32U);
    EXPECT_EQ(target.metric_type(), METRIC_IP);
    EXPECT_EQ(target.rotator_type(), RotatorType::MatrixRotator);

    std::array<PID, kTargetNum> actual_ids{};
    std::array<float, kTargetNum> actual_distances{};
    target.search(
        target_centroid.data(), kTargetNum, 1, actual_ids.data(), actual_distances.data()
    );
    EXPECT_EQ(actual_ids, expected_ids);
    EXPECT_EQ(actual_distances, expected_distances);

    std::remove(path.c_str());
}

TEST(IvfPersistenceTest, RejectsPointIdsUsingTheSearchBufferMarker) {
    constexpr size_t kNum = 33;
    constexpr size_t kDim = 64;
    std::vector<float> data(kNum * kDim, 0.25F);
    std::vector<float> centroid(kDim, 0.0F);
    std::vector<PID> clusters(kNum, 0);
    const std::string path = ::testing::TempDir() + "rabitq_ivf_bad_id.index";

    IVF source(kNum, kDim, 1, 4);
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
    std::remove(path.c_str());
}

// ── add / remove ────────────────────────────────────────────────────────────────────────

constexpr size_t kDynDim = 65;  // pads to 128, so padded and original dimensions differ
constexpr size_t kDynClusters = 4;

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
        for (size_t c = 0; c < kDynClusters; ++c) {
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
        index.search(query, k, kDynClusters, hits.ids.data(), hits.distances.data(), *hacc);
    } else {
        index.search(query, k, kDynClusters, hits.ids.data(), hits.distances.data());
    }
    return hits;
}

TEST(IvfAddTest, EveryWayOfAddingTheSamePointsGivesByteIdenticalIndexes) {
    constexpr size_t kOld = 200;
    constexpr size_t kNew = 200;
    const auto points = random_rows(kOld + kNew, 11);
    const auto centroids = random_rows(kDynClusters, 12);
    const float* extra = points.data() + (kOld * kDynDim);
    const std::string base_path = ::testing::TempDir() + "rabitq_ivf_add_base.index";
    const std::string path = ::testing::TempDir() + "rabitq_ivf_add.index";

    for (auto metric : {METRIC_L2, METRIC_IP}) {
        const auto assigned =
            nearest_centroids(centroids, points.data(), kOld + kNew, metric);
        for (size_t bits : {1UL, 4UL, 9UL, 32UL}) {
            SCOPED_TRACE(::testing::Message() << "metric=" << metric << " bits=" << bits);
            IVF base(kOld, kDynDim, kDynClusters, bits, metric);
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

            IVF threaded;
            threaded.load(base_path.c_str());
            threaded.add(extra, kNew, nullptr, false, 4);
            EXPECT_TRUE(saved_bytes(threaded, path) == expected) << "threaded";
        }
    }
    std::remove(base_path.c_str());
    std::remove(path.c_str());
}

TEST(IvfAddTest, KeepsExistingDistancesAndQuantizesNewPointsAccurately) {
    constexpr size_t kOld = 150;
    constexpr size_t kNew = 150;
    constexpr size_t kQueries = 8;
    const auto points = random_rows(kOld + kNew, 21);
    const auto centroids = random_rows(kDynClusters, 22);
    const auto queries = random_rows(kQueries, 23);

    for (auto metric : {METRIC_L2, METRIC_IP}) {
        const auto assigned =
            nearest_centroids(centroids, points.data(), kOld + kNew, metric);
        for (size_t bits : {1UL, 2UL, 4UL, 9UL, 32UL}) {
            SCOPED_TRACE(::testing::Message() << "metric=" << metric << " bits=" << bits);
            IVF index(kOld, kDynDim, kDynClusters, bits, metric);
            index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
            std::vector<Hits> before;
            for (size_t q = 0; q < kQueries; ++q) {
                before.push_back(search_everything(index, queries.data() + (q * kDynDim)));
            }

            index.add(points.data() + (kOld * kDynDim), kNew, nullptr, false, 1);

            double old_error = 0;
            double new_error = 0;
            for (size_t q = 0; q < kQueries; ++q) {
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
            old_error /= kQueries * kOld;
            new_error /= kQueries * kNew;
            // Mis-encoded points would be off by the spread of the data, not by a factor.
            EXPECT_LT(new_error, 3 * old_error + 1e-4) << old_error << " vs " << new_error;
        }
    }
}

TEST(IvfAddTest, AddedRawVectorsAreRerankedExactly) {
    constexpr size_t kOld = 40;
    constexpr size_t kNew = 60;
    const auto points = random_rows(kOld + kNew, 31);
    const auto centroids = random_rows(kDynClusters, 32);
    const auto queries = random_rows(3, 33);
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        IVF index(kOld, kDynDim, kDynClusters, 32, metric);
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
    const auto centroids = random_rows(kDynClusters, 42);
    const auto assigned =
        nearest_centroids(centroids, points.data(), kOld + kNew, METRIC_L2);
    const std::string base_path = ::testing::TempDir() + "rabitq_ivf_add_faster_base.index";
    const std::string path = ::testing::TempDir() + "rabitq_ivf_add_faster.index";

    IVF base(kOld, kDynDim, kDynClusters, 5);
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
    std::remove(base_path.c_str());
    std::remove(path.c_str());
}

TEST(IvfAddTest, RoutesNewPointsWithHNSWCentroids) {
    constexpr size_t kDim = 64;
    constexpr size_t kCount = 20000;
    // Centroid i has first coordinate i + 1, so an inner product with e0 is largest for the
    // last centroid and the routing decision is unambiguous.
    std::vector<float> centroids(kCount * kDim, 0.0F);
    for (size_t i = 0; i < kCount; ++i) {
        centroids[i * kDim] = static_cast<float>(i + 1);
    }
    std::array<float, kDim> first{};
    first[0] = 1.0F;
    const PID cluster = 0;
    IVF index(1, kDim, kCount, 32, METRIC_IP);
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
    constexpr size_t kNum = 70;
    const auto points = random_rows(kNum + 1, 51);
    const auto centroids = random_rows(kDynClusters, 52);
    const auto assigned = nearest_centroids(centroids, points.data(), kNum, METRIC_L2);
    const std::string path = ::testing::TempDir() + "rabitq_ivf_add_reject.index";

    IVF unbuilt(kNum, kDynDim, kDynClusters, 4);
    EXPECT_THROW(unbuilt.add(points.data(), 1), std::logic_error);
    IVF unconfigured;
    EXPECT_THROW(unconfigured.add(points.data(), 1), std::logic_error);

    IVF index(kNum, kDynDim, kDynClusters, 4);
    index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
    const std::string expected = saved_bytes(index, path);

    const std::array<PID, 2> bad_clusters{0, static_cast<PID>(kDynClusters)};
    EXPECT_THROW(index.add(points.data(), 2, bad_clusters.data()), std::invalid_argument);
    EXPECT_THROW(index.add(nullptr, 1), std::invalid_argument);
    // The ID check happens before the data is read.
    EXPECT_THROW(
        index.add(points.data(), buffer::kSearchBufferMaxPointCount), std::invalid_argument
    );
    EXPECT_NO_THROW(index.add(nullptr, 0));
    EXPECT_NO_THROW(index.add(points.data(), 0));
    EXPECT_EQ(index.max_elements(), kNum);
    EXPECT_TRUE(saved_bytes(index, path) == expected);
    std::remove(path.c_str());
}

TEST(IvfRemoveTest, ExcludesRemovedPointsOnEveryScanPath) {
    constexpr size_t kNum = 200;
    constexpr size_t kQueries = 5;
    const auto points = random_rows(kNum, 61);
    const auto centroids = random_rows(kDynClusters, 62);
    const auto queries = random_rows(kQueries, 63);

    std::vector<PID> removed_ids;
    std::vector<bool> is_removed(kNum, false);
    for (PID id = 0; id < kNum; id += 3) {
        removed_ids.push_back(id);
        is_removed[id] = true;
    }
    removed_ids.push_back(kNum - 1);  // the last point sits in a partial batch
    is_removed[kNum - 1] = true;
    removed_ids.push_back(0);  // repeated IDs count once
    const size_t unique_removed =
        static_cast<size_t>(std::count(is_removed.begin(), is_removed.end(), true));

    for (auto metric : {METRIC_L2, METRIC_IP}) {
        const auto assigned = nearest_centroids(centroids, points.data(), kNum, metric);
        // 1-3 bits use standard FastScan, 4-9 bits and raw storage exercise other paths too
        for (size_t bits : {1UL, 2UL, 3UL, 4UL, 8UL, 9UL, 32UL}) {
            for (bool hacc : {false, true}) {
                SCOPED_TRACE(
                    ::testing::Message()
                    << "metric=" << metric << " bits=" << bits << " hacc=" << hacc
                );
                IVF index(kNum, kDynDim, kDynClusters, bits, metric);
                index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
                std::vector<Hits> before;
                for (size_t q = 0; q < kQueries; ++q) {
                    before.push_back(
                        search_everything(index, queries.data() + (q * kDynDim), hacc)
                    );
                }

                EXPECT_EQ(
                    index.remove(removed_ids.data(), removed_ids.size()), unique_removed
                );
                EXPECT_EQ(index.remove(removed_ids.data(), removed_ids.size()), 0U);
                EXPECT_EQ(index.max_elements(), kNum);

                for (size_t q = 0; q < kQueries; ++q) {
                    const auto after =
                        search_everything(index, queries.data() + (q * kDynDim), hacc);
                    // Survivors keep their order and distances; the rest are sentinels.
                    size_t next = 0;
                    for (size_t r = 0; r < kNum; ++r) {
                        if (is_removed[before[q].ids[r]]) {
                            continue;
                        }
                        ASSERT_EQ(after.ids[next], before[q].ids[r]);
                        ASSERT_EQ(after.distances[next], before[q].distances[r]);
                        ++next;
                    }
                    ASSERT_EQ(next, kNum - unique_removed);
                    for (size_t r = next; r < kNum; ++r) {
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
                        kDynClusters,
                        top.data(),
                        top_distances.data(),
                        hacc
                    );
                    for (size_t r = 0; r < top.size(); ++r) {
                        ASSERT_LT(top[r], kNum);
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
    const auto points = random_rows(kOld + kNew, 71);
    const auto centroids = random_rows(kDynClusters, 72);
    const auto queries = random_rows(4, 73);
    const auto assigned =
        nearest_centroids(centroids, points.data(), kOld + kNew, METRIC_L2);
    const std::string path = ::testing::TempDir() + "rabitq_ivf_remove.index";

    for (size_t bits : {1UL, 4UL, 32UL}) {
        SCOPED_TRACE(::testing::Message() << "bits=" << bits);
        IVF index(kOld, kDynDim, kDynClusters, bits);
        index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
        const std::vector<PID> removed{1, 33, 34, 98, 99};
        ASSERT_EQ(index.remove(removed.data(), removed.size()), removed.size());
        index.save(path.c_str());

        IVF loaded;
        loaded.load(path.c_str());
        for (size_t q = 0; q < 4; ++q) {
            const float* query = queries.data() + (q * kDynDim);
            const auto expected = search_everything(index, query);
            const auto actual = search_everything(loaded, query);
            EXPECT_EQ(actual.ids, expected.ids);
            EXPECT_EQ(actual.distances, expected.distances);
        }
        EXPECT_EQ(loaded.remove(removed.data(), removed.size()), 0U);

        // Adding repacks the tail batches that hold removed points; they must stay removed.
        loaded.add(points.data() + (kOld * kDynDim), kNew, nullptr, false, 1);
        for (size_t q = 0; q < 4; ++q) {
            const auto hits = search_everything(loaded, queries.data() + (q * kDynDim));
            size_t live = 0;
            for (size_t r = 0; r < hits.ids.size(); ++r) {
                if (hits.ids[r] == kPidMax) {
                    EXPECT_EQ(hits.distances[r], std::numeric_limits<float>::infinity());
                    continue;
                }
                ++live;
                EXPECT_EQ(std::count(removed.begin(), removed.end(), hits.ids[r]), 0);
            }
            EXPECT_EQ(live, kOld + kNew - removed.size());
        }
    }
    std::remove(path.c_str());
}

TEST(IvfRemoveTest, RejectsBadInputAndRemovesNothingOnFailure) {
    constexpr size_t kNum = 70;
    const auto points = random_rows(kNum, 81);
    const auto centroids = random_rows(kDynClusters, 82);
    const auto assigned = nearest_centroids(centroids, points.data(), kNum, METRIC_L2);
    const std::string path = ::testing::TempDir() + "rabitq_ivf_remove_reject.index";

    const PID id = 0;
    IVF unbuilt(kNum, kDynDim, kDynClusters, 4);
    EXPECT_THROW(unbuilt.remove(&id, 1), std::logic_error);

    IVF index(kNum, kDynDim, kDynClusters, 4);
    index.construct(points.data(), centroids.data(), assigned.data(), false, 1);
    const std::string expected = saved_bytes(index, path);

    const std::array<PID, 2> out_of_range{3, static_cast<PID>(kNum)};
    EXPECT_THROW(index.remove(out_of_range.data(), 2), std::invalid_argument);
    EXPECT_THROW(index.remove(nullptr, 1), std::invalid_argument);
    EXPECT_EQ(index.remove(nullptr, 0), 0U);
    EXPECT_TRUE(saved_bytes(index, path) == expected);

    // Removing everything leaves a searchable index that returns only sentinels.
    std::vector<PID> everything(kNum);
    for (size_t i = 0; i < kNum; ++i) {
        everything[i] = static_cast<PID>(i);
    }
    EXPECT_EQ(index.remove(everything.data(), kNum), kNum);
    const auto hits = search_everything(index, points.data());
    for (size_t r = 0; r < kNum; ++r) {
        EXPECT_EQ(hits.ids[r], kPidMax);
    }
    std::remove(path.c_str());
}

}  // namespace
}  // namespace rabitqlib::ivf
