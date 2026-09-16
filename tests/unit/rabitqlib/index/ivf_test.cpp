#include "rabitqlib/index/ivf/ivf.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
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

}  // namespace
}  // namespace rabitqlib::ivf
