#include "rabitqlib/index/ivf/ivf.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace rabitqlib::ivf {
namespace {

TEST(IvfConfigurationTest, RejectsUnsupportedMetric) {
    EXPECT_THROW(
        (IVF(8, 64, 1, 1, static_cast<MetricType>(255), RotatorType::MatrixRotator)),
        std::invalid_argument
    );
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

}  // namespace
}  // namespace rabitqlib::ivf
