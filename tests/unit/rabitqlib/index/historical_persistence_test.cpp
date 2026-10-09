#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/symqg/qg.hpp"

namespace rabitqlib {
namespace {
class HistoricalPersistenceTest : public ::testing::TestWithParam<std::string> {};

TEST_P(HistoricalPersistenceTest, LoadsSearchesResavesAndPreservesStateAfterFailedLoad) {
    const std::filesystem::path directory(RABITQ_TEST_FIXTURE_DIR);
    const auto name = GetParam();
    const auto path = (directory / (name + ".index")).string();
    std::ifstream expected(directory / (name + ".expected"));
    ASSERT_TRUE(expected.is_open());
    size_t count = 0, dim = 0, nq = 0, k = 0;
    expected >> count >> dim >> nq >> k;
    ASSERT_EQ(count, 33U);
    ASSERT_EQ(dim, 65U);
    ASSERT_EQ(nq, 3U);
    ASSERT_EQ(k, 5U);
    std::vector<float> queries(nq * dim), distances(nq * k), reference_distances(nq * k);
    std::vector<PID> ids(nq * k), reference_ids(nq * k);
    for (auto& value : queries)
        expected >> value;
    for (size_t i = 0; i < ids.size(); ++i)
        expected >> reference_ids[i] >> reference_distances[i];
    ASSERT_TRUE(expected.good());
    const auto output =
        std::filesystem::temp_directory_path() / ("rabitq_historical_" + name);
    const auto compare = [&] {
        EXPECT_EQ(ids, reference_ids);
        for (size_t i = 0; i < ids.size(); ++i) {
            EXPECT_NEAR(
                distances[i],
                reference_distances[i],
                2e-4F * std::max(1.0F, std::abs(reference_distances[i]))
            );
        }
    };
    const auto exercise = [&](auto& index, const auto& search) {
        index.load(path.c_str());
        EXPECT_EQ(index.dimension(), dim);
        const auto metric = name.find("_ip") != std::string::npos ? METRIC_IP : METRIC_L2;
        EXPECT_EQ(index.metric_type(), metric);
        search();
        compare();
        index.save(output.string().c_str());
        index.load(output.string().c_str());
        search();
        compare();
        std::filesystem::resize_file(output, std::filesystem::file_size(output) - 1);
        EXPECT_THROW(index.load(output.string().c_str()), std::runtime_error);
        search();
        compare();
        std::filesystem::remove(output);
    };
    if (name.find("ivf") == 0) {
        ivf::IVF index;
        exercise(index, [&] {
            EXPECT_EQ(index.padded_dim(), 128U);
            EXPECT_EQ(index.initializer_type(), ivf::InitializerType::Flat);
            index.search_batch(queries.data(), nq, k, 1, ids.data(), distances.data());
        });
    } else if (name.find("hnsw") == 0) {
        hnsw::HierarchicalNSW index;
        exercise(index, [&] {
            const auto results = index.search(queries.data(), nq, k, 64, 1);
            for (size_t i = 0; i < nq; ++i) {
                ASSERT_EQ(results[i].size(), k);
                for (size_t j = 0; j < k; ++j) {
                    ids[i * k + j] = results[i][j].second;
                    distances[i * k + j] = results[i][j].first;
                }
            }
        });
    } else {
        symqg::QuantizedGraph<float> index;
        exercise(index, [&] {
            EXPECT_EQ(index.padded_dim(), 128U);
            index.set_ef(64);
            index.search_batch(queries.data(), nq, k, ids.data(), distances.data(), 1);
        });
    }
}

INSTANTIATE_TEST_SUITE_P(
    V052,
    HistoricalPersistenceTest,
    ::testing::Values(
        "ivf4_l2",
        "ivf4_ip",
        "ivf32_l2",
        "ivf32_ip",
        "hnsw4_l2",
        "hnsw4_ip",
        "qg0_l2",
        "qg0_ip",
        "qg4_l2",
        "qg4_ip",
        "qg8_l2",
        "qg8_ip"
    ),
    [](const auto& info) { return info.param; }
);
}  // namespace
}  // namespace rabitqlib
