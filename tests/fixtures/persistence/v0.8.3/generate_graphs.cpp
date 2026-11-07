// Build only against the pinned v0.8.3 source, never the current checkout. Writes the
// HNSW-RaBitQ and SymphonyQG fixtures; generate.cpp writes the IVF-RaBitQ ones.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/types.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/index/types.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        throw std::invalid_argument("Pass the fixture output directory");
    }
    using namespace rabitqlib;
    // The IVF fixtures' data: cluster sizes 0, 1, 32 and 97.
    constexpr size_t kCount = 130, kDim = 65, kClusters = 4, kQueries = 3, kTopK = 5;
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    uint32_t state = 713;
    const auto next = [&state] {
        state = state * 1664525U + 1013904223U;
        return static_cast<float>(static_cast<int>((state >> 8) % 2001) - 1000) / 256.0F;
    };
    std::vector<float> data(kCount * kDim), queries(kQueries * kDim);
    std::vector<float> centroids(kClusters * kDim);
    for (auto& value : data) {
        value = next();
    }
    for (auto& value : queries) {
        value = next();
    }
    for (auto& value : centroids) {
        value = next();
    }
    std::vector<PID> clusters(kCount, 3);
    clusters[0] = 1;
    for (size_t i = 1; i < 33; ++i) {
        clusters[i] = 2;
    }

    const auto write_expected = [&](const std::string& name,
                                    const std::vector<PID>& ids,
                                    const std::vector<float>& distances) {
        std::ofstream expected(directory / (name + ".expected"), std::ios::binary);
        expected.exceptions(std::ios::failbit | std::ios::badbit);
        expected << std::setprecision(std::numeric_limits<float>::max_digits10);
        expected << kCount << ' ' << kDim << ' ' << kQueries << ' ' << kTopK << '\n';
        for (size_t i = 0; i < queries.size(); ++i) {
            expected << queries[i] << (i + 1 == queries.size() ? '\n' : ' ');
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            expected << ids[i] << ' ' << distances[i] << '\n';
        }
        expected.close();
    };

    const struct {
        const char* name;
        size_t total_bits;
        MetricType metric;
    } hnsw_cases[] = {
        {"hnsw1_l2", 1, METRIC_L2},
        {"hnsw5_ip", 5, METRIC_IP},
        {"hnsw9_l2", 9, METRIC_L2},
    };
    for (const auto& c : hnsw_cases) {
        const std::string name = c.name;
        hnsw::HierarchicalNSW index(kCount, kDim, c.total_bits, 8, 64, 42, c.metric);
        std::vector<PID> assigned = clusters;
        index.construct(kClusters, centroids.data(), kCount, data.data(), assigned.data(), 1);
        const PID removed = 0;
        index.remove(&removed, 1);
        index.save((directory / (name + ".index")).string().c_str());
        std::vector<PID> ids(kQueries * kTopK);
        std::vector<float> distances(ids.size());
        const auto result = index.search(queries.data(), kQueries, kTopK, 64, 1);
        for (size_t i = 0; i < kQueries; ++i) {
            for (size_t j = 0; j < kTopK; ++j) {
                ids[i * kTopK + j] = result[i][j].second;
                distances[i * kTopK + j] = result[i][j].first;
            }
        }
        write_expected(name, ids, distances);
    }

    const struct {
        const char* name;
        size_t bits;  // 0 stores raw vectors
        MetricType metric;
        RotatorType rotator;
    } qg_cases[] = {
        {"qg0_ip", 0, METRIC_IP, RotatorType::FhtKacRotator},
        {"qg4_l2", 4, METRIC_L2, RotatorType::FhtKacRotator},
        {"qg8_ip", 8, METRIC_IP, RotatorType::FhtKacRotator},
        {"qg4_matrix_l2", 4, METRIC_L2, RotatorType::MatrixRotator},
    };
    for (const auto& c : qg_cases) {
        const std::string name = c.name;
        symqg::QuantizedGraph<float> index(kCount, kDim, 32, c.metric, c.rotator, c.bits, 42);
        symqg::QGBuilder builder(index, 64, data.data(), 1, symqg::QGInitialization::PiPNN, 42);
        builder.build();
        index.save((directory / (name + ".index")).string().c_str());
        std::vector<PID> ids(kQueries * kTopK);
        std::vector<float> distances(ids.size());
        index.set_ef(64);
        index.search_batch(queries.data(), kQueries, kTopK, ids.data(), distances.data(), 1);
        write_expected(name, ids, distances);
    }
    return 0;
}
