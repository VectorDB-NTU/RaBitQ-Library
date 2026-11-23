// Build only against the pinned v0.9.0 source, never the current checkout. Writes index
// format 2 through the 0.9 index classes, and through ivf::IVF and QuantizedGraph for the
// matrix rotator, which the new classes do not offer.
#include <cmath>
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
#include "rabitqlib/index/hnsw_index.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/ivf/types.hpp"
#include "rabitqlib/index/ivf_index.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/index/symqg_index.hpp"
#include "rabitqlib/index/types.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        throw std::invalid_argument("Pass the fixture output directory");
    }
    using namespace rabitqlib;
    // The v0.8.3 fixtures' data: cluster sizes 0, 1, 32 and 97.
    constexpr size_t kCount = 130, kDim = 65, kClusters = 4, kQueries = 3, kTopK = 5;
    constexpr uint32_t kSeed = 42;
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
    // Cosine indexes scale data and queries to unit length; give them unit centroids too.
    std::vector<float> unit_centroids(centroids);
    for (size_t c = 0; c < kClusters; ++c) {
        float* row = unit_centroids.data() + (c * kDim);
        double norm = 0;
        for (size_t j = 0; j < kDim; ++j) {
            norm += static_cast<double>(row[j]) * row[j];
        }
        const auto scale = static_cast<float>(1.0 / std::sqrt(norm));
        for (size_t j = 0; j < kDim; ++j) {
            row[j] *= scale;
        }
    }
    const auto centroids_for = [&](MetricType metric) {
        return metric == METRIC_COSINE ? unit_centroids.data() : centroids.data();
    };
    std::vector<PID> clusters(kCount, 3);
    clusters[0] = 1;
    for (size_t i = 1; i < 33; ++i) {
        clusters[i] = 2;
    }
    // Point 0 is the only point of cluster 1. Point 32 is the nearest neighbour of the
    // third query, so search results show whether removal markers are honoured.
    const Id removed[] = {0, 32};
    const auto path_of = [&](const std::string& name) {
        return (directory / (name + ".index")).string();
    };
    std::vector<Id> ids(kQueries * kTopK);
    std::vector<float> distances(ids.size());
    const auto write_expected = [&](const std::string& name) {
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
    AddOptions add_options;
    add_options.cluster_ids = clusters.data();
    add_options.num_threads = 1;

    const struct {
        const char* name;
        size_t nbits;
        MetricType metric;
        ivf::Routing routing;
    } ivf_cases[] = {
        {"ivf4_flat_l2", 4, METRIC_L2, ivf::Routing::Flat},
        {"ivf7_flatrabitq_cosine", 7, METRIC_COSINE, ivf::Routing::FlatRaBitQ},
        {"ivf32_hnsw_ip", 32, METRIC_IP, ivf::Routing::HNSW},
    };
    for (const auto& c : ivf_cases) {
        IvfOptions options;
        options.metric = c.metric;
        options.nbits = c.nbits;
        options.routing = c.routing;
        options.seed = kSeed;
        IvfIndex index(kDim, kClusters, options);
        index.set_centroids(centroids_for(c.metric), 1);
        index.add(data.data(), kCount, add_options);
        index.remove(removed, 2);
        index.save(path_of(c.name).c_str());
        IvfSearchOptions search;
        search.nprobe = kClusters;
        index.search(queries.data(), kQueries, kTopK, ids.data(), distances.data(), search);
        write_expected(c.name);
    }
    {
        ivf::IVF index(
            kDim,
            kClusters,
            1,
            METRIC_L2,
            kSeed,
            RotatorType::MatrixRotator,
            ivf::InitializerType::Flat
        );
        index.set_centroids(centroids.data(), 1);
        index.add(data.data(), kCount, clusters.data(), false, 1);
        const PID legacy_removed[] = {0, 32};
        index.remove(legacy_removed, 2);
        index.save(path_of("ivf1_flat_matrix_l2").c_str());
        std::vector<PID> legacy_ids(ids.size());
        index.search_batch(
            queries.data(), kQueries, kTopK, kClusters, legacy_ids.data(), distances.data()
        );
        ids.assign(legacy_ids.begin(), legacy_ids.end());
        write_expected("ivf1_flat_matrix_l2");
    }

    GraphSearchOptions graph_search;
    graph_search.ef_search = 64;
    const struct {
        const char* name;
        size_t nbits;
        MetricType metric;
    } hnsw_cases[] = {
        {"hnsw1_l2", 1, METRIC_L2},
        {"hnsw5_cosine", 5, METRIC_COSINE},
        {"hnsw9_ip", 9, METRIC_IP},
    };
    for (const auto& c : hnsw_cases) {
        HnswOptions options;
        options.metric = c.metric;
        options.nbits = c.nbits;
        options.num_clusters = kClusters;
        options.M = 8;
        options.ef_construction = 64;
        options.seed = kSeed;
        HnswIndex index(kDim, options);
        index.set_centroids(centroids_for(c.metric));
        index.add(data.data(), kCount, add_options);
        index.remove(removed, 2);
        index.save(path_of(c.name).c_str());
        index.search(
            queries.data(), kQueries, kTopK, ids.data(), distances.data(), graph_search
        );
        write_expected(c.name);
    }

    const struct {
        const char* name;
        size_t nbits;
        MetricType metric;
    } qg_cases[] = {
        {"qg32_ip", 32, METRIC_IP},
        {"qg4_cosine", 4, METRIC_COSINE},
        {"qg8_l2", 8, METRIC_L2},
    };
    for (const auto& c : qg_cases) {
        SymqgOptions options;
        options.metric = c.metric;
        options.nbits = c.nbits;
        options.ef_construction = 64;
        options.seed = kSeed;
        SymqgIndex index(kDim, options);
        index.add(data.data(), kCount, 1);
        index.save(path_of(c.name).c_str());
        index.search(
            queries.data(), kQueries, kTopK, ids.data(), distances.data(), graph_search
        );
        write_expected(c.name);
    }
    {
        symqg::QuantizedGraph<float> index(
            kCount, kDim, 32, METRIC_L2, RotatorType::MatrixRotator, 4, kSeed
        );
        symqg::QGBuilder builder(
            index, 64, data.data(), 1, symqg::QGInitialization::PiPNN, kSeed
        );
        builder.build();
        index.save(path_of("qg4_matrix_l2").c_str());
        std::vector<PID> legacy_ids(ids.size());
        index.set_ef(64);
        index.search_batch(
            queries.data(), kQueries, kTopK, legacy_ids.data(), distances.data(), 1
        );
        ids.assign(legacy_ids.begin(), legacy_ids.end());
        write_expected("qg4_matrix_l2");
    }
    return 0;
}
