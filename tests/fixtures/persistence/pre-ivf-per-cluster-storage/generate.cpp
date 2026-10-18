// Build only against the pinned pre-refactor source, never the current checkout.
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
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/types.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        throw std::invalid_argument("Pass the fixture output directory");
    }
    using namespace rabitqlib;
    // Cluster sizes 0, 1, 32 and 97 cover empty, single, full-batch and tail clusters.
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
    for (auto& value : data)
        value = next();
    for (auto& value : queries)
        value = next();
    for (auto& value : centroids)
        value = next();
    std::vector<PID> clusters(kCount, 3);
    clusters[0] = 1;
    for (size_t i = 1; i < 33; ++i)
        clusters[i] = 2;
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (const std::string kind : {"ivf4", "ivf32"}) {
            const std::string name = kind + (metric == METRIC_L2 ? "_l2" : "_ip");
            const auto path = (directory / (name + ".index")).string();
            std::vector<PID> ids(kQueries * kTopK);
            std::vector<float> distances(ids.size());
            ivf::IVF index(kCount, kDim, kClusters, kind == "ivf32" ? 32 : 4, metric);
            index.construct(data.data(), centroids.data(), clusters.data(), false, 1);
            const PID removed = 0;
            index.remove(&removed, 1);
            index.save(path.c_str());
            index.search_batch(
                queries.data(), kQueries, kTopK, kClusters, ids.data(), distances.data()
            );
            std::ofstream expected(directory / (name + ".expected"), std::ios::binary);
            expected.exceptions(std::ios::failbit | std::ios::badbit);
            expected << std::setprecision(std::numeric_limits<float>::max_digits10);
            expected << kCount << ' ' << kDim << ' ' << kQueries << ' ' << kTopK << '\n';
            for (size_t i = 0; i < queries.size(); ++i)
                expected << queries[i] << (i + 1 == queries.size() ? '\n' : ' ');
            for (size_t i = 0; i < ids.size(); ++i)
                expected << ids[i] << ' ' << distances[i] << '\n';
            expected.close();
        }
    }
    return 0;
}
