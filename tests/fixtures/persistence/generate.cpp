// Build only against the pinned historical source, never the current checkout.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        throw std::invalid_argument("Pass the fixture output directory");
    }
    using namespace rabitqlib;
    constexpr size_t kCount = 33, kDim = 65, kQueries = 3, kTopK = 5;
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    uint32_t state = 713;
    const auto next = [&state] {
        state = state * 1664525U + 1013904223U;
        return static_cast<float>(static_cast<int>((state >> 8) % 2001) - 1000) / 256.0F;
    };
    std::vector<float> data(kCount * kDim), queries(kQueries * kDim), centroid(kDim);
    for (auto& value : data)
        value = next();
    for (auto& value : queries)
        value = next();
    for (size_t i = 0; i < kCount; ++i) {
        for (size_t d = 0; d < kDim; ++d)
            centroid[d] += data[i * kDim + d] / kCount;
    }
    std::vector<PID> clusters(kCount, 0);
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (const std::string kind : {"ivf4", "ivf32", "hnsw4", "qg0", "qg4", "qg8"}) {
            const std::string name = kind + (metric == METRIC_L2 ? "_l2" : "_ip");
            const auto path = (directory / (name + ".index")).string();
            std::vector<PID> ids(kQueries * kTopK);
            std::vector<float> distances(ids.size());
            if (kind.substr(0, 3) == "ivf") {
                ivf::IVF index(kCount, kDim, 1, kind == "ivf32" ? 32 : 4, metric);
                index.construct(data.data(), centroid.data(), clusters.data(), false, 1);
                const PID removed = 0;
                index.remove(&removed, 1);
                index.save(path.c_str());
                index.search_batch(
                    queries.data(), kQueries, kTopK, 1, ids.data(), distances.data()
                );
            } else if (kind == "hnsw4") {
                hnsw::HierarchicalNSW index(kCount, kDim, 4, 8, 64, 42, metric);
                index.construct(
                    1, centroid.data(), kCount, data.data(), clusters.data(), 1
                );
                const PID removed = 0;
                index.remove(&removed, 1);
                index.save(path.c_str());
                const auto result = index.search(queries.data(), kQueries, kTopK, 64, 1);
                for (size_t i = 0; i < kQueries; ++i) {
                    for (size_t j = 0; j < kTopK; ++j) {
                        ids[i * kTopK + j] = result[i][j].second;
                        distances[i * kTopK + j] = result[i][j].first;
                    }
                }
            } else {
                const size_t bits = kind == "qg0" ? 0 : (kind == "qg4" ? 4 : 8);
                symqg::QuantizedGraph<float> index(
                    kCount, kDim, 32, metric, RotatorType::FhtKacRotator, bits, 42
                );
                symqg::QGBuilder builder(
                    index, 64, data.data(), 1, symqg::QGInitialization::PiPNN, 42
                );
                builder.build();
                index.save(path.c_str());
                index.set_ef(64);
                index.search_batch(
                    queries.data(), kQueries, kTopK, ids.data(), distances.data(), 1
                );
            }
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
}
