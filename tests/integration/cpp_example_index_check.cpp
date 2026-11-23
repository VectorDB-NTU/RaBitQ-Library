#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/hnsw_index.hpp"
#include "rabitqlib/index/ivf_index.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/io.hpp"

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            throw std::invalid_argument("Expected kind, index, data, metric and bits");
        }
        const std::string kind(argv[1]);
        const std::string metric_name(argv[4]);
        const auto metric = metric_name == "ip"       ? rabitqlib::METRIC_IP
                            : metric_name == "cosine" ? rabitqlib::METRIC_COSINE
                                                      : rabitqlib::METRIC_L2;
        const size_t bits = std::stoull(argv[5]);
        rabitqlib::RowMajorArray<float> data;
        rabitqlib::load_vecs<float>(argv[3], data);
        const size_t count = data.rows();
        const size_t dim = data.cols();
        const auto check = [&](const auto& index, const auto& search) {
            if (index.dim() != dim || index.num_stored_points() != count ||
                index.num_points() != count || index.metric() != metric ||
                index.nbits() != bits) {
                throw std::runtime_error("Example index metadata does not match input");
            }
            // Unit vectors make self the exact winner for the IP and cosine fixtures too.
            for (size_t row : std::array<size_t, 3>{0, count / 2, count - 1}) {
                rabitqlib::Id id = rabitqlib::kNoId;
                float distance = 0;
                search(data.data() + row * dim, id, distance);
                if (id != static_cast<rabitqlib::Id>(row) || !std::isfinite(distance)) {
                    throw std::runtime_error("Example index did not retrieve its input");
                }
            }
        };
        if (kind == "ivf") {
            const auto index = rabitqlib::IvfIndex::load(argv[2]);
            rabitqlib::IvfSearchOptions options;
            options.nprobe = index.num_clusters();
            check(index, [&](const float* query, rabitqlib::Id& id, float& distance) {
                index.search(query, 1, 1, &id, &distance, options);
            });
        } else if (kind == "hnsw") {
            const auto index = rabitqlib::HnswIndex::load(argv[2]);
            rabitqlib::GraphSearchOptions options;
            options.ef_search = count;
            check(index, [&](const float* query, rabitqlib::Id& id, float& distance) {
                index.search(query, 1, 1, &id, &distance, options);
            });
        } else {
            throw std::invalid_argument("Expected ivf or hnsw");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
