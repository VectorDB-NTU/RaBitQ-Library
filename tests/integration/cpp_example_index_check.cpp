#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/io.hpp"

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            throw std::invalid_argument("Expected kind, index, data, metric and bits");
        }
        const std::string kind(argv[1]);
        const auto metric =
            std::string(argv[4]) == "ip" ? rabitqlib::METRIC_IP : rabitqlib::METRIC_L2;
        const size_t bits = std::stoull(argv[5]);
        rabitqlib::RowMajorArray<float> data;
        rabitqlib::load_vecs<float>(argv[3], data);
        const size_t count = data.rows();
        const size_t dim = data.cols();
        const auto check = [&](const auto& index, const auto& search) {
            if (index.dimension() != dim || index.num_stored_points() != count ||
                index.num_points() != count || index.metric_type() != metric ||
                index.nbits() != bits) {
                throw std::runtime_error("Example index metadata does not match input");
            }
            // Unit vectors make self the exact winner for the IP fixture as well.
            for (size_t row : std::array<size_t, 3>{0, count / 2, count - 1}) {
                rabitqlib::PID id = rabitqlib::kPidMax;
                float distance = 0;
                search(data.data() + row * dim, id, distance);
                if (id != row || !std::isfinite(distance)) {
                    throw std::runtime_error("Example index did not retrieve its input");
                }
            }
        };
        if (kind == "ivf") {
            rabitqlib::ivf::IVF index;
            index.load(argv[2]);
            check(index, [&](const float* query, rabitqlib::PID& id, float& distance) {
                index.search(query, 1, index.num_clusters(), &id, &distance);
            });
        } else if (kind == "hnsw") {
            rabitqlib::hnsw::HierarchicalNSW index;
            index.load(argv[2]);
            check(index, [&](const float* query, rabitqlib::PID& id, float& distance) {
                const auto hits = index.search(query, 1, 1, count, 1);
                if (hits.size() != 1 || hits[0].size() != 1) {
                    throw std::runtime_error("Example index returned no neighbor");
                }
                distance = hits[0][0].first;
                id = hits[0][0].second;
            });
        } else {
            throw std::invalid_argument("Expected ivf or hnsw");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
