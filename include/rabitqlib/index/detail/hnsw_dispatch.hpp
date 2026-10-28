#pragma once

#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "rabitqlib/index/types.hpp"

namespace rabitqlib::hnsw {
class HierarchicalNSW;
namespace detail {
// Compatibility adapter; individual backends return ordered results only.
std::priority_queue<std::pair<float, PID>> search_knn(
    HierarchicalNSW&, const float*, size_t, size_t ef_search = 0
);

// Ordered results used by batch search and backend parity tests.
std::vector<std::pair<float, PID>> search_knn_sorted(
    HierarchicalNSW&, const float*, size_t, size_t ef_search = 0
);
std::vector<std::pair<float, PID>> search_knn_sorted_generic(
    HierarchicalNSW&, const float*, size_t, size_t ef_search = 0
);
std::vector<std::pair<float, PID>> search_knn_sorted_neon(
    HierarchicalNSW&, const float*, size_t, size_t ef_search = 0
);
std::vector<std::pair<float, PID>> search_knn_sorted_avx2(
    HierarchicalNSW&, const float*, size_t, size_t ef_search = 0
);
std::vector<std::pair<float, PID>> search_knn_sorted_avx512_core(
    HierarchicalNSW&, const float*, size_t, size_t ef_search = 0
);
std::vector<std::pair<float, PID>> search_knn_sorted_avx512_popcnt(
    HierarchicalNSW&, const float*, size_t, size_t ef_search = 0
);

}  // namespace detail
}  // namespace rabitqlib::hnsw
