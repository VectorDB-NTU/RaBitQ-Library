#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "rabitqlib/index/types.hpp"

namespace rabitqlib::hnsw {
class HierarchicalNSW;
namespace detail {
// Ordered results used by batch search and backend parity tests.
std::vector<std::pair<float, PID>> search_knn_sorted(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search = 0
);
std::vector<std::pair<float, PID>> search_knn_sorted_generic(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search = 0
);
std::vector<std::pair<float, PID>> search_knn_sorted_neon(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search = 0
);
std::vector<std::pair<float, PID>> search_knn_sorted_avx2(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search = 0
);
std::vector<std::pair<float, PID>> search_knn_sorted_avx512_core(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search = 0
);
std::vector<std::pair<float, PID>> search_knn_sorted_avx512_popcnt(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search = 0
);

}  // namespace detail
}  // namespace rabitqlib::hnsw
