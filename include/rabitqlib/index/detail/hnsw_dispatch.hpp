#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "rabitqlib/index/detail/visited_set_fwd.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib {
template <typename T>
class SplitSingleQuery;
namespace buffer {
template <typename T>
class SearchBuffer;
}  // namespace buffer
}  // namespace rabitqlib

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

// Graph traversal for one prepared query, compiled once per ISA. Query preparation,
// the visited set, the ef_search-sized search_pool and result formatting stay in
// generic code so that only these functions are built with ISA flags. Fills
// `nearest` (capacity k) in ascending distance order and returns the number of results.
using SearchLayersFn = size_t (*)(
    const HierarchicalNSW& index,
    const SplitSingleQuery<float>& query_wrapper,
    const float* q_to_centroids,
    VisitedSet& visited,
    buffer::SearchBuffer<float>& search_pool,
    size_t k,
    AnnCandidate<float>* nearest
);
size_t search_layers_generic(
    const HierarchicalNSW& index,
    const SplitSingleQuery<float>& query_wrapper,
    const float* q_to_centroids,
    VisitedSet& visited,
    buffer::SearchBuffer<float>& search_pool,
    size_t k,
    AnnCandidate<float>* nearest
);
size_t search_layers_neon(
    const HierarchicalNSW& index,
    const SplitSingleQuery<float>& query_wrapper,
    const float* q_to_centroids,
    VisitedSet& visited,
    buffer::SearchBuffer<float>& search_pool,
    size_t k,
    AnnCandidate<float>* nearest
);
size_t search_layers_avx2(
    const HierarchicalNSW& index,
    const SplitSingleQuery<float>& query_wrapper,
    const float* q_to_centroids,
    VisitedSet& visited,
    buffer::SearchBuffer<float>& search_pool,
    size_t k,
    AnnCandidate<float>* nearest
);
size_t search_layers_avx512_core(
    const HierarchicalNSW& index,
    const SplitSingleQuery<float>& query_wrapper,
    const float* q_to_centroids,
    VisitedSet& visited,
    buffer::SearchBuffer<float>& search_pool,
    size_t k,
    AnnCandidate<float>* nearest
);
size_t search_layers_avx512_popcnt(
    const HierarchicalNSW& index,
    const SplitSingleQuery<float>& query_wrapper,
    const float* q_to_centroids,
    VisitedSet& visited,
    buffer::SearchBuffer<float>& search_pool,
    size_t k,
    AnnCandidate<float>* nearest
);

}  // namespace detail
}  // namespace rabitqlib::hnsw
