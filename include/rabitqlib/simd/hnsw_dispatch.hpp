#pragma once

#include <cstddef>
#include <queue>
#include <utility>

#include "rabitqlib/defines.hpp"

namespace rabitqlib::hnsw {
class HierarchicalNSW;
namespace detail {
std::priority_queue<std::pair<float, PID>> search_knn(
    HierarchicalNSW&, const float*, size_t
);

std::priority_queue<std::pair<float, PID>> search_knn_neon(
    HierarchicalNSW&, const float*, size_t
);

std::priority_queue<std::pair<float, PID>> search_knn_generic(
    HierarchicalNSW&, const float*, size_t
);
std::priority_queue<std::pair<float, PID>> search_knn_avx2(
    HierarchicalNSW&, const float*, size_t
);

std::priority_queue<std::pair<float, PID>> search_knn_avx512_core(
    HierarchicalNSW&, const float*, size_t
);

std::priority_queue<std::pair<float, PID>> search_knn_avx512_popcnt(
    HierarchicalNSW&, const float*, size_t
);

}  // namespace detail
}  // namespace rabitqlib::hnsw
