#pragma once

#include <cstddef>
#include <vector>

#include "rabitqlib/fastscan/query.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/detail/visited_set.hpp"

namespace rabitqlib::symqg::detail {

// Call-only view; storage and visited-set lifetimes are chosen by the caller.
struct SearchScratchView {
    float* rotated_query;
    float* estimated_distances;
    float* lookup_table;
    BatchQuery<float>& batch_query;
    buffer::SearchBuffer<float>& search_pool;
    buffer::SearchBuffer<float>& result_pool;
    VisitedSet& visited;
};

// Common owned buffers. Visited storage remains separate so single-query and
// clustering callers can retain their worker-local cache across calls.
struct SearchScratch {
    std::vector<float> rotated_query;
    std::vector<float> estimated_distances;
    std::vector<float> lookup_table;
    BatchQuery<float> batch_query;
    buffer::SearchBuffer<float> search_pool;
    buffer::SearchBuffer<float> result_pool;

    SearchScratch(size_t padded_dim, size_t degree, size_t ef_search, size_t k)
        : rotated_query(padded_dim)
        , estimated_distances(degree)
        , lookup_table(padded_dim * 4)
        , search_pool(ef_search)
        , result_pool(k) {}

    SearchScratchView view(VisitedSet& visited) {
        return {
            rotated_query.data(),
            estimated_distances.data(),
            lookup_table.data(),
            batch_query,
            search_pool,
            result_pool,
            visited};
    }
};

}  // namespace rabitqlib::symqg::detail
