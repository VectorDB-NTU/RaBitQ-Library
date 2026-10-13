#include <cstdint>

#include "rabitqlib/index/detail/visited_set.hpp"
#include "rabitqlib/index/symqg/detail/search_scratch.hpp"
#include "rabitqlib/index/symqg/qg.hpp"

void symqg_scratch_header_smoke(
    rabitqlib::symqg::QuantizedGraph<float>& graph,
    const float* query,
    uint32_t* ids,
    float* distances
) {
    rabitqlib::VisitedSet visited(graph.num_vertices(), graph.num_vertices());
    rabitqlib::symqg::detail::SearchScratch scratch(
        graph.padded_dim(), graph.degree_bound(), 10, 1
    );
    graph.search_with_scratch(
        query,
        1,
        ids,
        distances,
        scratch.rotated_query.data(),
        scratch.estimated_distances.data(),
        scratch.lookup_table.data(),
        scratch.batch_query,
        scratch.search_pool,
        scratch.result_pool,
        visited
    );
    rabitqlib::symqg::detail::search_with_workspace(
        graph, query, 1, ids, distances, scratch.view(visited)
    );
}
