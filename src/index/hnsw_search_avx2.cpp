#include <cstddef>
#include <cstdint>

#include "hnsw_search_avx2_kernels.hpp"
#include "hnsw_search_impl.hpp"  // IWYU pragma: keep (backend template definitions)
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib::hnsw::detail {

namespace {
// Internal linkage keeps every template instantiated with this kernel local to
// this ISA's translation unit.
struct HnswAvx2Kernel {
    static inline float warmup_ip_x0_q_512(
        const uint64_t* data,
        const uint64_t* query,
        float delta,
        float vl,
        size_t padded_dim,
        size_t b_query
    ) {
        return hnsw_warmup_ip_x0_q_512_avx2(data, query, delta, vl, padded_dim, b_query);
    }

    static inline float mask_ip_x0_q(
        const float* query, const uint64_t* data, size_t padded_dim
    ) {
        return hnsw_mask_ip_x0_q_avx2(query, data, padded_dim);
    }
};
}  // namespace

size_t search_layers_avx2(
    const HierarchicalNSW& index,
    const SplitSingleQuery<float>& query_wrapper,
    const float* q_to_centroids,
    VisitedSet& visited,
    buffer::SearchBuffer<float>& search_pool,
    size_t k,
    AnnCandidate<float>* nearest
) {
    return HnswSearchAccess::search_layers<HnswAvx2Kernel>(
        index, query_wrapper, q_to_centroids, visited, search_pool, k, nearest
    );
}

}  // namespace rabitqlib::hnsw::detail
