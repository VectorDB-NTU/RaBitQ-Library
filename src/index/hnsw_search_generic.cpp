#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "hnsw_search_impl.hpp"  // IWYU pragma: keep (backend template definitions)
#include "rabitqlib/core/detail/simd/space_dispatch.hpp"
#include "rabitqlib/core/detail/simd/warmup_dispatch.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib::hnsw::detail {

struct HnswGenericKernel {
    static inline float warmup_ip_x0_q_512(
        const uint64_t* data,
        const uint64_t* query,
        float delta,
        float vl,
        size_t padded_dim,
        size_t b_query
    ) {
        return simd::warmup_ip_x0_q_512_generic(
            data, query, delta, vl, padded_dim, b_query
        );
    }

    static inline float mask_ip_x0_q(
        const float* query, const uint64_t* data, size_t padded_dim
    ) {
        return simd::mask_ip_x0_q_generic(query, data, padded_dim);
    }
};

std::vector<std::pair<float, PID>> search_knn_sorted_generic(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search
) {
    return index.search_knn_sorted_direct<HnswGenericKernel>(rotated_query, k, ef_search);
}

}  // namespace rabitqlib::hnsw::detail
