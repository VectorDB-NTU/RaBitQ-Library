#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "../simd/hnsw_neon_kernels.hpp"
#include "hnsw_search_impl.hpp"  // IWYU pragma: keep (backend template definitions)
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib::hnsw::detail {

struct HnswNeonKernel {
    static inline float warmup_ip_x0_q_512(
        const uint64_t* data,
        const uint64_t* query,
        float delta,
        float vl,
        size_t padded_dim,
        size_t b_query
    ) {
        return simd::detail::warmup_ip_neon(
            reinterpret_cast<const uint8_t*>(data), query, delta, vl, padded_dim, b_query
        );
    }

    static inline float mask_ip_x0_q(
        const float* query, const uint64_t* data, size_t padded_dim
    ) {
        return simd::detail::mask_ip_neon(
            query, reinterpret_cast<const uint8_t*>(data), padded_dim
        );
    }
};

std::vector<std::pair<float, PID>> search_knn_sorted_neon(
    HierarchicalNSW& index, const float* rotated_query, size_t topk, size_t ef
) {
    return index.search_knn_sorted_direct<HnswNeonKernel>(
        rotated_query, topk, ef == 0 ? index.ef_ : ef
    );
}

}  // namespace rabitqlib::hnsw::detail
