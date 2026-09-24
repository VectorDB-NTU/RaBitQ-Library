#include <cstddef>
#include <cstdint>
#include <utility>

#include "../simd/hnsw_neon_kernels.hpp"
#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/simd/warmup_dispatch.hpp"

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

maxheap<std::pair<float, PID>> search_knn_neon(
    HierarchicalNSW& index, const float* rotated_query, size_t topk
) {
    return index.search_knn_direct<HnswNeonKernel>(rotated_query, topk);
}

}  // namespace rabitqlib::hnsw::detail
