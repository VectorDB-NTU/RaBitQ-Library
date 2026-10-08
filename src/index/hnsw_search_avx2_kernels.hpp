#pragma once

#include <cstddef>
#include <cstdint>

#include "../simd/packed_tail_avx2.hpp"
#include "../simd/warmup_kernels.hpp"
#include "rabitqlib/index/query.hpp"

namespace rabitqlib::hnsw::detail {

static inline float hnsw_mask_ip_x0_q_avx2(
    const float* query, const uint64_t* data, size_t padded_dim
) {
    return simd::detail::mask_ip_avx2(
        query, reinterpret_cast<const uint8_t*>(data), padded_dim
    );
}

static inline float hnsw_warmup_ip_x0_q_512_avx2(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;
    if (padded_dim < 512) {
        simd::detail::accumulate_warmup_tail(
            reinterpret_cast<const uint8_t*>(data),
            query,
            padded_dim,
            b_query,
            ip_scalar,
            ppc_scalar
        );
        return (delta * static_cast<float>(ip_scalar)) +
               (vl * static_cast<float>(ppc_scalar));
    }

    return simd::detail::warmup_blocks_avx2<SplitSingleQuery<float>::kNumBits>(
        reinterpret_cast<const uint8_t*>(data), query, delta, vl, padded_dim, b_query
    );
}

}  // namespace rabitqlib::hnsw::detail
