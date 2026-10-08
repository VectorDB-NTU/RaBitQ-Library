#pragma once

#include <cstddef>
#include <cstdint>

#include "../simd/warmup_kernels.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"

namespace rabitqlib::hnsw::detail {

static inline float hnsw_mask_ip_x0_q_avx512(
    const float* query, const uint64_t* data, size_t padded_dim
) {
    return simd::mask_ip_x0_q_avx2(
        query, reinterpret_cast<const uint8_t*>(data), padded_dim
    );
}

static inline float hnsw_warmup_ip_x0_q_512_avx512(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;

    // The fixed-width HNSW path favors vector popcount beyond two words.
    if (padded_dim <= 128) {
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

    return simd::detail::warmup_blocks_avx512<SplitSingleQuery<float>::kNumBits>(
        reinterpret_cast<const uint8_t*>(data), query, delta, vl, padded_dim, b_query
    );
}

}  // namespace rabitqlib::hnsw::detail
