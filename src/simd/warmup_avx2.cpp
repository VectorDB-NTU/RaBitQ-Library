#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "rabitqlib/core/detail/simd/warmup_dispatch.hpp"
#include "warmup_kernels.hpp"

namespace rabitqlib::simd {

float warmup_ip_x0_q_512_avx2(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;

    constexpr size_t kMaxQueryBits = 8;
    if (b_query > kMaxQueryBits) {
        throw std::invalid_argument("warmup_ip_x0_q_512 requires at most 8 query bits");
    }
    if (padded_dim < 512) {
        detail::accumulate_warmup_tail(
            data, query, padded_dim, b_query, ip_scalar, ppc_scalar
        );
        return (delta * static_cast<float>(ip_scalar)) +
               (vl * static_cast<float>(ppc_scalar));
    }

    return detail::warmup_blocks_avx2<kMaxQueryBits>(
        data, query, delta, vl, padded_dim, b_query
    );
}

float warmup_ip_x0_q_512_avx2(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    return warmup_ip_x0_q_512_avx2(
        reinterpret_cast<const uint8_t*>(data), query, delta, vl, padded_dim, b_query
    );
}

}  // namespace rabitqlib::simd
