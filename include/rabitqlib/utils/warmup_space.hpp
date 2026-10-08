#pragma once

#include <cstddef>
#include <cstdint>

#include "rabitqlib/utils/bitops.hpp"

namespace rabitqlib {

float warmup_ip_x0_q_512(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
);

float warmup_ip_x0_q_512(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
);

template <uint32_t b_query>
inline float warmup_ip_x0_q(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    [[maybe_unused]] size_t _b_query = 0  // not used
) {
    auto num_blk = padded_dim / 64;
    const auto* it_data = reinterpret_cast<const uint8_t*>(data);
    const auto* it_query = query;

    size_t ip = 0;
    size_t ppc = 0;

    for (size_t i = 0; i < num_blk; ++i) {
        const uint64_t x = rabitqlib::bitops::load_word(it_data, sizeof(uint64_t));
        ppc += bitops::popcount64(x);

        for (size_t j = 0; j < b_query; ++j) {
            uint64_t y = *static_cast<const uint64_t*>(it_query);
            ip += (bitops::popcount64(x & y) << j);
            it_query++;
        }
        it_data += sizeof(uint64_t);
    }

    if (padded_dim % 64 != 0) {
        const uint64_t x = rabitqlib::bitops::load_word(it_data, (padded_dim % 64) / 8);
        ppc += bitops::popcount64(x);
        for (size_t j = 0; j < b_query; ++j) {
            ip += (bitops::popcount64(x & *it_query) << j);
            ++it_query;
        }
    }

    return (delta * static_cast<float>(ip)) + (vl * static_cast<float>(ppc));
}

}  // namespace rabitqlib

template <uint32_t b_query, uint32_t padded_dim>
inline float warmup_ip_x0_q(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t _padded_dim = 0,  // not used
    size_t _b_query = 0      // not used
) {
    return rabitqlib::warmup_ip_x0_q<b_query>(data, query, delta, vl, padded_dim);
}
