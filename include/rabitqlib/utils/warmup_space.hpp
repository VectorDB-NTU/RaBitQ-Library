#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

template <uint32_t b_query>
inline float warmup_ip_x0_q(
    const uint64_t* data,   // pointer to data blocks (each 64 bits)
    const uint64_t* query,  // pointer to query words (each 64 bits), arranged so that for
                            // each data block the corresponding b_query query words follow
    float delta,
    float vl,
    size_t padded_dim,
    [[maybe_unused]] size_t _b_query = 0  // not used
) {
    const size_t num_blk = padded_dim / 64;
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;

    // Process blocks in chunks of 8
    const size_t vec_width = 8;
    size_t vec_end = (num_blk / vec_width) * vec_width;

    // Vector accumulators (each holds 8 64-bit lanes)
    __m512i ip_vec = _mm512_setzero_si512(
    );  // will accumulate weighted popcount intersections per block
    __m512i ppc_vec = _mm512_setzero_si512();  // will accumulate popcounts of data blocks

    // Loop over blocks in batches of 8
    for (size_t i = 0; i < vec_end; i += vec_width) {
        // Load eight 64-bit data blocks into x_vec.
        __m512i x_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + i));

        // Compute popcount for each 64-bit block in x_vec using the AVX512 VPOPCNTDQ
        // instruction. (Ensure you compile with the proper flags for VPOPCNTDQ.)
        __m512i popcnt_x_vec = _mm512_popcnt_epi64(x_vec);
        ppc_vec = _mm512_add_epi64(ppc_vec, popcnt_x_vec);

        // For accumulating the weighted popcounts per block.
        __m512i block_ip = _mm512_setzero_si512();

        // Process each query component (b_query is a compile-time constant, and is small).
        for (uint32_t j = 0; j < b_query; j++) {
            // We need to gather from query array the j-th query for each of the eight
            // blocks. For block (i + k) the index is: ( (i + k) * b_query + j ). We
            // construct an index vector of eight 64-bit indices.
            uint64_t indices[vec_width];
            for (size_t k = 0; k < vec_width; k++) {
                indices[k] = ((i + k) * b_query + j);
            }
            // Load indices from memory.
            __m512i index_vec = _mm512_loadu_si512(indices);
            // Gather 8 query words with a scale of 8 (since query is an array of 64-bit
            // integers).
            __m512i q_vec = _mm512_i64gather_epi64(index_vec, query, 8);

            // Compute bitwise AND of data blocks and corresponding query words.
            __m512i and_vec = _mm512_and_si512(x_vec, q_vec);
            // Compute popcount on each lane.
            __m512i popcnt_and = _mm512_popcnt_epi64(and_vec);

            // Multiply by the weighting factor (1 << j) for this query position.
            const uint64_t shift = 1ULL << j;
            __m512i shift_vec = _mm512_set1_epi64(shift);
            __m512i weighted = _mm512_mullo_epi64(popcnt_and, shift_vec);

            // Accumulate weighted popcounts for these blocks.
            block_ip = _mm512_add_epi64(block_ip, weighted);
        }
        // Add the block's query-weighted popcount to the overall ip vector.
        ip_vec = _mm512_add_epi64(ip_vec, block_ip);
    }

    // Horizontally reduce the vector accumulators.
    uint64_t ip_arr[vec_width];
    uint64_t ppc_arr[vec_width];
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(ip_arr), ip_vec);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(ppc_arr), ppc_vec);

    for (size_t k = 0; k < vec_width; k++) {
        ip_scalar += ip_arr[k];
        ppc_scalar += ppc_arr[k];
    }

    // Process remaining blocks that did not fit in the vectorized loop.
    for (size_t i = vec_end; i < num_blk; i++) {
        const uint64_t x = data[i];
        ppc_scalar += __builtin_popcountll(x);
        for (uint32_t j = 0; j < b_query; j++) {
            ip_scalar += __builtin_popcountll(x & query[i * b_query + j]) << j;
        }
    }

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));
}

template <uint32_t b_query, uint32_t padded_dim>
inline float warmup_ip_x0_q(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t _padded_dim = 0,  // not used
    size_t _b_query = 0      // not used
) {
    auto num_blk = padded_dim / 64;
    const auto* it_data = data;
    const auto* it_query = query;

    size_t ip = 0;
    size_t ppc = 0;

    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t x = *static_cast<const uint64_t*>(it_data);
        ppc += __builtin_popcountll(x);

        for (size_t j = 0; j < b_query; ++j) {
            uint64_t y = *static_cast<const uint64_t*>(it_query);
            ip += (__builtin_popcountll(x & y) << j);
            it_query++;
        }
        it_data++;
    }

    return (delta * static_cast<float>(ip)) + (vl * static_cast<float>(ppc));
}