#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "rabitqlib/utils/bitops.hpp"

namespace rabitqlib::simd::detail {

// Keep ISA-specific instantiations local to each backend translation unit.

// A partial 512-coordinate block contains at most eight words per query plane.
// Scalar POPCNT avoids staging/zeroing two full vectors for each short plane.
static inline void accumulate_warmup_tail(
    const uint8_t* data,
    const uint64_t* query,
    size_t dim,
    size_t bits,
    size_t& ip,
    size_t& count
) {
    const size_t full_words = dim / 64;
    const size_t words = (dim + 63) / 64;
    const auto accumulate = [&](uint64_t word, size_t index) {
        count += bitops::popcount64(word);
        for (size_t bit = 0; bit < bits; ++bit) {
            ip += size_t{bitops::popcount64(word & query[bit * words + index])} << bit;
        }
    };
    for (size_t i = 0; i < full_words; ++i) {
        uint64_t word;
        std::memcpy(&word, data + i * sizeof(word), sizeof(word));
        accumulate(word, i);
    }
    if (dim % 64 != 0) {
        accumulate(
            bitops::load_word(data + full_words * sizeof(uint64_t), sizeof(uint32_t)),
            full_words
        );
    }
}

static inline __m256i popcount_avx2(__m256i v) {
    // Lookup table for population count of 0-15
    const __m256i lookup = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4)
    );
    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    // Count low nibbles
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i cnt_lo = _mm256_shuffle_epi8(lookup, lo);

    // Count high nibbles
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    __m256i cnt_hi = _mm256_shuffle_epi8(lookup, hi);

    // Add counts (bytes)
    __m256i cnt_bytes = _mm256_add_epi8(cnt_lo, cnt_hi);

    // Sum bytes horizontally into 64-bit integers (SAD against 0)
    return _mm256_sad_epu8(cnt_bytes, _mm256_setzero_si256());
}

template <size_t MaxQueryBits>
static inline float warmup_blocks_avx2(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;

    __m256i acc_ip = _mm256_setzero_si256();
    __m256i acc_ppc = _mm256_setzero_si256();

    size_t i = 0;
    // Step by 512 bits at a time (64 bytes = 16 elements of 32-bit integers)
    size_t dim_end_512 = (padded_dim / 512) * 512;

    __m256i acc_bits[MaxQueryBits];
    for (size_t j = 0; j < b_query; ++j) {
        acc_bits[j] = _mm256_setzero_si256();
    }

    for (; i < dim_end_512; i += 512) {
        // Load 64 bytes of data using paired 32-byte loads
        __m256i data_vec_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
        __m256i data_vec_hi =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + 32));
        data += 64;

        acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_vec_lo));
        acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_vec_hi));

        for (size_t j = 0; j < b_query; ++j) {
            // Load 64 bytes of transposed query matching the 512-bit block layout
            __m256i query_vec_lo =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query));
            __m256i query_vec_hi =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query + 4));
            query += 8;  // Advance 8 x 64-bit ints (64 bytes)

            __m256i pop_lo = popcount_avx2(_mm256_and_si256(data_vec_lo, query_vec_lo));
            __m256i pop_hi = popcount_avx2(_mm256_and_si256(data_vec_hi, query_vec_hi));

            acc_bits[j] = _mm256_add_epi64(acc_bits[j], pop_lo);
            acc_bits[j] = _mm256_add_epi64(acc_bits[j], pop_hi);
        }
    }

    if (i < padded_dim) {
        accumulate_warmup_tail(data, query, padded_dim - i, b_query, ip_scalar, ppc_scalar);
    }

    for (size_t j = 0; j < b_query; ++j) {
        __m128i shift = _mm_cvtsi32_si128(static_cast<int>(j));
        acc_ip = _mm256_add_epi64(acc_ip, _mm256_sll_epi64(acc_bits[j], shift));
    }

    // Standard reduction for a single __m256i
    auto mm256_reduce_add_epi64 = [](__m256i v) {
        __m128i low = _mm256_castsi256_si128(v);
        __m128i high = _mm256_extracti128_si256(v, 1);
        __m128i sum = _mm_add_epi64(low, high);
        return _mm_extract_epi64(sum, 0) + _mm_extract_epi64(sum, 1);
    };

    ip_scalar += mm256_reduce_add_epi64(acc_ip);
    ppc_scalar += mm256_reduce_add_epi64(acc_ppc);

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));
}

#if defined(__AVX512F__) || defined(_MSC_VER)
template <size_t MaxQueryBits>
static inline float warmup_blocks_avx512(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    __m512i acc_ip = _mm512_setzero_si512();
    __m512i acc_ppc = _mm512_setzero_si512();

    size_t i = 0;
    size_t dim_end_512 = (padded_dim / 512) * 512;

    __m512i acc_bits[MaxQueryBits];
    for (size_t j = 0; j < b_query; ++j) {
        acc_bits[j] = _mm512_setzero_si512();
    }

    for (; i < dim_end_512; i += 512) {
        __m512i data_vec = _mm512_loadu_si512(data);
        data += 64;

        acc_ppc = _mm512_add_epi64(acc_ppc, _mm512_popcnt_epi64(data_vec));

        for (size_t j = 0; j < b_query; ++j) {
            __m512i query_vec = _mm512_loadu_si512(query);
            query += 8;

            __m512i pop = _mm512_popcnt_epi64(_mm512_and_si512(data_vec, query_vec));
            acc_bits[j] = _mm512_add_epi64(acc_bits[j], pop);
        }
    }

    size_t remaining_dim = padded_dim - i;
    if (remaining_dim > 0) {
        const size_t num_chunks = (remaining_dim + 63) / 64;
        const auto valid_mask = static_cast<__mmask8>((1u << num_chunks) - 1u);
        // Byte masking zero-extends the partial last word without staging short
        // stores through a stack buffer. The query planes already use whole words.
        const __m512i data_vec =
            _mm512_maskz_loadu_epi8((uint64_t{1} << (remaining_dim / 8)) - 1, data);
        acc_ppc = _mm512_add_epi64(acc_ppc, _mm512_popcnt_epi64(data_vec));

        for (size_t j = 0; j < b_query; ++j) {
            __m512i query_vec = _mm512_maskz_loadu_epi64(valid_mask, query);
            query += num_chunks;

            __m512i pop = _mm512_popcnt_epi64(_mm512_and_si512(data_vec, query_vec));
            acc_bits[j] = _mm512_add_epi64(acc_bits[j], pop);
        }
    }

    for (size_t j = 0; j < b_query; ++j) {
        __m128i shift = _mm_cvtsi32_si128(static_cast<int>(j));
        acc_ip = _mm512_add_epi64(acc_ip, _mm512_sll_epi64(acc_bits[j], shift));
    }

    const size_t ip_scalar = static_cast<size_t>(_mm512_reduce_add_epi64(acc_ip));
    const size_t ppc_scalar = static_cast<size_t>(_mm512_reduce_add_epi64(acc_ppc));

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));
}
#endif

}  // namespace rabitqlib::simd::detail
