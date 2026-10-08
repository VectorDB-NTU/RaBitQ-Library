#pragma once

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rabitqlib::simd::detail {
// Read exactly 2 * Bits bytes for 16 codes. Staging short stores through a
// 16-byte stack buffer blocks store forwarding on the following vector load.
template <size_t Bits>
inline __m128i load_excode_tail(const uint8_t* compact) {
    static_assert(Bits == 2 || Bits == 3 || Bits == 5 || Bits == 6 || Bits == 7);
    if constexpr (Bits == 2) {
        return _mm_loadu_si32(compact);
    } else if constexpr (Bits == 3) {
        uint16_t high;
        std::memcpy(&high, compact + 4, sizeof(high));
        return _mm_insert_epi16(_mm_loadu_si32(compact), high, 2);
    } else if constexpr (Bits == 5) {
        uint16_t high;
        std::memcpy(&high, compact + 8, sizeof(high));
        return _mm_insert_epi16(_mm_loadu_si64(compact), high, 4);
    } else if constexpr (Bits == 6) {
        return _mm_unpacklo_epi64(_mm_loadu_si64(compact), _mm_loadu_si32(compact + 8));
    } else {
        uint16_t high;
        std::memcpy(&high, compact + 12, sizeof(high));
        const __m128i upper = _mm_insert_epi16(_mm_loadu_si32(compact + 8), high, 2);
        return _mm_unpacklo_epi64(_mm_loadu_si64(compact), upper);
    }
}

template <size_t Bits>
inline void accumulate_excode_tail_avx2(
    const float* query, const uint8_t* compact, size_t dim, __m256& sum, __m256& sum_hi
) {
    assert(dim == 0 || dim == 32);
    if (dim == 0)
        return;
    // The fixed 32-coordinate tail contains two 16-code groups.
    // Select the two bytes spanning each code, then shift/mask eight codes at once.
    alignas(16) uint8_t positions[32];
    for (size_t i = 0; i < 16; ++i) {
        positions[2 * i] = static_cast<uint8_t>(i * Bits / 8);
        positions[2 * i + 1] = static_cast<uint8_t>(i * Bits / 8 + 1);
    }
    const __m128i low = _mm_load_si128(reinterpret_cast<const __m128i*>(positions));
    const __m128i high = _mm_load_si128(reinterpret_cast<const __m128i*>(positions + 16));
    constexpr size_t kShiftPeriod = Bits == 2 ? 32 : 8;
    const __m256i shifts = _mm256_setr_epi32(
        0,
        Bits % kShiftPeriod,
        (2 * Bits) % kShiftPeriod,
        (3 * Bits) % kShiftPeriod,
        (4 * Bits) % kShiftPeriod,
        (5 * Bits) % kShiftPeriod,
        (6 * Bits) % kShiftPeriod,
        (7 * Bits) % kShiftPeriod
    );
    const __m256i mask = _mm256_set1_epi32((1U << Bits) - 1);
    for (size_t i = 0; i < 32; i += 16) {
        const __m128i packed = load_excode_tail<Bits>(compact);
        __m256i words_lo, words_hi;
        if constexpr (Bits == 2) {
            // All 16 codes fit in one word; broadcast it instead of gathering bytes.
            words_lo = _mm256_broadcastd_epi32(packed);
            words_hi = _mm256_srli_epi32(words_lo, 16);
        } else {
            words_lo = _mm256_cvtepu16_epi32(_mm_shuffle_epi8(packed, low));
            words_hi = _mm256_cvtepu16_epi32(_mm_shuffle_epi8(packed, high));
        }
        const auto decode = [&](const __m256i words) {
            return _mm256_cvtepi32_ps(
                _mm256_and_si256(_mm256_srlv_epi32(words, shifts), mask)
            );
        };
        sum = _mm256_fmadd_ps(decode(words_lo), _mm256_loadu_ps(query + i), sum);
        sum_hi = _mm256_fmadd_ps(decode(words_hi), _mm256_loadu_ps(query + i + 8), sum_hi);
        compact += 2 * Bits;
    }
}

static inline float mask_ip_avx2(
    const float* query, const uint8_t* data, size_t padded_dim
) {
    const size_t num_blk = padded_dim / 64;
    const auto* it_data = data;
    const float* it_query = query;
    const __m256i shifts0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i shifts1 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    const __m256i shifts2 = _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23);
    const __m256i shifts3 = _mm256_setr_epi32(24, 25, 26, 27, 28, 29, 30, 31);
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    for (size_t i = 0; i < num_blk; ++i) {
        // Stored coordinates run from bit 63 to bit 0: high word first,
        // with each selected bit shifted into a maskload lane's sign bit.
        for (size_t half = 0; half < 2; ++half) {
            int32_t word;
            std::memcpy(&word, it_data + (1 - half) * sizeof(word), sizeof(word));
            const __m256i bits = _mm256_set1_epi32(word);
            sum0 = _mm256_add_ps(
                sum0, _mm256_maskload_ps(it_query, _mm256_sllv_epi32(bits, shifts0))
            );
            sum1 = _mm256_add_ps(
                sum1, _mm256_maskload_ps(it_query + 8, _mm256_sllv_epi32(bits, shifts1))
            );
            sum2 = _mm256_add_ps(
                sum2, _mm256_maskload_ps(it_query + 16, _mm256_sllv_epi32(bits, shifts2))
            );
            sum3 = _mm256_add_ps(
                sum3, _mm256_maskload_ps(it_query + 24, _mm256_sllv_epi32(bits, shifts3))
            );
            it_query += 32;
        }
        it_data += sizeof(uint64_t);
    }

    if (padded_dim % 64 != 0) {
        const auto accumulate = [](__m256 sum,
                                   const float* values,
                                   __m256i bits,
                                   __m256i shifts) {
            const __m256 mask =
                _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_sllv_epi32(bits, shifts), 31));
            return _mm256_add_ps(sum, _mm256_and_ps(_mm256_loadu_ps(values), mask));
        };
        int32_t signed_bits;
        std::memcpy(&signed_bits, it_data, sizeof(signed_bits));
        const __m256i bits = _mm256_set1_epi32(signed_bits);
        sum0 = accumulate(sum0, it_query, bits, shifts0);
        sum1 = accumulate(sum1, it_query + 8, bits, shifts1);
        sum2 = accumulate(sum2, it_query + 16, bits, shifts2);
        sum3 = accumulate(sum3, it_query + 24, bits, shifts3);
    }
    const __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    __m128 lanes = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    lanes = _mm_add_ps(lanes, _mm_movehl_ps(lanes, lanes));
    return _mm_cvtss_f32(_mm_add_ss(lanes, _mm_movehdup_ps(lanes)));
}
}  // namespace rabitqlib::simd::detail
