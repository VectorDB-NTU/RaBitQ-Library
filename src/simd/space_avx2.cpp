#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "packed_tail_avx2.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/utils/space.hpp"
#include "space_float_kernels.hpp"

namespace rabitqlib::simd {

namespace {

__m256 round_away_from_zero(__m256 value) {
    const __m256 truncated = _mm256_round_ps(value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m256 sign = _mm256_set1_ps(-0.0F);
    const __m256 fraction = _mm256_andnot_ps(sign, _mm256_sub_ps(value, truncated));
    const __m256 round_mask = _mm256_cmp_ps(fraction, _mm256_set1_ps(0.5F), _CMP_GE_OQ);
    const __m256 signed_one =
        _mm256_or_ps(_mm256_and_ps(value, sign), _mm256_set1_ps(1.0F));
    return _mm256_add_ps(truncated, _mm256_and_ps(round_mask, signed_one));
}

}  // namespace

float euclidean_sqr_avx2(const float* a, const float* b, size_t dim) {
    return raw_float<FloatOperation::SquaredL2>(a, b, dim);
}

float dot_product_avx2(const float* a, const float* b, size_t dim) {
    return raw_float<FloatOperation::Dot>(a, b, dim);
}

float dot_product_dis_avx2(const float* a, const float* b, size_t dim) {
    return raw_float<FloatOperation::InnerProductDistance>(a, b, dim);
}

float l2norm_sqr_avx2(const float* a, size_t dim) {
    return raw_float<FloatOperation::SquaredNorm>(a, a, dim);
}

void scalar_quantize_uint8_avx2(
    uint8_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    size_t mul8 = dim - (dim & 0b111);
    size_t i = 0;
    float one_over_delta = 1.0F / delta;
    __m256 lo256 = _mm256_set1_ps(lo);
    __m256 od256 = _mm256_set1_ps(one_over_delta);
    __m128i zero = _mm_setzero_si128();

    for (; i < mul8; i += 8) {
        __m256 cur = _mm256_loadu_ps(&vec0[i]);
        cur = _mm256_mul_ps(_mm256_sub_ps(cur, lo256), od256);
        __m256i i32 = _mm256_cvttps_epi32(round_away_from_zero(cur));
        __m128i lo32 = _mm256_castsi256_si128(i32);
        __m128i hi32 = _mm256_extracti128_si256(i32, 1);
        __m128i i16 = _mm_packus_epi32(lo32, hi32);
        __m128i i8 = _mm_packus_epi16(i16, zero);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(&result[i]), i8);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint8_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
}

void scalar_quantize_uint16_avx2(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    size_t mul8 = dim - (dim & 0b111);
    size_t i = 0;
    float one_over_delta = 1.0F / delta;
    __m256 lo256 = _mm256_set1_ps(lo);
    __m256 ow256 = _mm256_set1_ps(one_over_delta);
    for (; i < mul8; i += 8) {
        __m256 cur = _mm256_loadu_ps(&vec0[i]);
        cur = _mm256_mul_ps(_mm256_sub_ps(cur, lo256), ow256);
        __m256i i32 = _mm256_cvttps_epi32(round_away_from_zero(cur));
        __m128i lo32 = _mm256_castsi256_si128(i32);
        __m128i hi32 = _mm256_extracti128_si256(i32, 1);
        __m128i i16 = _mm_packus_epi32(lo32, hi32);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result + i), i16);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint16_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
}

void new_transpose_bin_avx2(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    // Reverse coordinates once; movemask then produces the stored bit order.
    const __m256i order = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1)
    );
    for (size_t i = 0; i < padded_dim - padded_dim % 64; i += 64) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 16));
        const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32));
        const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 48));
        // Arrange eight-coordinate groups so packs produces the final bit order.
        __m256i hi_odd = _mm256_shuffle_epi8(_mm256_permute2x128_si256(b, a, 0x20), order);
        __m256i hi_even = _mm256_shuffle_epi8(_mm256_permute2x128_si256(b, a, 0x31), order);
        __m256i lo_odd = _mm256_shuffle_epi8(_mm256_permute2x128_si256(d, c, 0x20), order);
        __m256i lo_even = _mm256_shuffle_epi8(_mm256_permute2x128_si256(d, c, 0x31), order);

        // the first (16 - b_query) bits are empty
        const int shift = static_cast<int>(16 - b_query);
        hi_odd = _mm256_slli_epi32(hi_odd, shift);
        hi_even = _mm256_slli_epi32(hi_even, shift);
        lo_odd = _mm256_slli_epi32(lo_odd, shift);
        lo_even = _mm256_slli_epi32(lo_even, shift);

        for (size_t j = 0; j < b_query; ++j) {
            // pack two 16-bit vectors to 8-bit interleaved vectors
            __m256i p0 = _mm256_packs_epi16(lo_even, lo_odd);
            __m256i p1 = _mm256_packs_epi16(hi_even, hi_odd);

            const uint32_t m0 = _mm256_movemask_epi8(p0);
            const uint32_t m1 = _mm256_movemask_epi8(p1);

            const uint64_t v = uint64_t{m0} | (uint64_t{m1} << 32);

            tq[b_query - j - 1] = v;

            hi_odd = _mm256_slli_epi16(hi_odd, 1);
            hi_even = _mm256_slli_epi16(hi_even, 1);
            lo_odd = _mm256_slli_epi16(lo_odd, 1);
            lo_even = _mm256_slli_epi16(lo_even, 1);
        }
        tq += b_query;
        q += 64;
    }
    const size_t tail_dim = padded_dim % 64;
    // The partial word contains exactly 32 coordinates.
    if (tail_dim != 0) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 16));
        __m256i odd = _mm256_shuffle_epi8(_mm256_permute2x128_si256(b, a, 0x20), order);
        __m256i even = _mm256_shuffle_epi8(_mm256_permute2x128_si256(b, a, 0x31), order);
        odd = _mm256_slli_epi16(odd, static_cast<int>(16 - b_query));
        even = _mm256_slli_epi16(even, static_cast<int>(16 - b_query));
        for (size_t bit = 0; bit < b_query; ++bit) {
            const __m256i bytes = _mm256_packs_epi16(even, odd);
            tq[b_query - bit - 1] = static_cast<uint32_t>(_mm256_movemask_epi8(bytes));
            odd = _mm256_slli_epi16(odd, 1);
            even = _mm256_slli_epi16(even, 1);
        }
    }
}

void new_transpose_bin_512_avx2(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    const __m128i reverse_bytes =
        _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const auto load_reversed = [&](const uint8_t* values) {
        const __m256i loaded = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
        return _mm256_permute4x64_epi64(
            _mm256_shuffle_epi8(loaded, _mm256_broadcastsi128_si256(reverse_bytes)), 0x4E
        );
    };
    // Shift each plane to byte sign bits. Across at most eight planes, carries
    // from the neighboring byte never reach those sign bits.
    for (size_t i = 0; i < padded_dim;) {
        size_t block_size = 512;
        if (i + 512 > padded_dim) {
            block_size = padded_dim - i;
        }
        // Each chunk represents 64 bytes (512 bits) of dimensions
        const size_t full_chunks = block_size / 64;
        const size_t tail_dim = block_size % 64;
        const size_t num_chunks = full_chunks + (tail_dim != 0);

        for (size_t k = 0; k < full_chunks; ++k) {
            // Load 64 bytes using two sequential 32-byte AVX2 registers
            const uint8_t* current_q_lo = q + i + k * 64;
            const uint8_t* current_q_hi = q + i + k * 64 + 32;

            __m256i vec_lo = _mm256_slli_epi16(
                load_reversed(current_q_lo), static_cast<int>(8 - b_query)
            );
            __m256i vec_hi = _mm256_slli_epi16(
                load_reversed(current_q_hi), static_cast<int>(8 - b_query)
            );

            for (size_t j = 0; j < b_query; ++j) {
                const auto m_lo = static_cast<uint32_t>(_mm256_movemask_epi8(vec_lo));
                const auto m_hi = static_cast<uint32_t>(_mm256_movemask_epi8(vec_hi));

                // Combine both 32-bit masks into a single 64-bit mask
                uint64_t m = (static_cast<uint64_t>(m_lo) << 32) | m_hi;

                // Write into the 64-bit structured macro-layout
                tq[(b_query - j - 1) * num_chunks + k] = m;
                vec_lo = _mm256_slli_epi16(vec_lo, 1);
                vec_hi = _mm256_slli_epi16(vec_hi, 1);
            }
        }

        if (tail_dim != 0) {
            __m256i values = _mm256_slli_epi16(
                load_reversed(q + i + full_chunks * 64), static_cast<int>(8 - b_query)
            );
            for (size_t bit = 0; bit < b_query; ++bit) {
                tq[(b_query - bit - 1) * num_chunks + full_chunks] =
                    static_cast<uint32_t>(_mm256_movemask_epi8(values));
                values = _mm256_slli_epi16(values, 1);
            }
        }

        i += block_size;
        tq += num_chunks * b_query;
    }
}

float mask_ip_x0_q_avx2(const float* query, const uint8_t* data, size_t padded_dim) {
    return detail::mask_ip_avx2(query, data, padded_dim);
}

float mask_ip_x0_q_avx2(const float* query, const uint64_t* data, size_t padded_dim) {
    return mask_ip_x0_q_avx2(query, reinterpret_cast<const uint8_t*>(data), padded_dim);
}

}  // namespace rabitqlib::simd
