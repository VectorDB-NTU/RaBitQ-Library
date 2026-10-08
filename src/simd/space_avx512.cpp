#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/utils/space.hpp"
#include "space_float_kernels.hpp"

namespace rabitqlib::simd {

namespace {

__m512 round_away_from_zero(__m512 value) {
    const __m512 truncated =
        _mm512_roundscale_ps(value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m512 sign = _mm512_set1_ps(-0.0F);
    const __m512 fraction = _mm512_andnot_ps(sign, _mm512_sub_ps(value, truncated));
    const __mmask16 round_mask =
        _mm512_cmp_ps_mask(fraction, _mm512_set1_ps(0.5F), _CMP_GE_OQ);
    const __m512 signed_one =
        _mm512_or_ps(_mm512_and_ps(value, sign), _mm512_set1_ps(1.0F));
    return _mm512_mask_add_ps(truncated, round_mask, truncated, signed_one);
}

}  // namespace

float euclidean_sqr_avx512(const float* a, const float* b, size_t dim) {
    return raw_float<FloatOperation::SquaredL2>(a, b, dim);
}

float dot_product_avx512(const float* a, const float* b, size_t dim) {
    return raw_float<FloatOperation::Dot>(a, b, dim);
}

float dot_product_dis_avx512(const float* a, const float* b, size_t dim) {
    return raw_float<FloatOperation::InnerProductDistance>(a, b, dim);
}

float l2norm_sqr_avx512(const float* a, size_t dim) {
    return raw_float<FloatOperation::SquaredNorm>(a, a, dim);
}

void scalar_quantize_uint8_avx512(
    uint8_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    size_t mul16 = dim - (dim & 0b1111);
    size_t i = 0;
    float one_over_delta = 1.0F / delta;
    __m512 lo512 = _mm512_set1_ps(lo);
    __m512 od512 = _mm512_set1_ps(one_over_delta);
    for (; i < mul16; i += 16) {
        __m512 cur = _mm512_loadu_ps(&vec0[i]);
        cur = _mm512_mul_ps(_mm512_sub_ps(cur, lo512), od512);
        __m128i i8 = _mm512_cvtusepi32_epi8(_mm512_cvttps_epi32(round_away_from_zero(cur)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&result[i]), i8);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint8_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
}

void scalar_quantize_uint16_avx512(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    size_t mul16 = dim - (dim & 0b1111);
    size_t i = 0;
    float one_over_delta = 1.0F / delta;
    __m512 lo512 = _mm512_set1_ps(lo);
    __m512 ow512 = _mm512_set1_ps(one_over_delta);
    for (; i < mul16; i += 16) {
        __m512 cur = _mm512_loadu_ps(&vec0[i]);
        cur = _mm512_mul_ps(_mm512_sub_ps(cur, lo512), ow512);
        __m256i i16 =
            _mm512_cvtusepi32_epi16(_mm512_cvttps_epi32(round_away_from_zero(cur)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i]), i16);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint16_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
}

void new_transpose_bin_avx512(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    // Reverse coordinates once; mask extraction then produces the stored bit order.
    const auto reverse_words = [](__m512i values) {
        const __m512i order = _mm512_broadcast_i32x4(
            _mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1)
        );
        values = _mm512_shuffle_epi8(values, order);
        return _mm512_shuffle_i64x2(values, values, 0x1B);
    };
    // 512 / 16 = 32
    for (size_t i = 0; i < padded_dim - padded_dim % 64; i += 64) {
        __m512i vec_00_to_31 = reverse_words(_mm512_loadu_si512(q));
        __m512i vec_32_to_63 = reverse_words(_mm512_loadu_si512(q + 32));

        // the first (16 - b_query) bits are empty
        vec_00_to_31 = _mm512_slli_epi32(vec_00_to_31, (16 - b_query));
        vec_32_to_63 = _mm512_slli_epi32(vec_32_to_63, (16 - b_query));

        for (size_t j = 0; j < b_query; ++j) {
            uint32_t v0 = _mm512_movepi16_mask(vec_00_to_31);  // get most significant bit
            uint32_t v1 = _mm512_movepi16_mask(vec_32_to_63);  // get most significant bit
            const uint64_t v = uint64_t{v1} | (uint64_t{v0} << 32);

            tq[b_query - j - 1] = v;

            vec_00_to_31 = _mm512_slli_epi16(vec_00_to_31, 1);
            vec_32_to_63 = _mm512_slli_epi16(vec_32_to_63, 1);
        }
        tq += b_query;
        q += 64;
    }
    if (padded_dim % 64 != 0) {
        __m512i values = reverse_words(_mm512_loadu_si512(q));
        values = _mm512_slli_epi16(values, static_cast<int>(16 - b_query));
        for (size_t bit = 0; bit < b_query; ++bit) {
            tq[b_query - bit - 1] = _mm512_movepi16_mask(values);
            values = _mm512_slli_epi16(values, 1);
        }
    }
}

void new_transpose_bin_512_avx512(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    const auto reverse_bytes = [](__m512i values) {
        const __m512i order = _mm512_broadcast_i32x4(
            _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
        );
        values = _mm512_shuffle_epi8(values, order);
        return _mm512_shuffle_i64x2(values, values, 0x1B);
    };
    // Keep full 512-dim blocks as 8 chunks, but store the tail as compact
    // [b_query x num_chunks] so runtime can use maskz loads without query padding.
    for (size_t i = 0; i < padded_dim;) {
        size_t block_size = 512;
        if (i + 512 > padded_dim) {
            block_size = padded_dim - i;
        }
        const size_t full_chunks = block_size / 64;
        const size_t tail_dim = block_size % 64;
        const size_t num_chunks = full_chunks + (tail_dim != 0);

        for (size_t k = 0; k < full_chunks; ++k) {
            const uint8_t* current_q = q + i + k * 64;
            const __m512i vec = reverse_bytes(_mm512_loadu_si512(current_q));

            for (size_t j = 0; j < b_query; ++j) {
                int bit_idx = static_cast<int>(b_query - 1 - j);
                // The signed byte preserves the intended one-bit mask, including bit 7.
                const char bit_mask = static_cast<char>(
                    1U << bit_idx
                );  // NOLINT(bugprone-narrowing-conversions)
                __mmask64 m = _mm512_test_epi8_mask(vec, _mm512_set1_epi8(bit_mask));
                tq[(b_query - j - 1) * num_chunks + k] = static_cast<uint64_t>(m);
            }
        }

        if (tail_dim != 0) {
            constexpr __mmask64 valid = 0xFFFFFFFFULL;
            const __m512i values =
                reverse_bytes(_mm512_maskz_loadu_epi8(valid, q + i + full_chunks * 64));
            for (size_t bit = 0; bit < b_query; ++bit) {
                const __m512i mask = _mm512_set1_epi8(static_cast<char>(1U << bit));
                const uint64_t selected = _mm512_test_epi8_mask(values, mask);
                tq[bit * num_chunks + full_chunks] = selected >> 32;
            }
        }

        i += block_size;
        tq += num_chunks * b_query;
    }
}

float mask_ip_x0_q_avx512(const float* query, const uint8_t* data, size_t padded_dim) {
    // Reuse the AVX2 implementation: independent accumulators avoid the long
    // reduction dependency of the wider masked-load loop.
    return mask_ip_x0_q_avx2(query, data, padded_dim);
}

float mask_ip_x0_q_avx512(const float* query, const uint64_t* data, size_t padded_dim) {
    return mask_ip_x0_q_avx512(query, reinterpret_cast<const uint8_t*>(data), padded_dim);
}

}  // namespace rabitqlib::simd
