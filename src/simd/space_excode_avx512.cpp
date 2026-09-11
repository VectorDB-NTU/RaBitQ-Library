#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "rabitqlib/simd/space_dispatch.hpp"

namespace rabitqlib::simd::excode_ipimpl {

namespace {
[[nodiscard]] inline uint64_t load_u64(const uint8_t* data) noexcept {
    uint64_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

[[nodiscard]] inline __m128i set_u64x(uint64_t high, uint64_t low) noexcept {
    int64_t signed_high = 0;
    int64_t signed_low = 0;
    std::memcpy(&signed_high, &high, sizeof(signed_high));
    std::memcpy(&signed_low, &low, sizeof(signed_low));
    return _mm_set_epi64x(signed_high, signed_low);
}
}  // namespace

// ip16: this function is used to compute inner product of
// vectors padded to multiple of 16
// fxu1: the inner product is computed between float and 1-bit unsigned int (lay out can be
// found rabitq_impl.hpp)
// avx512: only applicable for avx512
float ip16_fxu1_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum0 = _mm512_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
    const auto load_query = [&](size_t k) {
        __mmask16 mask = 0;
        std::memcpy(&mask, compact_code + k / 8, sizeof(mask));
        return _mm512_maskz_loadu_ps(mask, query + k);
    };
    size_t i = 0;
    for (; i + 64 <= dim; i += 64) {
        sum0 = _mm512_add_ps(sum0, load_query(i));
        sum1 = _mm512_add_ps(sum1, load_query(i + 16));
        sum2 = _mm512_add_ps(sum2, load_query(i + 32));
        sum3 = _mm512_add_ps(sum3, load_query(i + 48));
    }
    for (; i < dim; i += 16) {
        sum0 = _mm512_add_ps(sum0, load_query(i));
    }
    return _mm512_reduce_add_ps(
        _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3))
    );
}

float ip64_fxu2_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum0 = _mm512_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;

    float result = 0;
    const __m128i mask = _mm_set1_epi8(0b00000011);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));

        __m128i vec_00_to_15 = _mm_and_si128(compact, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact, 6), mask);
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum0 = _mm512_fmadd_ps(q, cf, sum0);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum1 = _mm512_fmadd_ps(q, cf, sum1);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum2 = _mm512_fmadd_ps(q, cf, sum2);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum3 = _mm512_fmadd_ps(q, cf, sum3);

        compact_code += 16;
    }

    result = _mm512_reduce_add_ps(
        _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3))
    );

    return result;
}

float ip64_fxu3_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum = _mm512_setzero_ps();

    float result = 0;
    const __m128i mask = _mm_set1_epi8(0b11);
    const __m128i top_mask = _mm_set1_epi8(0b100);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        compact_code += 16;

        const uint64_t top_bit = load_u64(compact_code);
        compact_code += 8;

        __m128i vec_00_to_15 = _mm_and_si128(compact2, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact2, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact2, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact2, 6), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(set_u64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(set_u64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(set_u64x(top_bit >> 3, top_bit >> 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(set_u64x(top_bit >> 5, top_bit >> 4), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
    }

    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip16_fxu4_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum0 = _mm512_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
    // Each eight-byte block stores dimensions 0-7 in low nibbles and 8-15 in high nibbles.
    const auto unpack_code = [&](size_t k) {
        __m128i bytes =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(compact_code + k / 2));
        __m128i lo = _mm_and_si128(bytes, _mm_set1_epi8(15));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(15));
        __m128i expanded = _mm_unpacklo_epi64(lo, hi);
        return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(expanded));
    };
    size_t i = 0;
    for (; i + 64 <= dim; i += 64) {
        sum0 = _mm512_fmadd_ps(unpack_code(i), _mm512_loadu_ps(query + i), sum0);
        sum1 = _mm512_fmadd_ps(unpack_code(i + 16), _mm512_loadu_ps(query + i + 16), sum1);
        sum2 = _mm512_fmadd_ps(unpack_code(i + 32), _mm512_loadu_ps(query + i + 32), sum2);
        sum3 = _mm512_fmadd_ps(unpack_code(i + 48), _mm512_loadu_ps(query + i + 48), sum3);
    }
    for (; i < dim; i += 16) {
        sum0 = _mm512_fmadd_ps(unpack_code(i), _mm512_loadu_ps(query + i), sum0);
    }
    return _mm512_reduce_add_ps(
        _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3))
    );
}

float ip64_fxu5_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum = _mm512_setzero_ps();

    float result = 0.0F;
    const __m128i mask = _mm_set1_epi8(0b1111);
    const __m128i top_mask = _mm_set1_epi8(0b10000);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact4_1 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        __m128i compact4_2 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 16));
        compact_code += 32;

        const uint64_t top_bit = load_u64(compact_code);
        compact_code += 8;

        __m128i vec_00_to_15 = _mm_and_si128(compact4_1, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact4_1, 4), mask);
        __m128i vec_32_to_47 = _mm_and_si128(compact4_2, mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact4_2, 4), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(set_u64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(set_u64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(set_u64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(set_u64x(top_bit >> 3, top_bit >> 2), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip64_fxu6_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum0 = _mm512_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;

    float result = 0.0F;
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    const __m128i mask2 = _mm_set1_epi8(static_cast<char>(0b11000000));

    for (size_t i = 0; i < dim; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 32));

        compact_code += 48;

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_and_si128(cpt3, mask6);
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4)
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2)
        );

        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum0 = _mm512_fmadd_ps(q, cf, sum0);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum1 = _mm512_fmadd_ps(q, cf, sum1);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum2 = _mm512_fmadd_ps(q, cf, sum2);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum3 = _mm512_fmadd_ps(q, cf, sum3);
    }
    result = _mm512_reduce_add_ps(
        _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3))
    );

    return result;
}

float ip64_fxu7_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum = _mm512_setzero_ps();

    float result = 0.0F;
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    const __m128i mask2 = _mm_set1_epi8(static_cast<char>(0b11000000));
    const __m128i top_mask = _mm_set1_epi8(0b1000000);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 32));
        compact_code += 48;

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_and_si128(cpt3, mask6);
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4)
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2)
        );

        const uint64_t top_bit = load_u64(compact_code);
        compact_code += 8;

        __m128i top_00_to_15 =
            _mm_and_si128(set_u64x(top_bit << 5, top_bit << 6), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(set_u64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(set_u64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(set_u64x(top_bit >> 1, top_bit << 0), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
    }

    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip16_fxu8_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ code, size_t dim
) {
    __m512 sum0 = _mm512_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
    const auto unpack_code = [&](size_t k) {
        return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(code + k))
        ));
    };
    size_t i = 0;
    for (; i + 64 <= dim; i += 64) {
        sum0 = _mm512_fmadd_ps(unpack_code(i), _mm512_loadu_ps(query + i), sum0);
        sum1 = _mm512_fmadd_ps(unpack_code(i + 16), _mm512_loadu_ps(query + i + 16), sum1);
        sum2 = _mm512_fmadd_ps(unpack_code(i + 32), _mm512_loadu_ps(query + i + 32), sum2);
        sum3 = _mm512_fmadd_ps(unpack_code(i + 48), _mm512_loadu_ps(query + i + 48), sum3);
    }
    for (; i < dim; i += 16) {
        sum0 = _mm512_fmadd_ps(unpack_code(i), _mm512_loadu_ps(query + i), sum0);
    }
    return _mm512_reduce_add_ps(
        _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3))
    );
}

}  // namespace rabitqlib::simd::excode_ipimpl
