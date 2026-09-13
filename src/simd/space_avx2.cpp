#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/utils/space.hpp"
#include "space_float_kernels.hpp"

namespace rabitqlib::simd {

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
        __m256i i32 = _mm256_cvtps_epi32(cur);
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
        __m256i i32 = _mm256_cvtps_epi32(cur);
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
    for (size_t i = 0; i < padded_dim; i += 64) {
        __m256i vec_00_to_15 = _mm256_loadu_si256((__m256i const*)(q));
        __m256i vec_16_to_31 = _mm256_loadu_si256((__m256i const*)(q + 16));
        __m256i vec_32_to_47 = _mm256_loadu_si256((__m256i const*)(q + 32));
        __m256i vec_48_to_63 = _mm256_loadu_si256((__m256i const*)(q + 48));

        // the first (16 - b_query) bits are empty
        const int shift = static_cast<int>(16 - b_query);
        vec_00_to_15 = _mm256_slli_epi32(vec_00_to_15, shift);
        vec_16_to_31 = _mm256_slli_epi32(vec_16_to_31, shift);
        vec_32_to_47 = _mm256_slli_epi32(vec_32_to_47, shift);
        vec_48_to_63 = _mm256_slli_epi32(vec_48_to_63, shift);

        for (size_t j = 0; j < b_query; ++j) {
            // pack two 16-bit vectors to 8-bit interleaved vectors
            __m256i p0 = _mm256_packs_epi16(vec_00_to_15, vec_16_to_31);
            __m256i p1 = _mm256_packs_epi16(vec_32_to_47, vec_48_to_63);

            uint32_t m0 = _mm256_movemask_epi8(p0);
            uint32_t m1 = _mm256_movemask_epi8(p1);

            // Fix AVX2 Lane Ordering of the interleaved mask
            auto fix_avx2_mask = [](uint32_t m) {
                return (m & 0xFF0000FF) | ((m & 0x00FF0000) >> 8) | ((m & 0x0000FF00) << 8);
            };

            m0 = fix_avx2_mask(m0);
            m1 = fix_avx2_mask(m1);

            m0 = reverse_bits(m0);
            m1 = reverse_bits(m1);

            uint64_t v = (static_cast<uint64_t>(m0) << 32) | m1;

            tq[b_query - j - 1] = v;

            vec_00_to_15 = _mm256_slli_epi16(vec_00_to_15, 1);
            vec_16_to_31 = _mm256_slli_epi16(vec_16_to_31, 1);
            vec_32_to_47 = _mm256_slli_epi16(vec_32_to_47, 1);
            vec_48_to_63 = _mm256_slli_epi16(vec_48_to_63, 1);
        }
        tq += b_query;
        q += 64;
    }
}

void new_transpose_bin_512_avx2(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    for (size_t i = 0; i < padded_dim;) {
        size_t block_size = 512;
        if (i + 512 > padded_dim) {
            block_size = padded_dim - i;
        }
        // Each chunk represents 64 bytes (512 bits) of dimensions
        size_t num_chunks = block_size / 64;

        for (size_t k = 0; k < num_chunks; ++k) {
            // Load 64 bytes using two sequential 32-byte AVX2 registers
            const uint8_t* current_q_lo = q + i + k * 64;
            const uint8_t* current_q_hi = q + i + k * 64 + 32;

            __m256i vec_lo =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_q_lo));
            __m256i vec_hi =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_q_hi));

            for (size_t j = 0; j < b_query; ++j) {
                int bit_idx = static_cast<int>(b_query - 1 - j);
                __m256i mask_vec = _mm256_set1_epi8(static_cast<char>(1 << bit_idx));

                // Process lower 32 bytes
                __m256i res_lo = _mm256_and_si256(vec_lo, mask_vec);
                __m256i eq_lo = _mm256_cmpeq_epi8(res_lo, _mm256_setzero_si256());
                uint32_t m_lo = ~static_cast<uint32_t>(_mm256_movemask_epi8(eq_lo));

                // Process upper 32 bytes
                __m256i res_hi = _mm256_and_si256(vec_hi, mask_vec);
                __m256i eq_hi = _mm256_cmpeq_epi8(res_hi, _mm256_setzero_si256());
                uint32_t m_hi = ~static_cast<uint32_t>(_mm256_movemask_epi8(eq_hi));

                // Combine both 32-bit masks into a single 64-bit mask
                uint64_t m = (static_cast<uint64_t>(m_hi) << 32) | m_lo;

                // Write into the 64-bit structured macro-layout
                tq[(b_query - j - 1) * num_chunks + k] = reverse_bits_u64(m);
            }
        }

        i += block_size;
        tq += num_chunks * b_query;
    }
}

float mask_ip_x0_q_avx2(const float* query, const uint64_t* data, size_t padded_dim) {
    const size_t num_blk = padded_dim / 64;
    const auto* it_data = reinterpret_cast<const uint8_t*>(data);
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

    const __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    __m128 lanes = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    lanes = _mm_add_ps(lanes, _mm_movehl_ps(lanes, lanes));
    return _mm_cvtss_f32(_mm_add_ss(lanes, _mm_movehdup_ps(lanes)));
}

}  // namespace rabitqlib::simd
