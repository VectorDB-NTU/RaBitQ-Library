#include <cstddef>
#include <cstdint>

#include "pack_excode_kernels.hpp"
#include "rabitqlib/simd/pack_excode_dispatch.hpp"

namespace rabitqlib::simd {

void packing_2bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_2bit_excode_intrinsics(o_raw, o_compact, dim);
}

void packing_3bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_3bit_excode_intrinsics(o_raw, o_compact, dim);
}

void packing_4bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_4bit_excode_intrinsics<64>(o_raw, o_compact, dim);
}

void packing_5bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_5bit_excode_intrinsics(o_raw, o_compact, dim);
}

void packing_6bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    const __m256i mask = _mm256_set1_epi8(0x3F);
    const __m256i shifts = _mm256_setr_epi32(6, 6, 6, 6, 4, 4, 4, 4);
    for (size_t i = 0; i < dim - dim % 64; i += 64) {
        // Preserve the full-block layout: the last 16 codes contribute two
        // bits to each of the preceding three groups. Load each input once.
        const __m256i first = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(o_raw));
        const __m256i second =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(o_raw + 32));
        const __m128i third = _mm256_castsi256_si128(second);
        const __m128i last = _mm256_extracti128_si256(second, 1);
        const __m256i high =
            _mm256_sllv_epi32(_mm256_permute4x64_epi64(second, 0xEE), shifts);
        // Quantized codes are in [0, 63], so the low six bits need no mask.
        const __m256i packed = _mm256_or_si256(first, _mm256_andnot_si256(mask, high));
        const __m128i mask128 = _mm256_castsi256_si128(mask);
        const __m128i end =
            _mm_or_si128(third, _mm_andnot_si128(mask128, _mm_slli_epi16(last, 2)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(o_compact), packed);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 32), end);
        o_raw += 64;
        o_compact += 48;
    }
    if (dim % 64 != 0) {
        const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(o_raw));
        const __m256i pairs = _mm256_maddubs_epi16(_mm256_set1_epi16(0x4001), values);
        const __m256i groups = _mm256_madd_epi16(pairs, _mm256_set1_epi32(0x10000001));
        // Each lane holds 12 packed bytes. Remove the gap between lanes before
        // writing the complete 24-byte tail.
        const __m128i shuffle =
            _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -128, -128, -128, -128);
        const __m256i lanes =
            _mm256_shuffle_epi8(groups, _mm256_broadcastsi128_si256(shuffle));
        const __m256i packed =
            _mm256_permutevar8x32_epi32(lanes, _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7));
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(o_compact), _mm256_castsi256_si128(packed)
        );
        _mm_storel_epi64(
            reinterpret_cast<__m128i*>(o_compact + 16), _mm256_extracti128_si256(packed, 1)
        );
    }
}

void packing_7bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_7bit_excode_intrinsics(o_raw, o_compact, dim);
}

}  // namespace rabitqlib::simd
