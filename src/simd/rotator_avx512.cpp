#include <immintrin.h>

#include <cstdint>
#include <cstring>
#include <limits>

#include "rabitqlib/simd/rotator_dispatch.hpp"
#include "rotator_kernels.hpp"

namespace rabitqlib::simd {

namespace {

// Apply the next round's signs before storing the Kac outputs.
void kacs_walk_flip(const uint8_t* flip, float* data, size_t len) {
    const size_t half = len / 2;
    const __m512 sign =
        _mm512_castsi512_ps(_mm512_set1_epi32(std::numeric_limits<int>::min()));
    for (size_t i = 0; i < half; i += 16) {
        const __m512 x = _mm512_loadu_ps(data + i);
        const __m512 y = _mm512_loadu_ps(data + half + i);
        const __m512 sum = _mm512_add_ps(x, y);
        const __m512 difference = _mm512_sub_ps(x, y);
        uint16_t left_bits, right_bits;
        std::memcpy(&left_bits, flip + i / 8, sizeof(left_bits));
        std::memcpy(&right_bits, flip + (half + i) / 8, sizeof(right_bits));
        _mm512_storeu_ps(data + i, _mm512_mask_xor_ps(sum, left_bits, sum, sign));
        _mm512_storeu_ps(
            data + half + i, _mm512_mask_xor_ps(difference, right_bits, difference, sign)
        );
    }
}

}  // namespace

void flip_sign_avx512(const uint8_t* flip, float* data, size_t dim) {
    constexpr size_t kFloatsPerChunk = 64;  // Process 64 floats per iteration

    static_assert(
        kFloatsPerChunk % 16 == 0,
        "floats_per_chunk must be divisible by AVX512 register width"
    );

    size_t i = 0;
    for (; i + kFloatsPerChunk <= dim; i += kFloatsPerChunk) {
        // Load 64 bits (8 bytes) from the bit sequence
        uint64_t mask_bits;
        std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));

        // Split into four 16-bit mask segments
        const __mmask16 mask0 = _cvtu32_mask16(static_cast<uint32_t>(mask_bits & 0xFFFF));
        const __mmask16 mask1 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 16) & 0xFFFF));
        const __mmask16 mask2 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 32) & 0xFFFF));
        const __mmask16 mask3 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 48) & 0xFFFF));

        // Prepare sign-flip constant
        const __m512 sign_flip =
            _mm512_castsi512_ps(_mm512_set1_epi32(std::numeric_limits<int>::min()));

        // Process 16 floats at a time with each mask segment
        __m512 vec0 = _mm512_loadu_ps(&data[i]);
        vec0 = _mm512_mask_xor_ps(vec0, mask0, vec0, sign_flip);
        _mm512_storeu_ps(&data[i], vec0);

        __m512 vec1 = _mm512_loadu_ps(&data[i + 16]);
        vec1 = _mm512_mask_xor_ps(vec1, mask1, vec1, sign_flip);
        _mm512_storeu_ps(&data[i + 16], vec1);

        __m512 vec2 = _mm512_loadu_ps(&data[i + 32]);
        vec2 = _mm512_mask_xor_ps(vec2, mask2, vec2, sign_flip);
        _mm512_storeu_ps(&data[i + 32], vec2);

        __m512 vec3 = _mm512_loadu_ps(&data[i + 48]);
        vec3 = _mm512_mask_xor_ps(vec3, mask3, vec3, sign_flip);
        _mm512_storeu_ps(&data[i + 48], vec3);
    }
    if (i < dim) {
        uint32_t bits;
        std::memcpy(&bits, flip + i / 8, sizeof(bits));
        const __m512 sign =
            _mm512_castsi512_ps(_mm512_set1_epi32(std::numeric_limits<int>::min()));
        const __m512 lo = _mm512_loadu_ps(data + i);
        const __m512 hi = _mm512_loadu_ps(data + i + 16);
        _mm512_storeu_ps(
            data + i, _mm512_mask_xor_ps(lo, static_cast<__mmask16>(bits), lo, sign)
        );
        _mm512_storeu_ps(
            data + i + 16,
            _mm512_mask_xor_ps(hi, static_cast<__mmask16>(bits >> 16), hi, sign)
        );
    }
}

void kacs_walk_avx512(float* data, size_t len) {
    // Each half contains whole 16-float registers.
    for (size_t i = 0; i < len / 2; i += 16) {
        __m512 x = _mm512_loadu_ps(&data[i]);
        __m512 y = _mm512_loadu_ps(&data[i + (len / 2)]);

        __m512 new_x = _mm512_add_ps(x, y);
        __m512 new_y = _mm512_sub_ps(x, y);

        _mm512_storeu_ps(&data[i], new_x);
        _mm512_storeu_ps(&data[i + (len / 2)], new_y);
    }
}

void fht_rotate_avx512(
    const float* data,
    float* rotated_vec,
    size_t dim,
    size_t padded_dim,
    size_t trunc_dim,
    float fac,
    const uint8_t* flip
) {
    fht_rotate_impl<flip_sign_avx512, kacs_walk_avx512, kacs_walk_flip>(
        data, rotated_vec, dim, padded_dim, trunc_dim, fac, flip
    );
}

}  // namespace rabitqlib::simd
