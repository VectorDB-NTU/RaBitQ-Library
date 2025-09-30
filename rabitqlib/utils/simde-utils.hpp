#pragma once

#include <simde/x86/avx512.h>
#include <simde/x86/avx.h>

inline float simde_mm512_reduce_add_ps(simde__m512 sum) {
    simde__m256 low  = simde_mm512_castps512_ps256(sum);
    simde__m256 high = simde_mm512_extractf32x8_ps(sum, 1);
    simde__m256 sum256 = simde_mm256_add_ps(low, high);

    simde__m128 lo128 = simde_mm256_castps256_ps128(sum256);
    simde__m128 hi128 = simde_mm256_extractf128_ps(sum256, 1);
    simde__m128 sum128 = simde_mm_add_ps(lo128, hi128);
    sum128 = simde_mm_hadd_ps(sum128, sum128);
    sum128 = simde_mm_hadd_ps(sum128, sum128);
    return simde_mm_cvtss_f32(sum128);
}

inline simde__m512i simde_mm512_cvtepu8_epi32(simde__m128i a) {
    alignas(16) uint8_t vals_u8[16];
    simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(vals_u8), a);

    int32_t vals_i32[16];
    for (int i = 0; i < 16; ++i) {
        vals_i32[i] = static_cast<uint32_t>(vals_u8[i]);  // zero-extend
    }

    return simde_mm512_loadu_epi32(vals_i32);
}

inline simde__m512i simde_mm512_cvtepi8_epi32(simde__m128i a) {
    alignas(16) int8_t vals_i8[16];
    simde_mm_storeu_si128((simde__m128i*)vals_i8, a);

    int32_t vals_i32[16];
    for (int i = 0; i < 16; ++i) {
        vals_i32[i] = (int32_t)vals_i8[i];  // sign extend
    }

    return simde_mm512_loadu_epi32(vals_i32);
}

inline simde__mmask16 simde_cvtu32_mask16(unsigned int a) {
    return (simde__mmask16)(a & 0xFFFFu);
}

inline simde__m128i simde_mm512_cvtepi32_epi8(simde__m512i a) {
    alignas(64) int32_t tmp32[16];
    simde_mm512_storeu_epi32(tmp32, a);

    alignas(16) int8_t tmp8[16];
    for (int i = 0; i < 16; ++i) {
        tmp8[i] = (int8_t)(tmp32[i]);
    }

    return simde_mm_loadu_epi8(tmp8);
}

inline simde__m256i simde_mm512_cvtepi32_epi16(simde__m512i a) {
    alignas(64) int32_t input[16];
    simde_mm512_storeu_epi32(input, a);

    alignas(32) int16_t output[16];
    for (int i = 0; i < 16; ++i) {
        output[i] = (int16_t)(input[i]);
    }

    return simde_mm256_loadu_epi16(output);
}