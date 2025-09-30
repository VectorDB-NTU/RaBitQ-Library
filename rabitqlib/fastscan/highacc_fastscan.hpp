#pragma once

#include <simde/x86/avx512.h>
#include "../utils/simde-utils.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rabitqlib::fastscan {
/**
 * @brief Change u16 lookup table to u8. Since we use more bits (higher accuracy)
 * to quantize data vector by rabitq+, we also needs to increase the accuracy of data in
 * lut.
 * We split the higher & lower 8 bits of a u16 into two sub luts.
 **/
inline void transfer_lut_hacc(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
    size_t num_codebook = dim >> 2;

    for (size_t i = 0; i < num_codebook; i++) {

        constexpr size_t kRegBits = 512;
        constexpr size_t kLaneBits = 128;
        constexpr size_t kByteBits = 8;

        constexpr size_t kLutPerIter = kRegBits / kLaneBits;
        constexpr size_t kCodePerIter = 2 * kRegBits / kByteBits;
        constexpr size_t kCodePerLine = kLaneBits / kByteBits;

        uint8_t* fill_lo =
            hc_lut + (i / kLutPerIter * kCodePerIter) + ((i % kLutPerIter) * kCodePerLine);
        uint8_t* fill_hi = fill_lo + (kRegBits / kByteBits);

        simde__m512i tmp = simde_mm512_cvtepi16_epi32(simde_mm256_loadu_epi16(lut));
        simde__m128i lo = simde_mm512_cvtepi32_epi8(tmp);
        simde__m128i hi = simde_mm512_cvtepi32_epi8(simde_mm512_srli_epi32(tmp, 8));
        simde_mm_store_si128(reinterpret_cast<simde__m128i*>(fill_lo), lo);
        simde_mm_store_si128(reinterpret_cast<simde__m128i*>(fill_hi), hi);

        lut += 16;
    }
}

inline void accumulate_hacc(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
) {
    simde__m512i low_mask = simde_mm512_set1_epi8(0xf);
    simde__m512i accu[2][4];

    for (auto& a : accu) {
        for (auto& reg : a) {
            reg = simde_mm512_setzero_si512();
        }
    }

    size_t num_codebook = dim >> 2;

    // std::cerr << "FastScan YES!" << std::endl;
    for (size_t m = 0; m < num_codebook; m += 4) {
        simde__m512i c = simde_mm512_loadu_si512(codes);
        simde__m512i lo = simde_mm512_and_si512(c, low_mask);
        simde__m512i hi = simde_mm512_and_si512(simde_mm512_srli_epi16(c, 4), low_mask);

        // accumulate lower & upper results respectively
        // accu[0][0-3] for lower 8-bit result
        // accu[1][0-3] for upper 8-bit result
        for (auto& i : accu) {
            simde__m512i lut = simde_mm512_loadu_si512(hc_lut);

            simde__m512i res_lo = simde_mm512_shuffle_epi8(lut, lo);
            simde__m512i res_hi = simde_mm512_shuffle_epi8(lut, hi);

            i[0] = simde_mm512_add_epi16(i[0], res_lo);
            i[1] = simde_mm512_add_epi16(i[1], simde_mm512_srli_epi16(res_lo, 8));

            i[2] = simde_mm512_add_epi16(i[2], res_hi);
            i[3] = simde_mm512_add_epi16(i[3], simde_mm512_srli_epi16(res_hi, 8));

            hc_lut += 64;
        }
        codes += 64;
    }

    // std::cerr << "FastScan YES!" << std::endl;

    simde__m512i res[2];
    simde__m512i dis0[2];
    simde__m512i dis1[2];

    for (size_t i = 0; i < 2; ++i) {
        simde__m256i tmp0 = simde_mm256_add_epi16(
            simde_mm512_castsi512_si256(accu[i][0]), simde_mm512_extracti64x4_epi64(accu[i][0], 1)
        );
        simde__m256i tmp1 = simde_mm256_add_epi16(
            simde_mm512_castsi512_si256(accu[i][1]), simde_mm512_extracti64x4_epi64(accu[i][1], 1)
        );
        tmp0 = simde_mm256_sub_epi16(tmp0, simde_mm256_slli_epi16(tmp1, 8));

        dis0[i] = simde_mm512_add_epi32(
            simde_mm512_cvtepu16_epi32(simde_mm256_permute2f128_si256(tmp0, tmp1, 0x21)),
            simde_mm512_cvtepu16_epi32(simde_mm256_blend_epi32(tmp0, tmp1, 0xF0))
        );

        simde__m256i tmp2 = simde_mm256_add_epi16(
            simde_mm512_castsi512_si256(accu[i][2]), simde_mm512_extracti64x4_epi64(accu[i][2], 1)
        );
        simde__m256i tmp3 = simde_mm256_add_epi16(
            simde_mm512_castsi512_si256(accu[i][3]), simde_mm512_extracti64x4_epi64(accu[i][3], 1)
        );
        tmp2 = simde_mm256_sub_epi16(tmp2, simde_mm256_slli_epi16(tmp3, 8));

        dis1[i] = simde_mm512_add_epi32(
            simde_mm512_cvtepu16_epi32(simde_mm256_permute2f128_si256(tmp2, tmp3, 0x21)),
            simde_mm512_cvtepu16_epi32(simde_mm256_blend_epi32(tmp2, tmp3, 0xF0))
        );
    }
    // shift res of high, add res of low
    res[0] =
        simde_mm512_add_epi32(dis0[0], simde_mm512_slli_epi32(dis0[1], 8));  // res for vec 0 to 15
    res[1] =
        simde_mm512_add_epi32(dis1[0], simde_mm512_slli_epi32(dis1[1], 8));  // res for vec 16 to 31

    simde_mm512_storeu_epi32(accu_res, res[0]);
    simde_mm512_storeu_epi32(accu_res + 16, res[1]);
}
}  // namespace rabitqlib::fastscan