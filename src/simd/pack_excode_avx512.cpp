#include <cstddef>
#include <cstdint>

#include "pack_excode_kernels.hpp"
#include "rabitqlib/simd/pack_excode_dispatch.hpp"

namespace rabitqlib::simd {

namespace {
void pack_2bit_tail(const uint8_t* raw, uint8_t* compact, size_t dim) {
    if (dim == 0) {
        return;
    }
    // Each four input bytes become one output byte; both operations use the
    // same lane mask. The tail contains exactly 32 coordinates.
    constexpr __mmask16 mask = 0xFF;
    const __m512i values = _mm512_maskz_loadu_epi32(mask, raw);
    const __m512i pairs = _mm512_maddubs_epi16(_mm512_set1_epi16(0x0401), values);
    const __m512i groups = _mm512_madd_epi16(pairs, _mm512_set1_epi32(0x00100001));
    _mm512_mask_cvtepi32_storeu_epi8(compact, mask, groups);
}

void pack_6bit_tail(const uint8_t* raw, uint8_t* compact, size_t dim) {
    if (dim == 0) {
        return;
    }
    constexpr __mmask16 input_mask = 0xFF;
    const __m512i values = _mm512_maskz_loadu_epi32(input_mask, raw);
    const __m512i pairs = _mm512_maddubs_epi16(_mm512_set1_epi16(0x4001), values);
    const __m512i groups = _mm512_madd_epi16(pairs, _mm512_set1_epi32(0x10000001));
    // Four codes occupy three bytes. Compact each 128-bit lane into three
    // words, then compress those words into exactly 24 output bytes.
    const __m128i shuffle =
        _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -128, -128, -128, -128);
    const __m512i lanes = _mm512_shuffle_epi8(groups, _mm512_broadcast_i32x4(shuffle));
    constexpr __mmask16 selected = 0x77;
    _mm512_mask_compressstoreu_epi32(compact, selected, lanes);
}
}  // namespace

void packing_2bit_excode_avx512(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_2bit_excode_intrinsics<pack_2bit_tail>(o_raw, o_compact, dim);
}

// Reuse the AVX2 odd-width packers: wider code generation adds overhead here.
void packing_3bit_excode_avx512(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    packing_3bit_excode_avx2(o_raw, o_compact, dim);
}

void packing_4bit_excode_avx512(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_4bit_excode_intrinsics<128>(o_raw, o_compact, dim);
}

void packing_5bit_excode_avx512(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    packing_5bit_excode_avx2(o_raw, o_compact, dim);
}

void packing_6bit_excode_avx512(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_6bit_excode_intrinsics<pack_6bit_tail>(o_raw, o_compact, dim);
}

void packing_7bit_excode_avx512(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    packing_7bit_excode_avx2(o_raw, o_compact, dim);
}

}  // namespace rabitqlib::simd
