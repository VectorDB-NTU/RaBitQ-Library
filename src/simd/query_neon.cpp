#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "rabitqlib/simd/fastscan_dispatch.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"

namespace rabitqlib::simd {
namespace {
// Convert eight selected byte lanes to MSB-first coordinate bits.
inline uint8_t byte_mask(uint8x8_t selected) {
    const uint8x8_t weights = {128, 64, 32, 16, 8, 4, 2, 1};
    return vaddv_u8(vand_u8(selected, weights));
}

inline uint16_t byte_mask(uint8x16_t selected) {
    return (uint16_t{byte_mask(vget_low_u8(selected))} << 8) |
           byte_mask(vget_high_u8(selected));
}

inline uint32x4_t quantize4(const float* data, float32x4_t lo, float32x4_t reciprocal) {
    // FCVTAU rounds ties away from zero, unlike nearest-even FCVTNU.
    // Callers provide finite values whose rounded codes fit the output type.
    return vcvtaq_u32_f32(vmulq_f32(vsubq_f32(vld1q_f32(data), lo), reciprocal));
}
}  // namespace

void scalar_quantize_uint8_neon(
    uint8_t* result, const float* data, size_t dim, float lo, float delta
) {
    const float reciprocal = 1.0F / delta;
    const float32x4_t low = vdupq_n_f32(lo), scale = vdupq_n_f32(reciprocal);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const uint16x8_t codes = vcombine_u16(
            vmovn_u32(quantize4(data + i, low, scale)),
            vmovn_u32(quantize4(data + i + 4, low, scale))
        );
        vst1_u8(result + i, vmovn_u16(codes));
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint8_t>(std::round((data[i] - lo) * reciprocal));
    }
}

void scalar_quantize_uint16_neon(
    uint16_t* result, const float* data, size_t dim, float lo, float delta
) {
    const float reciprocal = 1.0F / delta;
    const float32x4_t low = vdupq_n_f32(lo), scale = vdupq_n_f32(reciprocal);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        vst1q_u16(
            result + i,
            vcombine_u16(
                vmovn_u32(quantize4(data + i, low, scale)),
                vmovn_u32(quantize4(data + i + 4, low, scale))
            )
        );
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint16_t>(std::round((data[i] - lo) * reciprocal));
    }
}

void new_transpose_bin_neon(
    const uint16_t* query, uint64_t* transposed, size_t dim, size_t bits
) {
    for (size_t block = 0; block < dim; block += 64) {
        uint16x8_t values[8];
        for (size_t c = 0; c < 8; ++c) {
            values[c] = vld1q_u16(query + block + c * 8);
        }
        for (size_t b = 0; b < bits; ++b) {
            const uint16x8_t bit = vdupq_n_u16(uint16_t{1} << b);
            uint64_t word = 0;
            for (auto value : values) {
                word = (word << 8) | byte_mask(vmovn_u16(vtstq_u16(value, bit)));
            }
            *transposed++ = word;
        }
    }
}

void new_transpose_bin_512_neon(
    const uint8_t* query, uint64_t* transposed, size_t dim, size_t bits
) {
    for (size_t block = 0; block < dim; block += 512) {
        const size_t chunks = std::min(size_t{512}, dim - block) / 64;
        for (size_t c = 0; c < chunks; ++c) {
            const uint8_t* row = query + block + c * 64;
            const uint8x16_t a = vld1q_u8(row), b = vld1q_u8(row + 16),
                             d = vld1q_u8(row + 32), e = vld1q_u8(row + 48);
            for (size_t plane = 0; plane < bits; ++plane) {
                const uint8x16_t bit = vdupq_n_u8(uint8_t{1} << plane);
                transposed[plane * chunks + c] =
                    (uint64_t{byte_mask(vtstq_u8(a, bit))} << 48) |
                    (uint64_t{byte_mask(vtstq_u8(b, bit))} << 32) |
                    (uint64_t{byte_mask(vtstq_u8(d, bit))} << 16) |
                    byte_mask(vtstq_u8(e, bit));
            }
        }
        transposed += chunks * bits;
    }
}
}  // namespace rabitqlib::simd

namespace rabitqlib::fastscan::simd {
void pack_lut_neon(size_t dim, const float* query, float* lut) {
    // Each table enumerates q0..q3 using bits 3..0. Add in that order,
    // including the initial +0, to preserve cancellation and signed zeros.
    volatile float zero = 0;
    const uint32x4_t select2 = {0, 0, UINT32_MAX, UINT32_MAX};
    const uint32x4_t select3 = {0, UINT32_MAX, 0, UINT32_MAX};
    for (size_t group = 0; group < dim / 4; ++group) {
        const float32x4_t z = vdupq_n_f32(zero);
        const float32x4_t q0 = vdupq_n_f32(query[0]), q1 = vdupq_n_f32(query[1]),
                          q2 = vdupq_n_f32(query[2]), q3 = vdupq_n_f32(query[3]);
        const float32x4_t high = vaddq_f32(z, q0);
        float32x4_t tables[4] = {z, vaddq_f32(z, q1), high, vaddq_f32(high, q1)};
        for (size_t i = 0; i < 4; ++i) {
            auto table = tables[i];
            table = vbslq_f32(select2, vaddq_f32(table, q2), table);
            table = vbslq_f32(select3, vaddq_f32(table, q3), table);
            vst1q_f32(lut + i * 4, table);
        }
        query += 4;
        lut += 16;
    }
}
}  // namespace rabitqlib::fastscan::simd
