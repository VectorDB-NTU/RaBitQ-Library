#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "rabitqlib/core/detail/simd/space_dispatch.hpp"

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

// Code is uint8_t or uint16_t; only the final narrowing and store differ.
template <typename Code>
void quantize(Code* result, const float* data, size_t dim, float lo, float delta) {
    if (delta == 0.0F) {
        std::fill_n(result, dim, Code{0});
        return;
    }
    const float reciprocal = 1.0F / delta;
    const float32x4_t low = vdupq_n_f32(lo), scale = vdupq_n_f32(reciprocal);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const uint16x8_t codes = vcombine_u16(
            vmovn_u32(quantize4(data + i, low, scale)),
            vmovn_u32(quantize4(data + i + 4, low, scale))
        );
        if constexpr (std::is_same_v<Code, uint8_t>) {
            vst1_u8(result + i, vmovn_u16(codes));
        } else {
            vst1q_u16(result + i, codes);
        }
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<Code>(std::round((data[i] - lo) * reciprocal));
    }
}
}  // namespace

void scalar_quantize_uint8_neon(
    uint8_t* result, const float* data, size_t dim, float lo, float delta
) {
    quantize(result, data, dim, lo, delta);
}

void scalar_quantize_uint16_neon(
    uint16_t* result, const float* data, size_t dim, float lo, float delta
) {
    quantize(result, data, dim, lo, delta);
}

void new_transpose_bin_neon(
    const uint16_t* query, uint64_t* transposed_query, size_t padded_dim, size_t b_query
) {
    for (size_t block = 0; block < padded_dim - padded_dim % 64; block += 64) {
        uint16x8_t values[8];
        for (size_t c = 0; c < 8; ++c) {
            values[c] = vld1q_u16(query + block + c * 8);
        }
        for (size_t b = 0; b < b_query; ++b) {
            const uint16x8_t bit = vdupq_n_u16(uint16_t{1} << b);
            uint64_t word = 0;
            for (auto value : values) {
                word = (word << 8) | byte_mask(vmovn_u16(vtstq_u16(value, bit)));
            }
            *transposed_query++ = word;
        }
    }
    if (padded_dim % 64 != 0) {
        new_transpose_bin_generic(
            query + padded_dim - padded_dim % 64, transposed_query, padded_dim % 64, b_query
        );
    }
}

void new_transpose_bin_512_neon(
    const uint8_t* query, uint64_t* transposed_query, size_t padded_dim, size_t b_query
) {
    for (size_t block = 0; block < padded_dim; block += 512) {
        const size_t block_dim = std::min(size_t{512}, padded_dim - block);
        const size_t full_chunks = block_dim / 64;
        const size_t chunks = (block_dim + 63) / 64;
        for (size_t c = 0; c < full_chunks; ++c) {
            const uint8_t* row = query + block + c * 64;
            const uint8x16_t a = vld1q_u8(row), b = vld1q_u8(row + 16),
                             d = vld1q_u8(row + 32), e = vld1q_u8(row + 48);
            for (size_t plane = 0; plane < b_query; ++plane) {
                const uint8x16_t bit = vdupq_n_u8(uint8_t{1} << plane);
                transposed_query[plane * chunks + c] =
                    (uint64_t{byte_mask(vtstq_u8(a, bit))} << 48) |
                    (uint64_t{byte_mask(vtstq_u8(b, bit))} << 32) |
                    (uint64_t{byte_mask(vtstq_u8(d, bit))} << 16) |
                    byte_mask(vtstq_u8(e, bit));
            }
        }
        const size_t tail_dim = block_dim % 64;
        if (tail_dim != 0) {
            const uint8_t* tail = query + block + full_chunks * 64;
            for (size_t plane = 0; plane < b_query; ++plane) {
                uint64_t word = 0;
                for (size_t i = 0; i < tail_dim; ++i) {
                    word |= uint64_t{(tail[i] >> plane) & 1U} << (tail_dim - 1 - i);
                }
                transposed_query[plane * chunks + full_chunks] = word;
            }
        }
        transposed_query += chunks * b_query;
    }
}
}  // namespace rabitqlib::simd
