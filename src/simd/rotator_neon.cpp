#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "rabitqlib/simd/rotator_dispatch.hpp"

namespace rabitqlib::simd {
namespace {
inline void butterfly(float32x4_t& a, float32x4_t& b) {
    const float32x4_t sum = vaddq_f32(a, b);
    b = vsubq_f32(a, b);
    a = sum;
}

inline float32x4_t fht4(float32x4_t x) {
    // Width-one butterflies, then width-two butterflies. Retain the scalar
    // operand order instead of reassociating sums or changing normalization.
    const float32x4_t paired = vrev64q_f32(x);
    x = vtrn1q_f32(vaddq_f32(x, paired), vsubq_f32(x, paired));
    const float32x2_t lo = vget_low_f32(x), hi = vget_high_f32(x);
    return vcombine_f32(vadd_f32(lo, hi), vsub_f32(lo, hi));
}

void fht(float* data, size_t dim) {
    // Fuse the first four stages in registers, avoiding a memory pass per stage.
    for (size_t i = 0; i < dim; i += 16) {
        float32x4_t a = fht4(vld1q_f32(data + i));
        float32x4_t b = fht4(vld1q_f32(data + i + 4));
        float32x4_t c = fht4(vld1q_f32(data + i + 8));
        float32x4_t d = fht4(vld1q_f32(data + i + 12));
        butterfly(a, b);
        butterfly(c, d);
        butterfly(a, c);
        butterfly(b, d);
        vst1q_f32(data + i, a);
        vst1q_f32(data + i + 4, b);
        vst1q_f32(data + i + 8, c);
        vst1q_f32(data + i + 12, d);
    }
    // Fuse pairs of larger stages, preserving the same butterfly dependencies.
    size_t width = 16;
    for (; width * 2 < dim; width *= 4) {
        for (size_t block = 0; block < dim; block += width * 4) {
            for (size_t i = block; i < block + width; i += 4) {
                float32x4_t a = vld1q_f32(data + i);
                float32x4_t b = vld1q_f32(data + i + width);
                float32x4_t c = vld1q_f32(data + i + 2 * width);
                float32x4_t d = vld1q_f32(data + i + 3 * width);
                butterfly(a, b);
                butterfly(c, d);
                butterfly(a, c);
                butterfly(b, d);
                vst1q_f32(data + i, a);
                vst1q_f32(data + i + width, b);
                vst1q_f32(data + i + 2 * width, c);
                vst1q_f32(data + i + 3 * width, d);
            }
        }
    }
    if (width < dim) {
        for (size_t i = 0; i < width; i += 4) {
            float32x4_t a = vld1q_f32(data + i), b = vld1q_f32(data + i + width);
            butterfly(a, b);
            vst1q_f32(data + i, a);
            vst1q_f32(data + i + width, b);
        }
    }
}

void rescale(float* data, size_t dim, float factor) {
    const float32x4_t scale = vdupq_n_f32(factor);
    for (size_t i = 0; i < dim; i += 4) {
        vst1q_f32(data + i, vmulq_f32(vld1q_f32(data + i), scale));
    }
}
}  // namespace

void flip_sign_neon(const uint8_t* flip, float* data, size_t dim) {
    const int32x4_t shifts_lo = {31, 30, 29, 28}, shifts_hi = {27, 26, 25, 24};
    const uint32x4_t sign = vdupq_n_u32(0x80000000U);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        // The stored mask is LSB-first: bit j flips coordinate i+j.
        const uint32x4_t bits = vdupq_n_u32(flip[i / 8]);
        const uint32x4_t lo = vandq_u32(vshlq_u32(bits, shifts_lo), sign);
        const uint32x4_t hi = vandq_u32(vshlq_u32(bits, shifts_hi), sign);
        vst1q_f32(
            data + i,
            vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vld1q_f32(data + i)), lo))
        );
        vst1q_f32(
            data + i + 4,
            vreinterpretq_f32_u32(
                veorq_u32(vreinterpretq_u32_f32(vld1q_f32(data + i + 4)), hi)
            )
        );
    }
    for (; i < dim; ++i) {
        if ((flip[i / 8] >> (i % 8)) & 1U) {
            data[i] = -data[i];
        }
    }
}

void kacs_walk_neon(float* data, size_t len) {
    const size_t half = len / 2;
    size_t i = 0;
    for (; i + 4 <= half; i += 4) {
        float32x4_t a = vld1q_f32(data + i), b = vld1q_f32(data + i + half);
        butterfly(a, b);
        vst1q_f32(data + i, a);
        vst1q_f32(data + i + half, b);
    }
    for (; i < half; ++i) {
        const float a = data[i], b = data[i + half];
        data[i] = a + b;
        data[i + half] = a - b;
    }
}

void fht_rotate_neon(
    const float* data,
    float* output,
    size_t dim,
    size_t padded,
    size_t trunc,
    float factor,
    const uint8_t* flip
) {
    if (trunc < 64 || trunc > 65536 || (trunc & (trunc - 1)) != 0) {
        throw std::invalid_argument("Unsupported dimension for FhtKacRotator");
    }
    std::memmove(output, data, dim * sizeof(float));
    std::fill(output + dim, output + padded, 0.0F);
    for (size_t pass = 0; pass < 4; ++pass) {
        flip_sign_neon(flip + pass * padded / 8, output, padded);
        const size_t start = pass % 2 == 0 ? 0 : padded - trunc;
        fht(output + start, trunc);
        rescale(output + start, trunc, factor);
        if (padded != trunc) {
            kacs_walk_neon(output, padded);
        }
    }
    if (padded != trunc) {
        rescale(output, padded, 0.25F);
    }
}
}  // namespace rabitqlib::simd
