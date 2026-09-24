#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace rabitqlib::simd::detail {

inline float mask_ip_neon(const float* query, const uint8_t* data, size_t dim) {
    float32x4_t sum0 = vdupq_n_f32(0), sum1 = sum0, sum2 = sum0, sum3 = sum0;
    const int32x4_t shifts = {-31, -30, -29, -28};
    for (size_t block = 0; block < dim; block += 64) {
        uint64_t word;
        std::memcpy(&word, data + block / 8, sizeof(word));
        // Sign-code coordinates run from bit 63 to bit 0 in each stored word.
        for (size_t half = 0; half < 2; ++half) {
            uint32x4_t bits = vdupq_n_u32(static_cast<uint32_t>(word >> 32));
            for (size_t group = 0; group < 2; ++group) {
                const auto masked = [&](size_t offset, int shift) {
                    const uint32x4_t selected = vandq_u32(
                        vshlq_u32(bits, vaddq_s32(shifts, vdupq_n_s32(shift))),
                        vdupq_n_u32(1)
                    );
                    return vreinterpretq_f32_u32(vandq_u32(
                        vsubq_u32(vdupq_n_u32(0), selected),
                        vreinterpretq_u32_f32(vld1q_f32(query + offset))
                    ));
                };
                sum0 = vaddq_f32(sum0, masked(0, 0));
                sum1 = vaddq_f32(sum1, masked(4, 4));
                sum2 = vaddq_f32(sum2, masked(8, 8));
                sum3 = vaddq_f32(sum3, masked(12, 12));
                bits = vshlq_n_u32(bits, 16);
                query += 16;
            }
            word <<= 32;
        }
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
}

inline float warmup_ip_neon(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t dim,
    size_t bits
) {
    if (bits > 8) {
        throw std::invalid_argument("warmup_ip_x0_q_512 requires at most 8 query bits");
    }
    uint64_t ip = 0, count = 0;
    for (size_t block = 0; block < dim; block += 512) {
        const size_t chunks = std::min(size_t{512}, dim - block) / 64;
        size_t chunk = 0;
        for (; chunk + 2 <= chunks; chunk += 2) {
            const uint8x16_t code = vld1q_u8(data + block / 8 + chunk * 8);
            count += vaddlvq_u8(vcntq_u8(code));
            for (size_t b = 0; b < bits; ++b) {
                const uint8x16_t plane =
                    vreinterpretq_u8_u64(vld1q_u64(query + b * chunks + chunk));
                ip += uint64_t{vaddlvq_u8(vcntq_u8(vandq_u8(code, plane)))} << b;
            }
        }
        if (chunk < chunks) {
            const uint8x8_t code = vld1_u8(data + block / 8 + chunk * 8);
            count += vaddlv_u8(vcnt_u8(code));
            for (size_t b = 0; b < bits; ++b) {
                const uint8x8_t plane =
                    vreinterpret_u8_u64(vld1_u64(query + b * chunks + chunk));
                ip += uint64_t{vaddlv_u8(vcnt_u8(vand_u8(code, plane)))} << b;
            }
        }
        if (bits != 0) {
            query += chunks * bits;
        }
    }
    return delta * static_cast<float>(ip) + vl * static_cast<float>(count);
}
}  // namespace rabitqlib::simd::detail
