#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#include "hnsw_neon_kernels.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/simd/warmup_dispatch.hpp"

namespace rabitqlib::simd {
namespace {
struct FloatDot {
    float32x4_t sum0 = vdupq_n_f32(0), sum1 = sum0, sum2 = sum0, sum3 = sum0;

    void add(uint8x16_t codes, const float* query) {
        const uint16x8_t lo = vmovl_u8(vget_low_u8(codes));
        const uint16x8_t hi = vmovl_u8(vget_high_u8(codes));
        sum0 =
            vfmaq_f32(sum0, vld1q_f32(query), vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))));
        sum1 = vfmaq_f32(
            sum1, vld1q_f32(query + 4), vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo)))
        );
        sum2 = vfmaq_f32(
            sum2, vld1q_f32(query + 8), vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)))
        );
        sum3 = vfmaq_f32(
            sum3, vld1q_f32(query + 12), vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi)))
        );
    }
    float result() const {
        return vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
    }
};

// Odd-width top bits use byte i%8, bit i/8, independently of the low-bit layout.
template <int Bits, int Group>
uint8x16_t add_top(uint8x16_t low, const uint8_t* compact) {
    const uint8x8_t top = vld1_u8(compact + (Bits - 1) * 8);
    const uint8x16_t repeated = vcombine_u8(top, top);
    const int8x16_t shift =
        vcombine_s8(vdup_n_s8(Bits - 1 - Group * 2), vdup_n_s8(Bits - 2 - Group * 2));
    return vorrq_u8(low, vandq_u8(vshlq_u8(repeated, shift), vdupq_n_u8(1U << (Bits - 1))));
}

template <int Bits>
float packed_ip(const float* query, const uint8_t* compact, size_t dim) {
    FloatDot sum;
    if constexpr (Bits == 1 || Bits == 4 || Bits == 8) {
        for (size_t i = 0; i < dim; i += 16) {
            uint8x16_t codes;
            if constexpr (Bits == 1) {
                const int8x8_t shifts = {0, -1, -2, -3, -4, -5, -6, -7};
                codes = vandq_u8(
                    vcombine_u8(
                        vshl_u8(vdup_n_u8(compact[0]), shifts),
                        vshl_u8(vdup_n_u8(compact[1]), shifts)
                    ),
                    vdupq_n_u8(1)
                );
            } else if constexpr (Bits == 4) {
                const uint8x8_t packed = vld1_u8(compact);
                codes = vcombine_u8(vand_u8(packed, vdup_n_u8(15)), vshr_n_u8(packed, 4));
            } else {
                codes = vld1q_u8(compact);
            }
            sum.add(codes, query + i);
            compact += 2 * Bits;
        }
    } else {
        for (size_t i = 0; i < dim; i += 64) {
            uint8x16_t c0, c1, c2, c3;
            if constexpr (Bits <= 3) {
                const uint8x16_t packed = vld1q_u8(compact), mask = vdupq_n_u8(3);
                c0 = vandq_u8(packed, mask);
                c1 = vandq_u8(vshrq_n_u8(packed, 2), mask);
                c2 = vandq_u8(vshrq_n_u8(packed, 4), mask);
                c3 = vshrq_n_u8(packed, 6);
            } else if constexpr (Bits == 5) {
                const uint8x16_t lo = vld1q_u8(compact), hi = vld1q_u8(compact + 16);
                c0 = vandq_u8(lo, vdupq_n_u8(15));
                c1 = vshrq_n_u8(lo, 4);
                c2 = vandq_u8(hi, vdupq_n_u8(15));
                c3 = vshrq_n_u8(hi, 4);
            } else {
                const uint8x16_t p0 = vld1q_u8(compact), p1 = vld1q_u8(compact + 16),
                                 p2 = vld1q_u8(compact + 32);
                c0 = vandq_u8(p0, vdupq_n_u8(63));
                c1 = vandq_u8(p1, vdupq_n_u8(63));
                c2 = vandq_u8(p2, vdupq_n_u8(63));
                c3 = vorrq_u8(
                    vshrq_n_u8(p0, 6),
                    vorrq_u8(
                        vshlq_n_u8(vshrq_n_u8(p1, 6), 2), vshlq_n_u8(vshrq_n_u8(p2, 6), 4)
                    )
                );
            }
            if constexpr (Bits % 2 == 1) {
                c0 = add_top<Bits, 0>(c0, compact);
                c1 = add_top<Bits, 1>(c1, compact);
                c2 = add_top<Bits, 2>(c2, compact);
                c3 = add_top<Bits, 3>(c3, compact);
            }
            sum.add(c0, query + i);
            sum.add(c1, query + i + 16);
            sum.add(c2, query + i + 32);
            sum.add(c3, query + i + 48);
            compact += 8 * Bits;
        }
    }
    return sum.result();
}
}  // namespace

namespace excode_ipimpl {
#define RABITQ_NEON_IP(Block, Bits)                         \
    float ip##Block##_fxu##Bits##_neon(                     \
        const float* query, const uint8_t* code, size_t dim \
    ) {                                                     \
        return packed_ip<Bits>(query, code, dim);           \
    }
RABITQ_NEON_IP(16, 1)
RABITQ_NEON_IP(64, 2)
RABITQ_NEON_IP(64, 3)
RABITQ_NEON_IP(16, 4)
RABITQ_NEON_IP(64, 5)
RABITQ_NEON_IP(64, 6)
RABITQ_NEON_IP(64, 7)
RABITQ_NEON_IP(16, 8)
#undef RABITQ_NEON_IP
}  // namespace excode_ipimpl

float mask_ip_x0_q_neon(const float* query, const uint8_t* data, size_t dim) {
    return detail::mask_ip_neon(query, data, dim);
}
float mask_ip_x0_q_neon(const float* query, const uint64_t* data, size_t dim) {
    return detail::mask_ip_neon(query, reinterpret_cast<const uint8_t*>(data), dim);
}
float warmup_ip_x0_q_512_neon(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t dim,
    size_t bits
) {
    return detail::warmup_ip_neon(data, query, delta, vl, dim, bits);
}
float warmup_ip_x0_q_512_neon(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t dim,
    size_t bits
) {
    return detail::warmup_ip_neon(
        reinterpret_cast<const uint8_t*>(data), query, delta, vl, dim, bits
    );
}
}  // namespace rabitqlib::simd
