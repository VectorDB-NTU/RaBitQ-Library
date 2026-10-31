#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "rabitqlib/fastscan/detail/simd.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"

namespace rabitqlib::fastscan::simd {
namespace {
template <bool HighAccuracy>
void accumulate_impl(
    const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim
) {
    std::array<int64_t, 32> totals{};
    // Bound each uint32 lane to 256*65535, then widen across chunks.
    constexpr size_t kChunkGroups = 256;
    for (size_t start = 0; start < dim / 4; start += kChunkGroups) {
        uint32x4_t sums[8];
        for (auto& sum : sums)
            sum = vdupq_n_u32(0);
        const size_t end = std::min(start + kChunkGroups, dim / 4);
        for (size_t group = start; group < end; ++group) {
            const uint8x16_t packed = vld1q_u8(codes + group * 16);
            const uint8x16_t indices[2] = {
                vandq_u8(packed, vdupq_n_u8(15)), vshrq_n_u8(packed, 4)};
            const size_t offset =
                HighAccuracy ? group / 2 * 64 + group % 2 * 16 : group * 16;
            const uint8x16_t low = vld1q_u8(lut + offset);
            for (size_t half = 0; half < 2; ++half) {
                const uint8x16_t values = vqtbl1q_u8(low, indices[half]);
                uint16x8_t lo = vmovl_u8(vget_low_u8(values));
                uint16x8_t hi = vmovl_u8(vget_high_u8(values));
                if constexpr (HighAccuracy) {
                    const uint8x16_t high =
                        vqtbl1q_u8(vld1q_u8(lut + offset + 32), indices[half]);
                    lo = vorrq_u16(lo, vshlq_n_u16(vmovl_u8(vget_low_u8(high)), 8));
                    hi = vorrq_u16(hi, vshlq_n_u16(vmovl_u8(vget_high_u8(high)), 8));
                }
                sums[half * 4] = vaddw_u16(sums[half * 4], vget_low_u16(lo));
                sums[half * 4 + 1] = vaddw_u16(sums[half * 4 + 1], vget_high_u16(lo));
                sums[half * 4 + 2] = vaddw_u16(sums[half * 4 + 2], vget_low_u16(hi));
                sums[half * 4 + 3] = vaddw_u16(sums[half * 4 + 3], vget_high_u16(hi));
            }
        }
        uint32_t chunk[32];
        for (size_t i = 0; i < 8; ++i)
            vst1q_u32(chunk + i * 4, sums[i]);
        for (size_t i = 0; i < 32; ++i)
            totals[i] += chunk[i];
    }
    for (auto value : totals) {
        if (value > std::numeric_limits<int32_t>::max()) {
            throw std::overflow_error(
                HighAccuracy ? "high-accuracy FastScan result exceeds int32_t"
                             : "FastScan result exceeds int32_t"
            );
        }
    }
    for (size_t i = 0; i < 32; ++i) {
        result[kPerm0[i % 16] + i / 16 * 16] = static_cast<int32_t>(totals[i]);
    }
}
}  // namespace
void accumulate_neon(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lut,
    int32_t* __restrict__ result,
    size_t dim
) {
    accumulate_impl<false>(codes, lut, result, dim);
}
void accumulate_hacc_neon(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lut,
    int32_t* result,
    size_t dim
) {
    accumulate_impl<true>(codes, lut, result, dim);
}
}  // namespace rabitqlib::fastscan::simd

namespace rabitqlib::fastscan::simd {
namespace {
template <bool ComputeRange>
void pack_lut_impl(
    size_t dim, const float* query, float* lut, float& minimum, float& maximum
) {
    auto min_values = vdupq_n_f32(std::numeric_limits<float>::infinity()),
         max_values = vdupq_n_f32(-std::numeric_limits<float>::infinity());
    auto nan_mask = vdupq_n_u32(0);

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
            if constexpr (ComputeRange) {
                min_values = vminq_f32(min_values, table);
                max_values = vmaxq_f32(max_values, table);
                nan_mask = vorrq_u32(nan_mask, vmvnq_u32(vceqq_f32(table, table)));
            }
            vst1q_f32(lut + i * 4, table);
        }
        query += 4;
        lut += 16;
    }
    if constexpr (ComputeRange) {
        minimum = vmaxvq_u32(nan_mask) ? std::numeric_limits<float>::quiet_NaN()
                                       : vminvq_f32(min_values);
        maximum = vmaxvq_f32(max_values);
    }
}
}  // namespace

void pack_lut_neon(size_t dim, const float* query, float* lut) {
    float unused_minimum = 0, unused_maximum = 0;
    pack_lut_impl<false>(dim, query, lut, unused_minimum, unused_maximum);
}

// Preserve the existing ordered subset additions; reduce their values in registers.
// A NaN minimum signals that the wrapper must use the original reduction.
void pack_lut_range_neon(
    size_t dim, const float* query, float* lut, float& minimum, float& maximum
) {
    pack_lut_impl<true>(dim, query, lut, minimum, maximum);
}
}  // namespace rabitqlib::fastscan::simd
