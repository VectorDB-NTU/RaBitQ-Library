#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/simd/fastscan_dispatch.hpp"

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
    const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim
) {
    accumulate_impl<false>(codes, lut, result, dim);
}
void accumulate_hacc_neon(
    const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim
) {
    accumulate_impl<true>(codes, lut, result, dim);
}
}  // namespace rabitqlib::fastscan::simd
