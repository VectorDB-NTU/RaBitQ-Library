#include <arm_neon.h>

#include "rabitqlib/simd/fastscan_dispatch.hpp"

namespace rabitqlib::fastscan::simd {

void accumulate_neon(
    const std::uint8_t* codes,
    const std::uint8_t* lut,
    std::uint16_t* result,
    std::size_t dim
) {
    auto lo0 = vdupq_n_u16(0);
    auto lo1 = vdupq_n_u16(0);
    auto hi0 = vdupq_n_u16(0);
    auto hi1 = vdupq_n_u16(0);
    const auto mask = vdupq_n_u8(15);

    for (std::size_t offset = 0; offset < dim * 4; offset += 16) {
        const auto packed = vld1q_u8(codes + offset);
        const auto table = vld1q_u8(lut + offset);
        const auto low = vqtbl1q_u8(table, vandq_u8(packed, mask));
        const auto high = vqtbl1q_u8(table, vshrq_n_u8(packed, 4));
        lo0 = vaddw_u8(lo0, vget_low_u8(low));
        lo1 = vaddw_high_u8(lo1, low);
        hi0 = vaddw_u8(hi0, vget_low_u8(high));
        hi1 = vaddw_high_u8(hi1, high);
    }

    // Upstream packs lanes as 0,8,1,9,...,7,15; unzip restores vector order.
    vst1q_u16(result, vuzp1q_u16(lo0, lo1));
    vst1q_u16(result + 8, vuzp2q_u16(lo0, lo1));
    vst1q_u16(result + 16, vuzp1q_u16(hi0, hi1));
    vst1q_u16(result + 24, vuzp2q_u16(hi0, hi1));
}

}  // namespace rabitqlib::fastscan::simd
