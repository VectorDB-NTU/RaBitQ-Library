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
// Each chunk has at most 256 groups, so each byte plane sums to at most 65280.
constexpr size_t kChunkGroups = 256;

template <bool HighAccuracy>
void accumulate_chunk(
    const uint8_t* codes, const uint8_t* lut, int32_t* output, size_t groups
) {
    uint16x8_t low_sums[4]{}, high_sums[4]{};
    for (size_t group = 0; group < groups; ++group) {
        const uint8x16_t packed = vld1q_u8(codes + group * 16);
        const uint8x16_t indices[2] = {
            vandq_u8(packed, vdupq_n_u8(15)), vshrq_n_u8(packed, 4)};
        // Pair32 stores low0, low1, high0, high1 as four 16-byte tables.
        const size_t offset = HighAccuracy ? group / 2 * 64 + group % 2 * 16 : group * 16;
        const uint8x16_t low = vld1q_u8(lut + offset);
        for (size_t half = 0; half < 2; ++half) {
            const uint8x16_t values = vqtbl1q_u8(low, indices[half]);
            low_sums[2 * half] = vaddw_u8(low_sums[2 * half], vget_low_u8(values));
            low_sums[2 * half + 1] = vaddw_high_u8(low_sums[2 * half + 1], values);
            if constexpr (HighAccuracy) {
                const uint8x16_t high =
                    vqtbl1q_u8(vld1q_u8(lut + offset + 32), indices[half]);
                high_sums[2 * half] = vaddw_u8(high_sums[2 * half], vget_low_u8(high));
                high_sums[2 * half + 1] = vaddw_high_u8(high_sums[2 * half + 1], high);
            }
        }
    }
    for (size_t half = 0; half < 2; ++half) {
        // kPerm0 interleaves rows 0..7 with 8..15 in each half of the batch.
        const auto low = vuzpq_u16(low_sums[2 * half], low_sums[2 * half + 1]);
        const auto high = vuzpq_u16(high_sums[2 * half], high_sums[2 * half + 1]);
        for (size_t i = 0; i < 2; ++i) {
            uint32x4_t first = vmovl_u16(vget_low_u16(low.val[i]));
            uint32x4_t last = vmovl_high_u16(low.val[i]);
            if constexpr (HighAccuracy) {
                first = vaddq_u32(first, vshll_n_u16(vget_low_u16(high.val[i]), 8));
                last = vaddq_u32(last, vshll_high_n_u16(high.val[i], 8));
            }
            int32_t* dst = output + half * 16 + i * 8;
            vst1q_s32(dst, vreinterpretq_s32_u32(first));
            vst1q_s32(dst + 4, vreinterpretq_s32_u32(last));
        }
    }
}

template <bool HighAccuracy>
void accumulate_impl(
    const uint8_t* codes, const uint8_t* lut, int32_t* accumulated_values, size_t dim
) {
    const size_t groups = dim / 4;
    if (groups <= kChunkGroups) {
        accumulate_chunk<HighAccuracy>(codes, lut, accumulated_values, groups);
        return;
    }
    std::array<int64_t, kBatchSize> totals{};
    for (size_t start = 0; start < groups; start += kChunkGroups) {
        std::array<int32_t, kBatchSize> chunk;
        accumulate_chunk<HighAccuracy>(
            codes + start * 16,
            lut + start * (HighAccuracy ? 32 : 16),
            chunk.data(),
            std::min(kChunkGroups, groups - start)
        );
        for (size_t i = 0; i < kBatchSize; ++i) {
            totals[i] += chunk[i];
        }
    }
    for (auto value : totals) {
        if (value > std::numeric_limits<int32_t>::max()) {
            throw std::overflow_error(
                HighAccuracy ? "FastScan high-accuracy result exceeds int32_t"
                             : "FastScan result exceeds int32_t"
            );
        }
    }
    for (size_t lane = 0; lane < kBatchSize; ++lane) {
        accumulated_values[lane] = static_cast<int32_t>(totals[lane]);
    }
}
}  // namespace
void accumulate_neon(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* __restrict__ accumulated_values,
    size_t dim
) {
    accumulate_impl<false>(codes, lookup_table, accumulated_values, dim);
}
void accumulate_hacc_neon(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* accumulated_values,
    size_t dim
) {
    accumulate_impl<true>(codes, lookup_table, accumulated_values, dim);
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
