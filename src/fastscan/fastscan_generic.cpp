#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "rabitqlib/fastscan/detail/simd.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"

namespace rabitqlib::fastscan::simd {
namespace {
template <bool ComputeRange>
void pack_lut_impl(
    size_t dim, const float* query, float* lut, float& minimum, float& maximum
) {
    float min_values = std::numeric_limits<float>::infinity(),
          max_values = -std::numeric_limits<float>::infinity();
    bool nan_mask = false;

    // Keep the initial +0 addition observable; MSVC otherwise turns +0 + -0 into -0.
    volatile float zero = 0;
    for (size_t group = 0; group < dim / 4; ++group) {
        lut[0] = zero;
        for (size_t j = 1; j < 16; ++j) {
            const size_t lowbit = j & (size_t{0} - j);
            lut[j] = lut[j - lowbit] + query[kPos[j]];
        }
        if constexpr (ComputeRange) {
            for (size_t j = 0; j < 16; ++j) {
                min_values = std::min(min_values, lut[j]);
                max_values = std::max(max_values, lut[j]);
                nan_mask |= std::isnan(lut[j]);
            }
        }
        query += 4;
        lut += 16;
    }
    if constexpr (ComputeRange) {
        minimum = nan_mask ? std::numeric_limits<float>::quiet_NaN() : min_values;
        maximum = max_values;
    }
}
}  // namespace

void pack_lut_generic(size_t dim, const float* query, float* lut) {
    float unused_minimum = 0, unused_maximum = 0;
    pack_lut_impl<false>(dim, query, lut, unused_minimum, unused_maximum);
}

// Preserve the existing ordered subset additions; reduce their values in registers.
// A NaN minimum signals that the wrapper must use the original reduction.
void pack_lut_range_generic(
    size_t dim, const float* query, float* lut, float& minimum, float& maximum
) {
    pack_lut_impl<true>(dim, query, lut, minimum, maximum);
}
}  // namespace rabitqlib::fastscan::simd

namespace rabitqlib::fastscan::simd {
void transfer_lut_hacc_generic(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
    // HaccLutLayout::Pair32: two low-byte tables, then two high-byte tables.
    for (size_t group = 0; group < dim / 4; ++group) {
        const size_t offset = group / 2 * 64 + group % 2 * 16;
        for (size_t i = 0; i < 16; ++i) {
            hc_lut[offset + i] = static_cast<uint8_t>(lut[group * 16 + i]);
            hc_lut[offset + 32 + i] = static_cast<uint8_t>(lut[group * 16 + i] >> 8);
        }
    }
}
namespace {
template <bool HighAccuracy>
void accumulate_generic_impl(
    const uint8_t* codes,
    const uint8_t* lookup_table,
    int32_t* accumulated_values,
    size_t dim
) {
    int64_t totals[kBatchSize]{};
    for (size_t group = 0; group < dim / 4; ++group) {
        const size_t offset = HighAccuracy ? group / 2 * 64 + group % 2 * 16 : group * 16;
        for (size_t lane = 0; lane < 16; ++lane) {
            const uint8_t code = codes[group * 16 + lane];
            for (size_t half = 0; half < 2; ++half) {
                const size_t entry = (code >> (half * 4)) & 15;
                int value = lookup_table[offset + entry];
                if constexpr (HighAccuracy) {
                    value += int{lookup_table[offset + 32 + entry]} * 256;
                }
                totals[kPerm0[lane] + half * 16] += value;
            }
        }
    }
    for (size_t lane = 0; lane < kBatchSize; ++lane) {
        if (totals[lane] > std::numeric_limits<int32_t>::max()) {
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
void accumulate_generic(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* __restrict__ accumulated_values,
    size_t dim
) {
    accumulate_generic_impl<false>(codes, lookup_table, accumulated_values, dim);
}
void accumulate_unsupported(
    const uint8_t* __restrict__, const uint8_t* __restrict__, int32_t* __restrict__, size_t
) {
    throw std::runtime_error(
        "Standard FastScan accumulation requires AVX2/FMA, AVX-512, or ARM NEON; "
        "no supported SIMD backend is available"
    );
}
void accumulate_hacc_generic(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* accumulated_values,
    size_t dim
) {
    accumulate_generic_impl<true>(codes, lookup_table, accumulated_values, dim);
}
}  // namespace rabitqlib::fastscan::simd
