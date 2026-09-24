#include <cstddef>
#include <limits>
#include <stdexcept>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/simd/fastscan_dispatch.hpp"

namespace rabitqlib::fastscan::simd {
void pack_lut_generic(size_t dim, const float* query, float* lut) {
    // Keep the initial +0 addition observable; MSVC otherwise turns +0 + -0 into -0.
    volatile float zero = 0;
    for (size_t group = 0; group < dim / 4; ++group) {
        lut[0] = zero;
        for (size_t j = 1; j < 16; ++j) {
            lut[j] = lut[j - LOWBIT(j)] + query[kPos[j]];
        }
        query += 4;
        lut += 16;
    }
}
}  // namespace rabitqlib::fastscan::simd

namespace rabitqlib::fastscan::simd {
void transfer_lut_hacc_generic(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
    // Match the AVX2 layout: two low-byte tables, then two high-byte tables.
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
    const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim
) {
    int64_t totals[32]{};
    for (size_t group = 0; group < dim / 4; ++group) {
        const size_t offset = HighAccuracy ? group / 2 * 64 + group % 2 * 16 : group * 16;
        for (size_t lane = 0; lane < 16; ++lane) {
            const uint8_t code = codes[group * 16 + lane];
            for (size_t half = 0; half < 2; ++half) {
                const size_t entry = (code >> (half * 4)) & 15;
                int value = lut[offset + entry];
                if constexpr (HighAccuracy)
                    value += int{lut[offset + 32 + entry]} * 256;
                totals[kPerm0[lane] + half * 16] += value;
            }
        }
    }
    for (size_t lane = 0; lane < 32; ++lane) {
        if (totals[lane] > std::numeric_limits<int32_t>::max()) {
            throw std::overflow_error(
                HighAccuracy ? "high-accuracy FastScan result exceeds int32_t"
                             : "FastScan result exceeds int32_t"
            );
        }
    }
    for (size_t lane = 0; lane < 32; ++lane)
        result[lane] = static_cast<int32_t>(totals[lane]);
}
}  // namespace
void accumulate_generic(
    const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim
) {
    accumulate_generic_impl<false>(codes, lut, result, dim);
}
void accumulate_unsupported(const uint8_t*, const uint8_t*, int32_t*, size_t) {
    throw std::runtime_error(
        "Standard FastScan accumulation requires AVX2/FMA, AVX-512, or ARM NEON; "
        "no supported SIMD backend is available"
    );
}
void accumulate_hacc_generic(
    const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim
) {
    accumulate_generic_impl<true>(codes, lut, result, dim);
}
}  // namespace rabitqlib::fastscan::simd
