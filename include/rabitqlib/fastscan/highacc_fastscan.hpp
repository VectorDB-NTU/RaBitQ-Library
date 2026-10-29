#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::fastscan {
/**
 * @brief Store a 16-bit query LUT as low/high byte planes for HACC.
 * Input contains dim * 4 uint16_t entries; output needs dim * 8 bytes.
 * The runtime-selected backend determines the byte-plane layout. Pair this
 * dispatched producer with accumulate_hacc rather than assuming an ISA layout.
 */
void transfer_lut_hacc(const uint16_t* lut, size_t dim, uint8_t* hc_lut);

// Accumulate one batch into int32_t results. Throws if a result exceeds int32_t.
void accumulate_hacc(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accumulated_values,
    size_t dim
);
}  // namespace rabitqlib::fastscan
