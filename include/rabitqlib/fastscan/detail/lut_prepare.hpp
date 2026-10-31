#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::fastscan::detail {
// Caller validates a positive dimension divisible by 16 and non-overlapping scratch.
// Return whether every generated LUT entry is finite. Exceptional and zero bounds
// retain the original range reduction, including its signed-zero behavior.
bool pack_lut_with_range(
    size_t dim, const float* query, float* lut, float& minimum, float& maximum
);

// Use only after pack_lut_with_range returns true, with its minimum and range delta.
// General quantization and high-accuracy uint16 quantization remain unchanged.
void quantize_lut_uint8(
    uint8_t* result, const float* values, size_t count, float lo, float delta
);
}  // namespace rabitqlib::fastscan::detail
