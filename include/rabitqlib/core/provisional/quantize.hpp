#pragma once

#include <cstddef>
#include <cstdint>

#include "rabitqlib/core/api.hpp"     // v1::Factors
#include "rabitqlib/core/metric.hpp"  // IWYU pragma: export

namespace rabitqlib::quant::provisional {

// Provisional quantization building blocks for rabitqlib.quant. Any minor release may
// change or remove them; they are not part of the stable quant::v1 API. Vectors are in
// the rotated, padded working space. Inputs are borrowed only for the call, outputs are
// written only after every check passes, and the calls allocate temporaries. If such an
// allocation fails, combined_code's code contents are unspecified.

/**
 * Unsigned total_bits-bit codes of one residual (vector minus centroid), the codes
 * quant::v1::encode and the indexes store: the sign bit followed by the extra bits.
 * t_const <= 0 runs the exact search; t_const > 0 is a fast-mode scale (faster_config).
 *
 * Requires dim to be a positive multiple of 32, residual to hold dim finite values,
 * total_bits in [1, 8] for uint8_t codes or [1, 9] for uint16_t codes, and code to
 * hold dim values that do not overlap residual.
 */
void combined_code(
    const float* residual, size_t dim, size_t total_bits, uint8_t* code, double t_const = -1
);
void combined_code(
    const float* residual,
    size_t dim,
    size_t total_bits,
    uint16_t* code,
    double t_const = -1
);

/**
 * Factors of a total_bits-bit code for residual. Pass the full code with its width for
 * the full-code factors, or a prefix (code >> ex_bits) with the prefix width for the
 * prefix factors. centroid == nullptr means zero; it only affects add.
 *
 * Requires dim to be a positive multiple of 32, residual and centroid to hold dim finite
 * values, every code value to be below 2^total_bits, and the code to quantize residual:
 * <residual, code - (2^total_bits - 1) / 2> must be positive unless residual is zero.
 */
v1::Factors code_factors(
    const float* residual,
    const float* centroid,
    size_t dim,
    const uint8_t* code,
    size_t total_bits,
    MetricType metric
);
v1::Factors code_factors(
    const float* residual,
    const float* centroid,
    size_t dim,
    const uint16_t* code,
    size_t total_bits,
    MetricType metric
);

}  // namespace rabitqlib::quant::provisional
