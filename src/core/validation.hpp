#pragma once

// Input checks shared by the stable v1 API (api.cpp) and the provisional API
// (provisional.cpp), so both reject the same inputs with the same messages. This header
// is private to the core sources and is not installed.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "rabitqlib/core/api.hpp"
#include "rabitqlib/utils/input_checks.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::quant::validation {

// Eigen indexes the working vectors with long.
inline void check_dim_range(size_t dim) {
    if (dim > static_cast<size_t>(std::numeric_limits<long>::max())) {
        throw std::length_error("dim exceeds the supported index range");
    }
}

inline void check_t_const(double t_const) {
    if (!std::isfinite(t_const) || t_const > std::numeric_limits<int>::max() / 2.0) {
        throw std::invalid_argument("t_const must be finite and at most INT_MAX / 2");
    }
}

inline void check_disjoint(
    const void* first, size_t first_bytes, const void* second, size_t second_bytes
) {
    if (first_bytes == 0 || second_bytes == 0) {
        return;
    }
    const auto first_address = reinterpret_cast<std::uintptr_t>(first);
    const auto second_address = reinterpret_cast<std::uintptr_t>(second);
    if (first_address < second_address + second_bytes &&
        second_address < first_address + first_bytes) {
        throw std::invalid_argument(
            "Writable buffers must not overlap other buffers or inputs"
        );
    }
}

inline void check_finite_input(const float* values, size_t dim) {
    if (!all_finite(values, dim)) {
        throw std::invalid_argument("Input vectors must contain only finite values");
    }
}

// Rejects a residual whose squared norm overflows before the quantizer's
// float-to-integer conversions. Returns the squared norm.
inline float checked_residual_squared_norm(const float* residual, size_t dim) {
    const float squared_norm = l2norm_sqr<float>(residual, dim);
    if (!std::isfinite(squared_norm)) {
        throw std::overflow_error("Residual squared norm is not representable in float32");
    }
    return squared_norm;
}

inline void check_factors_representable(const v1::Factors& factors) {
    if (!std::isfinite(factors.add) || !std::isfinite(factors.rescale) ||
        !std::isfinite(factors.error)) {
        throw std::overflow_error("Quantization factors are not representable in float32");
    }
}

}  // namespace rabitqlib::quant::validation
