#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rabitqlib/core/api.hpp"
#include "rabitqlib/core/detail/rabitq_impl.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/provisional/quantize.hpp"
#include "validation.hpp"

namespace rabitqlib::quant::provisional {
namespace {

// Widest code a TP element holds, capped at one sign bit plus eight extra bits.
template <typename TP>
constexpr size_t max_total_bits() {
    return std::min<size_t>(9, std::numeric_limits<TP>::digits);
}

void check_dim(size_t dim) {
    if (dim == 0 || dim % 32 != 0) {
        throw std::invalid_argument("dim must be a positive multiple of 32");
    }
    validation::check_dim_range(dim);
}

template <typename TP>
void check_total_bits(size_t total_bits) {
    if (total_bits < 1 || total_bits > max_total_bits<TP>()) {
        throw std::invalid_argument(
            max_total_bits<TP>() == 8 ? "total_bits must be in [1, 8] for uint8_t codes"
                                      : "total_bits must be in [1, 9] for uint16_t codes"
        );
    }
}

// A residual is required; a centroid is optional and checked only when present.
void check_residual(const float* residual, size_t dim) {
    if (residual == nullptr) {
        throw std::invalid_argument("residual must not be null");
    }
    validation::check_finite_input(residual, dim);
}

template <typename TP>
void combined_code_impl(
    const float* residual, size_t dim, size_t total_bits, TP* code, double t_const
) {
    check_dim(dim);
    check_total_bits<TP>(total_bits);
    check_residual(residual, dim);
    if (code == nullptr) {
        throw std::invalid_argument("code must not be null");
    }
    validation::check_disjoint(code, dim * sizeof(TP), residual, dim * sizeof(float));
    validation::check_t_const(t_const);
    (void)validation::checked_residual_squared_norm(residual, dim);

    // Every check has passed; the quantizer can now fail only on allocation, after
    // which the contents of code are unspecified.
    rabitq_impl::total_bits::combined_code<float, TP>(
        residual, dim, total_bits, code, t_const
    );
}

template <typename TP>
v1::Factors code_factors_impl(
    const float* residual,
    const float* centroid,
    size_t dim,
    const TP* code,
    size_t total_bits,
    MetricType metric
) {
    check_dim(dim);
    check_total_bits<TP>(total_bits);
    validate_metric_type(metric);
    check_residual(residual, dim);
    if (centroid != nullptr) {
        validation::check_finite_input(centroid, dim);
    }
    if (code == nullptr) {
        throw std::invalid_argument("code must not be null");
    }
    (void)validation::checked_residual_squared_norm(residual, dim);

    // The detail function takes int codes and their offset-binary constant.
    const auto maximum = static_cast<int>((1U << total_bits) - 1);
    const float cb = -static_cast<float>(maximum) / 2.0F;
    std::vector<int> widened(dim);
    for (size_t i = 0; i < dim; ++i) {
        widened[i] = static_cast<int>(code[i]);
        if (widened[i] > maximum) {
            throw std::invalid_argument("Unpacked code exceeds the configured bit width");
        }
    }
    // The factor derivation rejects a code that does not quantize this residual (its
    // own code or a prefix of it), testing the inner product it divides by.

    std::vector<float> zero;
    if (centroid == nullptr) {
        zero.assign(dim, 0.0F);
        centroid = zero.data();
    }
    v1::Factors factors;
    rabitq_impl::xy_bits::code_factors<float>(
        residual,
        centroid,
        dim,
        widened.data(),
        cb,
        factors.add,
        factors.rescale,
        factors.error,
        metric
    );
    validation::check_factors_representable(factors);
    return factors;
}

}  // namespace

void combined_code(
    const float* residual, size_t dim, size_t total_bits, uint8_t* code, double t_const
) {
    combined_code_impl(residual, dim, total_bits, code, t_const);
}

void combined_code(
    const float* residual, size_t dim, size_t total_bits, uint16_t* code, double t_const
) {
    combined_code_impl(residual, dim, total_bits, code, t_const);
}

v1::Factors code_factors(
    const float* residual,
    const float* centroid,
    size_t dim,
    const uint8_t* code,
    size_t total_bits,
    MetricType metric
) {
    return code_factors_impl(residual, centroid, dim, code, total_bits, metric);
}

v1::Factors code_factors(
    const float* residual,
    const float* centroid,
    size_t dim,
    const uint16_t* code,
    size_t total_bits,
    MetricType metric
) {
    return code_factors_impl(residual, centroid, dim, code, total_bits, metric);
}

}  // namespace rabitqlib::quant::provisional
