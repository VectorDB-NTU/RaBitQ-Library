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
#include "rabitqlib/utils/input_checks.hpp"
#include "rabitqlib/utils/space.hpp"

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
    if (dim > static_cast<size_t>(std::numeric_limits<long>::max())) {
        throw std::length_error("dim exceeds the supported index range");
    }
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

void check_vector(const float* values, size_t dim, const char* null_message) {
    if (values == nullptr) {
        throw std::invalid_argument(null_message);
    }
    if (!all_finite(values, dim)) {
        throw std::invalid_argument("Input vectors must contain only finite values");
    }
}

// Rejects a residual whose squared norm overflows before the quantizer's
// float-to-integer conversions, as quant::v1::encode does. Returns the squared norm.
float checked_squared_norm(const float* residual, size_t dim) {
    const float squared_norm = l2norm_sqr<float>(residual, dim);
    if (!std::isfinite(squared_norm)) {
        throw std::overflow_error("Residual squared norm is not representable in float32");
    }
    return squared_norm;
}

bool overlaps(
    const void* first, size_t first_bytes, const void* second, size_t second_bytes
) {
    const auto first_address = reinterpret_cast<std::uintptr_t>(first);
    const auto second_address = reinterpret_cast<std::uintptr_t>(second);
    return first_address < second_address + second_bytes &&
           second_address < first_address + first_bytes;
}

template <typename TP>
void combined_code_impl(
    const float* residual, size_t dim, size_t total_bits, TP* code, double t_const
) {
    check_dim(dim);
    check_total_bits<TP>(total_bits);
    check_vector(residual, dim, "residual must not be null");
    if (code == nullptr) {
        throw std::invalid_argument("code must not be null");
    }
    if (overlaps(code, dim * sizeof(TP), residual, dim * sizeof(float))) {
        throw std::invalid_argument(
            "Writable buffers must not overlap other buffers or inputs"
        );
    }
    if (!std::isfinite(t_const) || t_const > std::numeric_limits<int>::max() / 2.0) {
        throw std::invalid_argument("t_const must be finite and at most INT_MAX / 2");
    }
    (void)checked_squared_norm(residual, dim);

    // Quantize into owned storage so a failure leaves the caller's codes unchanged.
    std::vector<TP> result(dim);
    rabitq_impl::total_bits::combined_code<float, TP>(
        residual, dim, total_bits, result.data(), t_const
    );
    std::copy(result.begin(), result.end(), code);
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
    check_vector(residual, dim, "residual must not be null");
    if (centroid != nullptr) {
        check_vector(centroid, dim, "centroid must not be null");
    }
    if (code == nullptr) {
        throw std::invalid_argument("code must not be null");
    }
    const float squared_norm = checked_squared_norm(residual, dim);

    // The detail function takes int codes and their offset-binary constant.
    const auto maximum = static_cast<int>((1U << total_bits) - 1);
    const float cb = -static_cast<float>(maximum) / 2.0F;
    std::vector<int> widened(dim);
    std::vector<float> centered(dim);
    for (size_t i = 0; i < dim; ++i) {
        widened[i] = static_cast<int>(code[i]);
        if (widened[i] > maximum) {
            throw std::invalid_argument("Unpacked code exceeds the configured bit width");
        }
        centered[i] = static_cast<float>(widened[i]) + cb;
    }
    // The factor derivation divides by this inner product; only a code that quantizes
    // this residual (its own code or a prefix of it) makes it positive.
    if (squared_norm != 0 && !(dot_product<float>(residual, centered.data(), dim) > 0)) {
        throw std::invalid_argument(
            "code does not quantize residual: <residual, centered code> must be positive"
        );
    }

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
    if (!std::isfinite(factors.add) || !std::isfinite(factors.rescale) ||
        !std::isfinite(factors.error)) {
        throw std::overflow_error("Quantization factors are not representable in float32");
    }
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
