#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "rabitqlib/core/data_layout.hpp"
#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/core/detail/quantized_level.hpp"
#include "rabitqlib/core/detail/scratch.hpp"
#include "rabitqlib/core/detail/simd/quantization_dispatch.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/pack_excode.hpp"
#include "rabitqlib/core/types.hpp"
#include "rabitqlib/utils/bitops.hpp"
#include "rabitqlib/utils/openmp.hpp"  // IWYU pragma: keep (configuration macros)
#include "rabitqlib/utils/space.hpp"

#if RABITQ_USE_OPENMP
#include <exception>
#endif

namespace rabitqlib::quant::rabitq_impl {

inline constexpr float kConstEpsilon = 1.9;

// Arithmetic only: callers own residual/centered storage and choose rescale arithmetic.
template <typename T>
struct CodeFactorTerms {
    T add;
    T error;
    T residual_norm;
    T residual_code_ip;
};

template <typename T>
inline CodeFactorTerms<T> code_factor_terms(
    const T* residual,
    const T* centroid,
    const T* centered,
    size_t dim,
    T l2_sqr,
    MetricType metric_type,
    const char* invalid_metric_message
) {
    // The error factor divides by dim - 1; the stable API and the indexes already
    // enforce this, the advanced encoders reach here directly.
    if (dim < 2) {
        throw std::invalid_argument("RaBitQ factors require dim >= 2");
    }
    // A zero residual needs no centered code or inner-product divisions.
    if (l2_sqr == 0) {
        if (metric_type == METRIC_L2) {
            return {0, 0, 0, 0};
        }
        if (metric_type == METRIC_IP) {
            return {1, 0, 0, 0};
        }
        throw std::invalid_argument(invalid_metric_message);
    }
    const T l2_norm = std::sqrt(l2_sqr);
    const T ip_resi_xucb = dot_product<T>(residual, centered, dim);
    const T ip_cent_xucb = dot_product<T>(centroid, centered, dim);
    assert(ip_resi_xucb > 0);

    // Cauchy-Schwarz makes this nonnegative; collinear vectors can round below zero.
    const T normalized_error =
        (((l2_sqr * l2norm_sqr<T>(centered, dim)) / (ip_resi_xucb * ip_resi_xucb)) - 1) /
        (dim - 1);
    const T tmp_error =
        l2_norm * kConstEpsilon * std::sqrt(std::max(normalized_error, T{0}));

    // Metric-specific derivation: docs/docs/rabitq/estimator.md.
    if (metric_type == METRIC_L2) {
        return {
            l2_sqr + (2 * l2_sqr * ip_cent_xucb / ip_resi_xucb),
            2 * tmp_error,
            l2_norm,
            ip_resi_xucb};
    }
    if (metric_type == METRIC_IP) {
        return {
            1 - dot_product<T>(residual, centroid, dim) +
                (l2_sqr * ip_cent_xucb / ip_resi_xucb),
            1 * tmp_error,
            l2_norm,
            ip_resi_xucb};
    }
    throw std::invalid_argument(invalid_metric_message);
}

// Rescale factor for a code centered at its own width, given code_factor_terms() of
// that centered code. Normalized ex-bit codes use their own rescale expression.
template <typename T>
inline T direct_rescale(T l2_sqr, T residual_code_ip, MetricType metric_type) {
    return l2_sqr == 0                ? T{0}
           : metric_type == METRIC_L2 ? -2 * l2_sqr / residual_code_ip
                                      : -l2_sqr / residual_code_ip;
}

// Inverse of pack_binary_to_bytes<uint64_t>: write each coordinate's sign bit (0 or
// 1) to signs. Full 64-coordinate words come first, followed by a final partial word
// of any multiple of 8 coordinates.
inline void unpack_sign_bits(const uint8_t* packed, size_t dim, uint16_t* signs) {
    for (size_t start = 0; start < dim; start += 64) {
        const size_t width = std::min(size_t{64}, dim - start);
        const uint64_t word = bitops::load_word(packed + start / 8, width / 8);
        for (size_t i = 0; i < width; ++i) {
            signs[start + i] = static_cast<uint16_t>((word >> (width - 1 - i)) & 1U);
        }
    }
}

namespace one_bit {

/**
 * Write the sign code and estimation factors for data - centroid: 1 for strictly
 * positive coordinates, 0 otherwise (including zero). Inputs, binary_code, residual
 * and centered scratch each contain dim elements.
 * f_add/f_rescale supply the additive/scaling terms for squared L2 or 1 - dot.
 * f_error is query-independent: multiplying it by the query residual norm gives
 * a probabilistic error margin, not a deterministic bound for every query.
 * See docs/docs/rabitq/estimator.md, "Estimator of Euclidean Distance" and
 * "Estimator of Inner Product", for the factor derivation.
 */
template <typename T>
inline void one_bit_code_with_factor_into(
    const T* data,
    const T* centroid,
    size_t dim,
    int* binary_code,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type,
    T* residual,
    T* centered
) {
    if (std::is_same_v<T, float> && dim <= 128) {
        // Fuse short float vectors; zero residuals retain sign 0.
        for (size_t i = 0; i < dim; ++i) {
            const T value = data[i] - centroid[i];
            residual[i] = value;
            binary_code[i] = static_cast<int>(value > 0);
            centered[i] = value > 0 ? T{0.5} : T{-0.5};
        }
    } else {
        // Wider vectors benefit from Eigen's packet width rather than the
        // compiler's narrower vectorization of the fused multi-output loop.
        RowMajorArrayMap<T> residual_arr(residual, 1, static_cast<Eigen::Index>(dim));
        RowMajorArrayMap<T> xu_cb(centered, 1, static_cast<Eigen::Index>(dim));
        ConstRowMajorArrayMap<T> data_arr(data, 1, static_cast<Eigen::Index>(dim));
        ConstRowMajorArrayMap<T> cent_arr(centroid, 1, static_cast<Eigen::Index>(dim));
        residual_arr = data_arr - cent_arr;
        RowMajorArrayMap<int> x_u(binary_code, 1, static_cast<Eigen::Index>(dim));
        x_u = (residual_arr > 0).template cast<int>();
        xu_cb = x_u.template cast<T>() + T{-0.5};
    }

    const T l2_sqr = l2norm_sqr<T>(residual, dim);
    const auto factors = code_factor_terms(
        residual,
        centroid,
        centered,
        dim,
        l2_sqr,
        metric_type,
        "Unsupported metric type in quantization"
    );
    f_add = factors.add;
    f_rescale = direct_rescale(l2_sqr, factors.residual_code_ip, metric_type);
    f_error = factors.error;
}

template <typename T>
inline void one_bit_code_with_factor(
    const T* data,
    const T* centroid,
    size_t dim,
    int* binary_code,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type = METRIC_L2
) {
    // Legacy calls retain their existing thread-local allocation/reuse behavior.
    thread_local RowMajorArray<T> residual;
    thread_local RowMajorArray<T> centered;
    // Eigen's resize frees the old allocation before allocating its replacement.
    // Allocate first so failure leaves the thread-local owner safe to reuse/destroy.
    if (residual.size() != static_cast<Eigen::Index>(dim)) {
        RowMajorArray<T> replacement(1, static_cast<Eigen::Index>(dim));
        residual.swap(replacement);
    }
    if (centered.size() != static_cast<Eigen::Index>(dim)) {
        RowMajorArray<T> replacement(1, static_cast<Eigen::Index>(dim));
        centered.swap(replacement);
    }
    one_bit_code_with_factor_into(
        data,
        centroid,
        dim,
        binary_code,
        f_add,
        f_rescale,
        f_error,
        metric_type,
        residual.data(),
        centered.data()
    );
}

/**
 * Pack the sign code into padded_dim / 8 bytes; padded_dim is a positive
 * multiple of 8. Both inputs contain padded_dim elements. The first coordinate
 * uses the highest used bit of each word; full and partial words use native
 * byte order. Zero residual coordinates have sign bit 0.
 * Factors have the same meaning as in one_bit_code_with_factor_into, including
 * the query-independent, probabilistic error factor f_error.
 */
template <typename T, typename TP>
inline void one_bit_compact_code_to_bytes(
    const T* data,
    const T* centroid,
    size_t padded_dim,
    uint8_t* compact_code,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type = METRIC_L2
) {
    thread_local std::vector<int> binary_code;
    binary_code.resize(padded_dim);
    one_bit_code_with_factor(
        data,
        centroid,
        padded_dim,
        binary_code.data(),
        f_add,
        f_rescale,
        f_error,
        metric_type
    );

    pack_binary_to_bytes<TP>(binary_code.data(), compact_code, padded_dim);
}

template <typename T, typename TP>
inline void one_bit_compact_code(
    const T* data,
    const T* centroid,
    size_t padded_dim,
    TP* compact_code,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type = METRIC_L2
) {
    one_bit_compact_code_to_bytes<T, TP>(
        data,
        centroid,
        padded_dim,
        reinterpret_cast<uint8_t*>(compact_code),
        f_add,
        f_rescale,
        f_error,
        metric_type
    );
}

// Requires a positive padded_dim divisible by 8.
template <
    typename T,
    typename TP,
    bool Parallel = false,
    typename FAdd,
    typename FRescale,
    typename FError>
inline void one_bit_compact_codes(
    const T* data,
    const T* centroid,
    size_t num_points,
    size_t padded_dim,
    TP* compact_code,
    FAdd f_add,
    FRescale f_rescale,
    FError f_error,
    MetricType metric_type = METRIC_L2
) {
    const auto encode_one = [&](size_t i) {
        T add;
        T rescale;
        T error;
        one_bit_compact_code_to_bytes<T, TP>(
            data + (padded_dim * i),
            centroid,
            padded_dim,
            reinterpret_cast<uint8_t*>(compact_code) + (padded_dim / 8 * i),
            add,
            rescale,
            error,
            metric_type
        );
        f_add[i] = add;
        f_rescale[i] = rescale;
        f_error[i] = error;
    };
#if RABITQ_USE_OPENMP
    if constexpr (Parallel) {
        std::exception_ptr failure;
#pragma omp parallel for
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_points);
             ++index) {
            try {
                encode_one(static_cast<size_t>(index));
            } catch (...) {
#pragma omp critical(rabitq_compact_codes_error)
                {
                    if (!failure) {
                        failure = std::current_exception();
                    }
                }
            }
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
    } else
#endif
    {
        // No serialized OpenMP region: an outer caller can catch scratch failures.
        for (size_t i = 0; i < num_points; ++i) {
            encode_one(i);
        }
    }
}

}  // namespace one_bit

namespace ex_bits {
// Lower endpoint of the scale search as a fraction of t_end, indexed by ex_bits.
// Internal linkage lets the encoder address the table directly instead of through the GOT.
constexpr std::array<float, 9> kTightStart = {
    0,
    0.15,
    0.20,
    0.52,
    0.59,
    0.71,
    0.75,
    0.77,
    0.81,
};

using ::rabitqlib::quant::detail::quantized_level_at_scale;

template <typename T, class Heap, class Codes>
inline double event_sweep(
    const T* o_abs,
    size_t dim,
    int max_code,
    double t_start,
    double t_end,
    Heap& next_t,
    Codes& cur_o_bar
) {
    using Event = RescaleEvent;
    next_t.reserve(dim);
    cur_o_bar.resize(dim);
    double sqr_denominator = static_cast<double>(dim) * 0.25;
    double numerator = 0;

    // init quantization codes for each coordinate by t_start
    for (size_t i = 0; i < dim; ++i) {
        const double magnitude = static_cast<double>(o_abs[i]);
        const int cur = quantized_level_at_scale(magnitude, t_start, max_code);
        cur_o_bar[i] = cur;
        sqr_denominator += (cur * cur) + cur;
        numerator += (cur + 0.5) * magnitude;
        // Never increment a saturated coordinate; skip zero coordinates.
        if (magnitude > 0 && cur < max_code) {
            const double next = static_cast<double>(cur + 1) / magnitude;
            if (next < t_end) {
                next_t.emplace_back(next, i);
            }
        }
    }
    std::make_heap(next_t.begin(), next_t.end(), std::greater<Event>{});

    // The initial state may already be the best state inside the interval.
    double max_ip = numerator / std::sqrt(sqr_denominator);
    double best_t = t_start;

    while (!next_t.empty()) {
        const double cur_t = next_t.front().first;
        // All coordinates crossing the same threshold change together.
        do {
            const size_t i = next_t.front().second;
            ++cur_o_bar[i];
            // With the new code k, q changes from k-0.5 to k+0.5: delta S = 2k.
            sqr_denominator += 2.0 * cur_o_bar[i];
            const double magnitude = static_cast<double>(o_abs[i]);
            numerator += magnitude;

            Event next{t_end, i};
            if (cur_o_bar[i] < max_code) {
                next.first = static_cast<double>(cur_o_bar[i] + 1) / magnitude;
            }
            if (next.first >= t_end) {
                // No further event for this coordinate: fill the root's hole
                // with the last heap entry, then restore the heap below.
                next = next_t.back();
                next_t.pop_back();
            }
            if (!next_t.empty()) {
                // Replace the consumed root event with `next`. Its threshold
                // cannot precede the old minimum, so one sift-down suffices;
                // priority_queue pop() followed by push() would repair twice.
                size_t parent = 0;
                size_t child = 1;
                while (child < next_t.size()) {
                    if (child + 1 < next_t.size() && next_t[child + 1] < next_t[child]) {
                        ++child;
                    }
                    if (!(next_t[child] < next)) {
                        break;
                    }
                    next_t[parent] = next_t[child];
                    parent = child;
                    child = 2 * parent + 1;
                }
                next_t[parent] = next;
            }
        } while (!next_t.empty() && next_t.front().first == cur_t);

        const double cur_ip = numerator / std::sqrt(sqr_denominator);
        if (cur_ip > max_ip) {
            max_ip = cur_ip;
            best_t = cur_t;
        }
    }
    return best_t;
}

// Input is the library's finite, nonnegative normalized magnitude vector.
template <typename T>
inline double best_rescale_factor_with_scratch(
    const T* o_abs, size_t dim, size_t ex_bits, Scratch<T>* scratch
) {
    if (ex_bits >= kTightStart.size()) {
        throw std::invalid_argument("ex_bits exceeds the supported search table");
    }
    if (dim == 0) {
        return 0;
    }
    const double max_o = static_cast<double>(*std::max_element(o_abs, o_abs + dim));
    if (max_o == 0) {
        return 0;
    }

    const int max_code = (1 << ex_bits) - 1;
    const double t_end = static_cast<double>(max_code + 10) / max_o;
    const double t_start = t_end * kTightStart[ex_bits];

    // Both paths maximize cosine similarity between the input and quantized
    // directions over the configured scale interval. Here a_i = o_abs[i] is a
    // rotated, normalized residual magnitude, and q_i = code_i + 0.5.
    // With ||a|| = 1, cosine(a, q) = dot(a, q / ||q||) = N / sqrt(S), where
    // N = sum(a_i * q_i) and S = sum(q_i * q_i).
    // SIMD evaluates whole vectors at selected scales and prunes intervals;
    // its search and bounds are in src/simd/rescale_search.hpp.
    if constexpr (std::is_same_v<T, float>) {
        const double t =
            scratch == nullptr
                ? simd::best_rescale_factor(o_abs, dim, max_code, t_start, t_end)
                : simd::best_rescale_factor_with_workspace(
                      o_abs,
                      dim,
                      max_code,
                      t_start,
                      t_end,
                      scratch->magnitudes,
                      scratch->reciprocals
                  );
        if (t >= 0) {
            return t;
        }
    }

    // Scalar event sweep, also used if SIMD is unavailable or cannot certify a
    // winner within its work limit. Each heap entry holds a coordinate's next
    // code-changing scale (code_i + 1) / a_i. Visiting only these events skips
    // ranges with constant codes and updates N and S in O(1) per coordinate.
    // Heap traversal is sequential and branch-dependent, so it offers little
    // SIMD parallelism; the SIMD path instead prunes ranges of candidate codes.
    using Event = RescaleEvent;
    if (scratch != nullptr) {
        ScratchAllocator allocator(scratch->resource, dim);
        auto [next_t, cur_o_bar] = event_sweep_buffers(allocator);
        return event_sweep(o_abs, dim, max_code, t_start, t_end, next_t, cur_o_bar);
    }
    std::vector<Event> next_t;
    std::vector<int> cur_o_bar;
    return event_sweep(o_abs, dim, max_code, t_start, t_end, next_t, cur_o_bar);
}

template <typename T>
inline double best_rescale_factor(const T* o_abs, size_t dim, size_t ex_bits) {
    return best_rescale_factor_with_scratch(
        o_abs, dim, ex_bits, static_cast<Scratch<T>*>(nullptr)
    );
}

template <typename T, typename TP>
inline T quantize_ex_with_scratch(
    const T* o_abs, TP* code, size_t dim, size_t ex_bits, Scratch<T>* scratch
) {
    const double t = best_rescale_factor_with_scratch<T>(o_abs, dim, ex_bits, scratch);
    const int max_code = (1 << ex_bits) - 1;
    double ipnorm = 0;
    for (size_t i = 0; i < dim; ++i) {
        const double magnitude = static_cast<double>(o_abs[i]);
        // Emit precisely the legal state evaluated by the search.
        const int value = quantized_level_at_scale(magnitude, t, max_code);
        code[i] = static_cast<TP>(value);
        ipnorm += (value + 0.5) * magnitude;
    }
    if (ipnorm == 0) {
        return static_cast<T>(1);
    }
    const T ipnorm_inv = static_cast<T>(1.0 / ipnorm);
    return std::isnormal(ipnorm_inv) ? ipnorm_inv : static_cast<T>(1);
}

template <typename T, typename TP>
inline T quantize_ex(const T* o_abs, TP* code, size_t dim, size_t ex_bits) {
    return quantize_ex_with_scratch(
        o_abs, code, dim, ex_bits, static_cast<Scratch<T>*>(nullptr)
    );
}

// For given dim and ex_bits, use random vectors to get the const rescale factor.
// faster_config reads these values from the frozen table (fast_scale_table.hpp);
// this computation generates that table and serves dimensions outside it.
inline double const_scaling_factor(size_t dim, size_t ex_bits) {
    constexpr Eigen::Index kConstNum = 100;

    RowMajorArray<double> rand = random_gaussian_matrix<double>(kConstNum, dim, 42);

    rand = rand.rowwise().normalized().abs();

    double sum = 0;
    for (Eigen::Index j = 0; j < kConstNum; ++j) {
        sum += best_rescale_factor(&rand(j, 0), dim, ex_bits);
    }

    double t_const = sum / kConstNum;

    return t_const;
}

template <typename T, typename TP>
inline T faster_quantize_ex(
    const T* o_abs, TP* code, size_t dim, size_t ex_bits, double t_const
) {
    constexpr double kEps = 1e-5;
    const int max_code = (1 << ex_bits) - 1;
    double ipnorm = 0;
    for (size_t i = 0; i < dim; ++i) {
        const double magnitude = static_cast<double>(o_abs[i]);
        const int value =
            std::min(static_cast<int>((t_const * magnitude) + kEps), max_code);
        code[i] = static_cast<TP>(value);
        // Unnormalized inner product: ip * norm.
        ipnorm += (value + 0.5) * magnitude;
    }
    const T ipnorm_inv = static_cast<T>(1 / ipnorm);
    return std::isnormal(ipnorm_inv) ? ipnorm_inv : static_cast<T>(1);
}

template <typename T, typename TP>
inline T ex_bits_code_with_norm(
    const T* residual,
    T residual_norm,
    size_t dim,
    size_t ex_bits,
    TP* ex_code,
    double t_const,
    Scratch<T>* scratch
) {
    ConstRowMajorArrayMap<T> residual_arr(residual, 1, static_cast<Eigen::Index>(dim));
    T ipnorm_inv = 1;
    if (residual_norm == 0) {
        // Match code_with_factor_from_residual(): no magnitude bits, so skip the revert.
        std::fill(ex_code, ex_code + dim, static_cast<TP>(0));
        return ipnorm_inv;
    } else {
        RowMajorArray<T> owned_absolute;
        auto abs_res = scratch_array(
            scratch == nullptr ? nullptr : scratch->absolute, dim, owned_absolute
        );
        abs_res = residual_arr.abs() / residual_norm;
        if (t_const > 0) {
            ipnorm_inv = faster_quantize_ex(abs_res.data(), ex_code, dim, ex_bits, t_const);
        } else {
            ipnorm_inv =
                quantize_ex_with_scratch(abs_res.data(), ex_code, dim, ex_bits, scratch);
        }
    }

    // Revert codes for negative or zero dims. A select instead of a branch avoids a
    // misprediction per random-sign coordinate and lets baseline builds vectorize.
    int32_t mask = (1 << ex_bits) - 1;
    for (size_t j = 0; j < dim; ++j) {
        const TP code = ex_code[j];
        const auto reverted = static_cast<TP>((~code) & mask);
        ex_code[j] = residual_arr.data()[j] <= 0 ? reverted : code;
    }

    return ipnorm_inv;
}

template <typename T, typename TP>
inline T ex_bits_code(
    const T* residual, size_t dim, size_t ex_bits, TP* ex_code, double t_const = -1
) {
    return ex_bits_code_with_norm(
        residual,
        std::sqrt(l2norm_sqr<T>(residual, dim)),
        dim,
        ex_bits,
        ex_code,
        t_const,
        static_cast<Scratch<T>*>(nullptr)
    );
}

// l2_sqr belongs to residual. FullCode returns the combined code used for factors.
template <bool FullCode, typename T, typename TP>
inline void code_with_factor_from_residual(
    const T* residual,
    T l2_sqr,
    const T* centroid,
    size_t dim,
    size_t ex_bits,
    TP* ex_code,
    T& f_add_ex,
    T& f_rescale_ex,
    T& f_error_ex,
    MetricType metric_type,
    double t_const,
    Scratch<T>* scratch
) {
    if (l2_sqr == 0) {
        std::fill(ex_code, ex_code + dim, static_cast<TP>(0));
        const auto factors = code_factor_terms<T>(
            residual,
            centroid,
            nullptr,
            dim,
            l2_sqr,
            metric_type,
            "Unsupported metric for ex_bits_code()"
        );
        f_add_ex = factors.add;
        f_rescale_ex = 0;
        f_error_ex = factors.error;
        if constexpr (FullCode) {
            // A tiny nonzero residual can have a squared norm of zero.
            for (size_t i = 0; i < dim; ++i) {
                ex_code[i] = static_cast<TP>(residual[i] > 0) << ex_bits;
            }
        }
        return;
    }

    T ipnorm_inv = ex_bits_code_with_norm<T, TP>(
        residual, std::sqrt(l2_sqr), dim, ex_bits, ex_code, t_const, scratch
    );
    RowMajorArray<int> owned_total;
    auto total_code =
        scratch_array(scratch == nullptr ? nullptr : scratch->total, dim, owned_total);
    total_code = RowMajorArrayMap<TP>(ex_code, 1, static_cast<Eigen::Index>(dim))
                     .template cast<int>();
    for (size_t i = 0; i < dim; ++i) {
        total_code(0, static_cast<Eigen::Index>(i)) += static_cast<int>(residual[i] > 0)
                                                       << ex_bits;
    }

    // Full-code factors use the metric-specific derivation in
    // docs/docs/rabitq/estimator.md, with xu_cb centered at this code width.
    float cb = -(static_cast<float>(1 << ex_bits) - 0.5F);
    RowMajorArray<T> owned_centered;
    auto xu_cb = scratch_array(
        scratch == nullptr ? nullptr : scratch->centered, dim, owned_centered
    );
    xu_cb = total_code.template cast<T>() + cb;

    const auto factors = code_factor_terms(
        residual,
        centroid,
        xu_cb.data(),
        dim,
        l2_sqr,
        metric_type,
        "Unsupported metric for ex_bits_code()"
    );
    f_add_ex = factors.add;
    // Retain the normalized-code rescale expression and its rounding order.
    f_rescale_ex = metric_type == METRIC_L2 ? ipnorm_inv * -2 * factors.residual_norm
                                            : ipnorm_inv * -factors.residual_norm;
    f_error_ex = factors.error;
    if constexpr (FullCode) {
        RowMajorArrayMap<TP>(ex_code, 1, static_cast<Eigen::Index>(dim)) =
            total_code.template cast<TP>();
    }
}

template <typename T, typename TP>
inline void ex_bits_code_with_factor(
    const T* data,
    const T* centroid,
    size_t dim,
    size_t ex_bits,
    TP* ex_code,
    T& f_add_ex,
    T& f_rescale_ex,
    T& f_error_ex,
    MetricType metric_type = METRIC_L2,
    double t_const = -1
) {
    RowMajorArray<T> residual =
        ConstRowMajorArrayMap<T>(data, 1, static_cast<Eigen::Index>(dim)) -
        ConstRowMajorArrayMap<T>(centroid, 1, static_cast<Eigen::Index>(dim));
    code_with_factor_from_residual<false>(
        residual.data(),
        l2norm_sqr<T>(residual.data(), dim),
        centroid,
        dim,
        ex_bits,
        ex_code,
        f_add_ex,
        f_rescale_ex,
        f_error_ex,
        metric_type,
        t_const,
        static_cast<Scratch<T>*>(nullptr)
    );
}

template <typename T, typename TP>
inline void ex_bits_compact_code(
    const T* data,
    const T* centroid,
    size_t padded_dim,
    size_t ex_bits,
    TP* compact_code,
    T& f_add_ex,
    T& f_rescale_ex,
    T& f_error_ex,
    MetricType metric_type = METRIC_L2,
    double t_const = -1
) {
    std::vector<uint8_t> ex_code(padded_dim);

    ex_bits_code_with_factor(
        data,
        centroid,
        padded_dim,
        ex_bits,
        ex_code.data(),
        f_add_ex,
        f_rescale_ex,
        f_error_ex,
        metric_type,
        t_const
    );

    packing_rabitqplus_code(ex_code.data(), compact_code, padded_dim, ex_bits);
}
}  // namespace ex_bits

namespace total_bits {

template <typename T, typename TP>
inline void combined_code(
    const T* residual, size_t dim, size_t total_bits, TP* total_code, double t_const = -1
) {
    const size_t ex_bits = total_bits - 1;

    ConstRowMajorArrayMap<T> residual_arr(residual, 1, static_cast<Eigen::Index>(dim));
    RowMajorArrayMap<TP> code_arr(total_code, 1, static_cast<Eigen::Index>(dim));

    if (ex_bits > 0) {
        ex_bits::ex_bits_code<T, TP>(residual, dim, ex_bits, total_code, t_const);
    } else {
        code_arr.setZero();
    }

    code_arr += (residual_arr > 0).template cast<TP>() * static_cast<TP>(1 << ex_bits);
}

// The per-ISA HNSW search sources include this header, so these entry points keep
// internal linkage (see "Implementation linkage" in docs/development.md).
template <typename T, typename TP>
static inline void rabitq_scalar_impl(
    const T* data,
    const T* centroid,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& delta,
    T& vl,
    double t_const = -1,
    ScalarQuantizerType scalar_quantizer_type = ScalarQuantizerType::RECONSTRUCTION
) {
    size_t ex_bits = total_bits - 1;

    ConstRowMajorArrayMap<T> data_arr(data, 1, static_cast<Eigen::Index>(dim));
    ConstRowMajorArrayMap<T> cent_arr(centroid, 1, static_cast<Eigen::Index>(dim));
    RowMajorArray<T> residual_arr = data_arr - cent_arr;

    if (l2norm_sqr(residual_arr.data(), dim) == 0) {
        std::fill(total_code, total_code + dim, static_cast<TP>(0));
        delta = 0;
        vl = 0;
        return;
    }

    combined_code<T, TP>(residual_arr.data(), dim, total_bits, total_code, t_const);

    float cb = -(static_cast<float>(1 << ex_bits) - 0.5F);
    RowMajorArrayMap<TP> total_code_arr(total_code, 1, static_cast<Eigen::Index>(dim));
    RowMajorArray<T> u_cb = total_code_arr.template cast<T>() + cb;

    T norm_data = std::sqrt(l2norm_sqr(residual_arr.data(), dim));
    T norm_quan = std::sqrt(l2norm_sqr(u_cb.data(), dim));
    T cos_similarity =
        dot_product<T>(residual_arr.data(), u_cb.data(), dim) / (norm_data * norm_quan);

    if (scalar_quantizer_type == ScalarQuantizerType::RECONSTRUCTION) {
        delta = norm_data / norm_quan * cos_similarity;
    } else if (scalar_quantizer_type == ScalarQuantizerType::UNBIASED_ESTIMATION) {
        delta = norm_data / norm_quan / cos_similarity;
    } else if (scalar_quantizer_type == ScalarQuantizerType::PLAIN) {
        delta = norm_data / norm_quan;
    }

    vl = delta * cb;
}

template <typename T, typename TP>
static inline void rabitq_full_impl_with_scratch(
    const T* data,
    const T* centroid,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type,
    double t_const,
    Scratch<T>* scratch
) {
    // One sign bit plus at most eight magnitude bits (xy_bits::kMaxCombinedBits).
    if (total_bits < 1 || total_bits > 9 ||
        total_bits > static_cast<size_t>(std::numeric_limits<TP>::digits)) {
        throw std::invalid_argument(
            "total_bits must be in [1, 9] and fit the code element type"
        );
    }
    const size_t ex_bits = total_bits - 1;
    if (ex_bits > 0) {
        RowMajorArray<T> owned_residual;
        auto residual = scratch_array(
            scratch == nullptr ? nullptr : scratch->residual, dim, owned_residual
        );
        residual = ConstRowMajorArrayMap<T>(data, 1, static_cast<Eigen::Index>(dim)) -
                   ConstRowMajorArrayMap<T>(centroid, 1, static_cast<Eigen::Index>(dim));
        ex_bits::code_with_factor_from_residual<true>(
            residual.data(),
            l2norm_sqr<T>(residual.data(), dim),
            centroid,
            dim,
            ex_bits,
            total_code,
            f_add,
            f_rescale,
            f_error,
            metric_type,
            t_const,
            scratch
        );
        return;
    }

    std::vector<int> owned_binary;
    int* binary_code = nullptr;
    if (scratch == nullptr) {
        owned_binary.resize(dim);
        binary_code = owned_binary.data();
        one_bit::one_bit_code_with_factor(
            data, centroid, dim, binary_code, f_add, f_rescale, f_error, metric_type
        );
    } else {
        binary_code = scratch->binary;
        one_bit::one_bit_code_with_factor_into(
            data,
            centroid,
            dim,
            binary_code,
            f_add,
            f_rescale,
            f_error,
            metric_type,
            scratch->residual,
            scratch->centered
        );
    }
    for (size_t i = 0; i < dim; ++i) {
        total_code[i] = static_cast<TP>(binary_code[i]);
    }
}

template <typename T, typename TP>
static inline void rabitq_full_impl(
    const T* data,
    const T* centroid,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type = METRIC_L2,
    double t_const = -1
) {
    rabitq_full_impl_with_scratch(
        data,
        centroid,
        dim,
        total_bits,
        total_code,
        f_add,
        f_rescale,
        f_error,
        metric_type,
        t_const,
        static_cast<Scratch<T>*>(nullptr)
    );
}

}  // namespace total_bits

namespace xy_bits {

inline constexpr size_t kMaxCombinedBits = 9;

static_assert(
    ex_bits::kTightStart.size() >= kMaxCombinedBits,
    "kTightStart must cover magnitude widths up to kMaxCombinedBits - 1"
);

inline void validate_bit_size(size_t base_bits, size_t ex_bits) {
    if (base_bits < 1 || base_bits > 8) {
        throw std::invalid_argument("base_bits must be in [1, 8]");
    }
    if (ex_bits > 8) {
        throw std::invalid_argument("ex_bits must be in [0, 8]");
    }
    if (base_bits + ex_bits > kMaxCombinedBits) {
        throw std::invalid_argument("base_bits + ex_bits exceeds the supported limit");
    }
}

/**
 * @brief Derive estimation factors for a residual-compatible integer code.
 *
 * For a nonzero residual, dot(residual, code + cb) must be strictly positive;
 * the code's integer range alone does not guarantee this. A zero residual
 * produces zero rescale/error factors and a metric-specific additive term.
 *
 * Centers the supplied code before applying the shared factor arithmetic.
 * With cb = -0.5 and a sign-bit code, it matches the one-bit factors.
 *
 * @param residual data - centroid
 * @param code integer code, values in [0, 2^bits)
 * @param cb offset-binary constant for that code width, -(2^bits - 1)/2
 */
template <typename T>
inline void code_factors(
    const T* residual,
    const T* centroid,
    size_t dim,
    const int* code,
    float cb,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type = METRIC_L2
) {
    ConstRowMajorArrayMap<int> code_arr(code, 1, static_cast<Eigen::Index>(dim));
    RowMajorArray<T> xu_cb = code_arr.template cast<T>() + cb;

    const T l2_sqr = l2norm_sqr<T>(residual, dim);
    const auto factors = code_factor_terms(
        residual,
        centroid,
        xu_cb.data(),
        dim,
        l2_sqr,
        metric_type,
        "Unsupported metric type in code_factors()"
    );
    f_add = factors.add;
    f_rescale = direct_rescale(l2_sqr, factors.residual_code_ip, metric_type);
    f_error = factors.error;
}

template <typename T>
inline void split_code_with_factor(
    const T* data,
    const T* centroid,
    size_t dim,
    size_t base_bits,
    size_t ex_bits,
    char* base_data,
    char* ex_data,
    MetricType metric_type = METRIC_L2,
    double t_const = -1
) {
    validate_bit_size(base_bits, ex_bits);

    ConstRowMajorArrayMap<T> data_arr(data, 1, static_cast<Eigen::Index>(dim));
    ConstRowMajorArrayMap<T> cent_arr(centroid, 1, static_cast<Eigen::Index>(dim));
    RowMajorArray<T> residual_arr = data_arr - cent_arr;
    const T* residual = residual_arr.data();

    const size_t total_bits = base_bits + ex_bits;
    std::vector<int> total_code(dim);
    total_bits::combined_code<T, int>(
        residual, dim, total_bits, total_code.data(), t_const
    );

    const auto ex_mask = (ex_bits > 0) ? static_cast<int>((1U << ex_bits) - 1) : 0;
    std::vector<int> base_code_int(dim);
    std::vector<uint8_t> ex_raw(ex_bits > 0 ? dim : 0, 0);
    for (size_t i = 0; i < dim; ++i) {
        base_code_int[i] = total_code[i] >> ex_bits;
        if (ex_bits > 0) {
            ex_raw[i] = static_cast<uint8_t>(total_code[i] & ex_mask);
        }
    }

    std::vector<uint8_t> base_raw(dim);
    for (size_t i = 0; i < dim; ++i) {
        base_raw[i] = static_cast<uint8_t>(base_code_int[i]);
    }

    // Base layer first: its factors describe the filter code on its own.
    BaseDataMap<T> base_map(base_data, dim, base_bits);
    T f_add_base;
    T f_rescale_base;
    T f_error_base;
    code_factors<T>(
        residual,
        centroid,
        dim,
        base_code_int.data(),
        -(static_cast<float>((1U << base_bits) - 1) / 2.0F),
        f_add_base,
        f_rescale_base,
        f_error_base,
        metric_type
    );
    base_map.f_add() = f_add_base;
    base_map.f_rescale() = f_rescale_base;
    base_map.f_error() = f_error_base;
    ex_bits::packing_rabitqplus_code(base_raw.data(), base_map.base_code(), dim, base_bits);

    if (ex_bits == 0) {
        return;
    }

    // Refine layer: the same derivation over the combined code.
    ExDataMap<T> ex_map(ex_data, dim, ex_bits);
    T f_add_ex;
    T f_rescale_ex;
    T f_error_ex = 0;
    code_factors<T>(
        residual,
        centroid,
        dim,
        total_code.data(),
        -(static_cast<float>((1U << total_bits) - 1) / 2.0F),
        f_add_ex,
        f_rescale_ex,
        f_error_ex,
        metric_type
    );
    ex_map.f_add_ex() = f_add_ex;
    ex_map.f_rescale_ex() = f_rescale_ex;

    ex_bits::packing_rabitqplus_code(ex_raw.data(), ex_map.ex_code(), dim, ex_bits);
}
}  // namespace xy_bits
}  // namespace rabitqlib::quant::rabitq_impl
