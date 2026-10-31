#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "rabitqlib/core/detail/quantized_level.hpp"

namespace rabitqlib::simd::detail {

using ::rabitqlib::quant::detail::quantized_level_at_scale;

struct RescaleScratch {
    std::vector<double> magnitudes;
    std::vector<double> reciprocals;
};

// Keep TLS initialization/destruction in generic code shared by both backends.
RescaleScratch& get_thread_local_rescale_scratch(size_t dim);

struct RescaleSearchState {
    double numerator;     // N = sum(a_i * q_i), where q_i = code_i + 0.5.
    double squared_norm;  // S = sum(q_i * q_i); the squared score is N^2 / S.
    // This code vector is unchanged throughout [first, next), clipped to the
    // configured search range. These boundaries let us skip constant-code spans.
    // For the current codes c_i and maximum code M:
    //   first = max(t_start, max_{c_i > 0} c_i / a_i),
    //   next  = min(t_end, min_{a_i > 0, c_i < M} (c_i + 1) / a_i).
    // Zero coordinates never change; saturated coordinates have no next event.
    double first;
    double next;
};

// Keep executable helpers local to each translation unit. This header is
// compiled with different ISA flags; merging external inline/template symbols
// could make the AVX2 backend call an AVX512-compiled helper.

// Recover exact event scales at the boundary coordinates a vector evaluator selected
// (size_t max for none): it compares reciprocal products, while the heap sweep
// divides. Then score coordinates [tail, dim) with the scalar definitions.
static inline RescaleSearchState finish_scale_state(
    RescaleSearchState s,
    const double* magnitudes,
    size_t first_index,
    size_t next_index,
    size_t tail,
    size_t dim,
    int max_code,
    double scale
) {
    constexpr size_t kNone = std::numeric_limits<size_t>::max();
    if (first_index != kNone)
        s.first = std::max(
            s.first,
            static_cast<double>(
                quantized_level_at_scale(magnitudes[first_index], scale, max_code)
            ) / magnitudes[first_index]
        );
    if (next_index != kNone)
        s.next = std::min(
            s.next,
            (quantized_level_at_scale(magnitudes[next_index], scale, max_code) + 1.0) /
                magnitudes[next_index]
        );
    for (size_t i = tail; i < dim; ++i) {
        int c = quantized_level_at_scale(magnitudes[i], scale, max_code);
        double q = c + .5;
        s.numerator += magnitudes[i] * q;
        s.squared_norm += q * q;
        if (c > 0)
            s.first = std::max(s.first, static_cast<double>(c) / magnitudes[i]);
        if (c < max_code && magnitudes[i] > 0)
            s.next = std::min(s.next, (c + 1.0) / magnitudes[i]);
    }
    return s;
}

// Finite IEEE-754 binary64 neighbors for outward-rounded bounds.
// Adjusting the representation leaves the floating-point rounding mode unchanged.
static inline double next_double_up(double value) {
    static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8);
    // Both signed zeros step to the smallest positive subnormal.
    if (value == 0)
        return std::numeric_limits<double>::denorm_min();
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Negative values have reversed bit ordering.
    bits += value > 0 ? uint64_t{1} : static_cast<uint64_t>(-1);
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline double next_double_down(double value) {
    // Both signed zeros step to the smallest-magnitude negative subnormal.
    if (value == 0)
        return -std::numeric_limits<double>::denorm_min();
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += value > 0 ? static_cast<uint64_t>(-1) : uint64_t{1};
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Its default member initializers give it a constructor, so keep the type
// TU-local as well.
namespace {
struct BoundedRescaleState {
    RescaleSearchState state{};
    // Enclose the true dot product despite SIMD summation rounding. S is exact
    // under the dimension guard, so these also give bounds on the true N^2 / S.
    double numerator_lower = 0;
    double numerator_upper = 0;
    double score_lower = 0;
    double score_upper = 0;
};
}  // namespace

// Upper bound on squared cosine similarity for every code between two evaluated
// states. Here a_i = input[i] is a rotated, normalized residual magnitude, and
// q_i = code_i + 0.5. With ||a|| = 1, cosine(a, q)^2 = N^2 / S, where
// N = sum(a_i * q_i) and S = sum(q_i * q_i). All magnitudes are nonnegative,
// so maximizing squared cosine also maximizes cosine, the heap sweep's objective.
// This bounds the whole interval; comparing endpoint scores alone cannot
// exclude a better interior code because the score need not be monotonic.
static inline double interval_squared_cosine_upper_bound(
    const BoundedRescaleState& left, const BoundedRescaleState& right
) {
    const double endpoints = std::max(left.score_upper, right.score_upper);
    // Adjacent plateaus have no intermediate codes. If the numerator upper
    // bounds are reversed, every interior numerator is already <= the left
    // upper bound, while its squared norm is >= the left norm; endpoints suffice.
    if (left.state.next == right.state.first ||
        left.numerator_upper >= right.numerator_upper)
        return endpoints;

    // Derive a bound without enumerating the events inside this interval:
    //
    // 1. An event at tau = k/a_i changes code_i from k-1 to k. It adds a_i
    //    to N and (k+0.5)^2 - (k-0.5)^2 = 2k to S. Thus delta S = 2*tau*delta N.
    // 2. All intervening event scales lie between l = left.next and
    //    u = right.first. Therefore 2l*delta N <= delta S <= 2u*delta N.
    // 3. For any intermediate numerator x, summing events forward from the
    //    left endpoint gives S(x) >= S0 + 2l*(x-N0). Summing the remaining
    //    events to the right endpoint gives S(x) >= S1 - 2u*(N1-x).
    //
    // N0/N1 are the true endpoint numerators; S0/S1 are their exact squared
    // norms. Replacing N0/N1 by their upper bounds U0/U1 only lowers these two
    // lines, so the denominator bound stays conservative:
    //   S(x) >= max(S0 + 2l*(x-U0), S1 - 2u*(U1-x)).
    // Hence x^2 divided by this maximum is an upper bound on the score.
    // Values x <= U0 are covered by left.score_upper; only [U0,U1] remains.
    // lower_t/upper_t and all subsequent arithmetic are rounded outward.
    const double lower_t = next_double_down(left.state.next);
    const double upper_t = next_double_up(right.state.first);
    const double delta_n_lower =
        next_double_down(right.numerator_upper - left.numerator_upper);
    const double delta_n_upper =
        next_double_up(right.numerator_upper - left.numerator_upper);
    const double delta_d_lower =
        next_double_down(right.state.squared_norm - left.state.squared_norm);
    const double delta_d_upper =
        next_double_up(right.state.squared_norm - left.state.squared_norm);

    // Each active denominator line has the form A+B*x. Where it is positive,
    // f(x) = x^2/(A+B*x) is convex: f''(x) = 2*A^2/(A+B*x)^3 >= 0.
    // Its maximum on a section is at a section endpoint. We therefore need
    // only the two endpoint scores and the intersection of the two lines.
    // Solving for the intersection's offset p from U0 gives
    //   p = (2u*(U1-U0) - (S1-S0)) / (2*(u-l)).
    // delta_d_* above enclose S1-S0; intersection_n/d_* below enclose the
    // numerator and denominator of this expression, not the cosine score.
    const double intersection_n_lower =
        next_double_down(next_double_down(2 * upper_t * delta_n_lower) - delta_d_upper);
    const double intersection_n_upper =
        next_double_up(next_double_up(2 * upper_t * delta_n_upper) - delta_d_lower);
    const double intersection_d_lower = 2 * next_double_down(upper_t - lower_t);
    const double intersection_d_upper = 2 * next_double_up(upper_t - lower_t);
    if (intersection_d_lower <= 0) {
        // The event-scale gap is too small to divide by reliably. Use the
        // looser monotonic bound N <= U1 and S >= S0 instead of pruning unsafely.
        return next_double_up(
            next_double_up(right.numerator_upper * right.numerator_upper) /
            left.state.squared_norm
        );
    }
    // Clamp the enclosed intersection to [0, U1-U0]. An intersection outside
    // that range introduces no interior maximum; the endpoints already cover it.
    const double offset_lower = std::max(
        0.0,
        std::min(
            delta_n_lower,
            next_double_down(std::max(0.0, intersection_n_lower) / intersection_d_upper)
        )
    );
    const double offset_upper = std::max(
        0.0,
        std::min(
            delta_n_upper,
            next_double_up(std::max(0.0, intersection_n_upper) / intersection_d_lower)
        )
    );
    // Both denominator lines increase with x. Use the larger possible offset
    // for the numerator and the smaller one for the denominator, producing an
    // upper score bound even when the intersection cannot be represented exactly.
    // S0 is also a valid lower bound on every intermediate squared norm.
    const double numerator = next_double_up(left.numerator_upper + offset_upper);
    const double left_denominator = next_double_down(
        left.state.squared_norm + next_double_down(2 * lower_t * offset_lower)
    );
    const double right_denominator = next_double_down(
        right.state.squared_norm -
        next_double_up(2 * upper_t * next_double_up(delta_n_upper - offset_lower))
    );
    const double denominator =
        std::max({left.state.squared_norm, left_denominator, right_denominator});
    return std::max(
        endpoints, next_double_up(next_double_up(numerator * numerator) / denominator)
    );
}

// SIMD branch-and-bound over the heap sweep's [t_start, t_end) range.
// Evaluate endpoint plateaus, then split gaps whose bounds can beat the best score.
// Evaluators return legal states and division-based plateau boundaries;
// only numerator accumulation may round.
// Return false on ambiguous scores or work limits: the caller discards the
// tentative result and runs the exact heap sweep, preserving its tie order.
template <typename Evaluator>
static bool try_find_best_scale_by_interval_search(
    const float* input,
    size_t dim,
    int max_code,
    double t_start,
    double t_end,
    Evaluator evaluator,
    double& best_t,
    double* magnitudes = nullptr,
    double* reciprocals = nullptr
) {
    if (dim == 0 || max_code == 0) {
        best_t = t_start;
        return true;
    }
    // S is a sum of quarter integers. This guard makes every addition exact.
    const uint64_t largest_odd = static_cast<uint64_t>(2 * max_code + 1);
    if (dim > (uint64_t{1} << 53) / (largest_odd * largest_odd))
        return false;

    // Promote float inputs exactly once and cache 1/a_i for repeated candidate
    // evaluations. The thread-local buffers are reused across vectors; each
    // call overwrites its active prefix. Zero reciprocals are masked by evaluators.
    if (magnitudes == nullptr) {
        auto& scratch = get_thread_local_rescale_scratch(dim);
        magnitudes = scratch.magnitudes.data();
        reciprocals = scratch.reciprocals.data();
    }
    for (size_t i = 0; i < dim; ++i) {
        magnitudes[i] = static_cast<double>(input[i]);
        reciprocals[i] = input[i] > 0 ? 1.0 / magnitudes[i] : 0;
    }

    // Float magnitudes times half-integer codes are exact in double. Positive
    // summation has relative error gamma_(dim+4), bounded by this delta under
    // the dimension guard above, including the SIMD horizontal reduction.
    const double delta =
        static_cast<double>(dim + 8) * std::numeric_limits<double>::epsilon();
    const double lower_divisor = next_double_up(1 + delta);
    const double upper_divisor = next_double_down(1 - delta);
    // These cap search work, not quantization accuracy: hitting either cap
    // requests the heap fallback instead of returning the best-so-far scale.
    constexpr size_t kEvaluationBudget = 512;
    constexpr size_t kMaxDepth = 64;
    size_t evaluations = 0;
    double best_lower = 0;  // Largest certified score lower bound seen so far.
    BoundedRescaleState best;
    bool have_best = false;

    const auto try_evaluate_scale_with_score_bounds = [&](double t,
                                                          BoundedRescaleState& result) {
        if (evaluations == kEvaluationBudget)
            return false;
        ++evaluations;
        result.state = evaluator(magnitudes, reciprocals, dim, max_code, t_start, t_end, t);
        // Reject evaluator states whose plateau does not contain t.
        if (result.state.first > t || result.state.next <= t)
            return false;
        // If the computed numerator is N_hat and its relative error is at most
        // delta, then N_hat/(1+delta) <= N <= N_hat/(1-delta). Round outward
        // again when squaring/dividing so pruning never relies on a rounded-up
        // candidate score or a rounded-down interval bound.
        result.numerator_lower = next_double_down(result.state.numerator / lower_divisor);
        result.numerator_upper = next_double_up(result.state.numerator / upper_divisor);
        result.score_lower = next_double_down(
            next_double_down(result.numerator_lower * result.numerator_lower) /
            result.state.squared_norm
        );
        result.score_upper = next_double_up(
            next_double_up(result.numerator_upper * result.numerator_upper) /
            result.state.squared_norm
        );
        best_lower = std::max(best_lower, result.score_lower);

        // Replace the winner only when the score enclosures are disjoint.
        // Equal S means the same code plateau, since codes are monotonic in t.
        // Overlapping scores from different plateaus require the heap's tie order.
        if (!have_best || result.score_lower > best.score_upper) {
            best = result;
            have_best = true;
        } else if (
            result.score_upper >= best.score_lower &&
            result.state.squared_norm != best.state.squared_norm
        ) {
            return false;
        }
        return true;
    };

    BoundedRescaleState first;
    BoundedRescaleState last;
    // t_end is excluded by the heap search too. Evaluate the preceding double
    // so an event exactly at t_end is not accidentally admitted here.
    if (!try_evaluate_scale_with_score_bounds(t_start, first) ||
        !try_evaluate_scale_with_score_bounds(next_double_down(t_end), last))
        return false;

    const auto try_search_interval = [&](auto&& self,
                                         const BoundedRescaleState& left,
                                         const BoundedRescaleState& right,
                                         double upper_bound,
                                         size_t depth) -> bool {
        // Codes increase coordinate-wise with scale, so equal squared norms
        // mean equal codes. Adjacent plateaus have no intermediate states;
        // otherwise prune only when the entire interval is provably worse.
        if (left.state.squared_norm == right.state.squared_norm ||
            left.state.next == right.state.first || upper_bound < best_lower)
            return true;
        if (left.state.next > right.state.first || depth == kMaxDepth)
            return false;

        // Split only [left.next, right.first); stay below the right plateau
        // despite rounding.
        double midpoint = left.state.next + (right.state.first - left.state.next) * 0.5;
        midpoint = std::max(
            left.state.next, std::min(midpoint, next_double_down(right.state.first))
        );
        if (midpoint >= right.state.first)
            return false;
        BoundedRescaleState middle;
        if (!try_evaluate_scale_with_score_bounds(midpoint, middle) ||
            middle.state.squared_norm <= left.state.squared_norm ||
            middle.state.squared_norm >= right.state.squared_norm)
            return false;

        // Exclude the middle plateau from both recursive gaps.
        const double left_upper = interval_squared_cosine_upper_bound(left, middle);
        const double right_upper = interval_squared_cosine_upper_bound(middle, right);
        // Explore the stronger bound first to improve subsequent pruning.
        if (left_upper > right_upper) {
            return self(self, left, middle, left_upper, depth + 1) &&
                   self(self, middle, right, right_upper, depth + 1);
        }
        return self(self, middle, right, right_upper, depth + 1) &&
               self(self, left, middle, left_upper, depth + 1);
    };
    if (!try_search_interval(
            try_search_interval,
            first,
            last,
            interval_squared_cosine_upper_bound(first, last),
            0
        ))
        return false;
    // Return the first scale of the winning plateau, not the sampled midpoint,
    // to preserve the event sweep's canonical scale convention.
    best_t = best.state.first;
    return true;
}

// The searched scale, or -1 to ask the caller to rerun the exact heap search.
template <typename Evaluator>
static double best_scale_or_fallback(
    const float* input,
    size_t dim,
    int max_code,
    double t_start,
    double t_end,
    Evaluator evaluator,
    double* magnitudes = nullptr,
    double* reciprocals = nullptr
) {
    double best_t = 0;
    return try_find_best_scale_by_interval_search(
               input,
               dim,
               max_code,
               t_start,
               t_end,
               evaluator,
               best_t,
               magnitudes,
               reciprocals
           )
               ? best_t
               : -1;
}

}  // namespace rabitqlib::simd::detail
