#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <limits>

#include "rabitqlib/simd/quantization_dispatch.hpp"
#include "rescale_search.hpp"
namespace rabitqlib::simd::detail {

// Evaluate one candidate scale, processing four coordinates per iteration.
// For a_i = magnitudes[i], c_i = min(floor(scale*a_i), max_code), q_i = c_i+0.5,
// return N = sum(a_i*q_i), S = sum(q_i*q_i), and the constant-code interval
// [first, next). The shared search uses N^2/S to compare and bound squared cosine.
// No code vector is stored here; the caller emits codes once the search finishes.
//
// Magnitudes are float32 values promoted exactly to double. For levels <= 255,
// unequal comparable level/magnitude ratios differ by at least 2^-33 relatively.
// Reciprocal multiplication has much less error, so its extreme indices remain
// valid. Recompute those two endpoints by division to retain exact event scales.
static RescaleSearchState evaluate_scale_state_avx2(
    const double* magnitudes,
    const double* reciprocals,
    size_t dim,
    int max_code,
    double t_start,
    double t_end,
    double scale
) {
    __m256d z = _mm256_setzero_pd(), one = _mm256_set1_pd(1), half = _mm256_set1_pd(.5),
            vt = _mm256_set1_pd(scale), vcap = _mm256_set1_pd(max_code);
    __m256d guard = _mm256_set1_pd(
                2 * std::numeric_limits<double>::epsilon() * (max_code + 1.0)
            ),
            vn = z, vd = z, vfirst = z,
            vnext = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    __m256d positions = _mm256_setr_pd(0, 1, 2, 3), first_idx = _mm256_set1_pd(-1),
            next_idx = first_idx;
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        __m256d v = _mm256_loadu_pd(magnitudes + i), vi = _mm256_loadu_pd(reciprocals + i),
                product = _mm256_mul_pd(v, vt),
                c = _mm256_min_pd(_mm256_floor_pd(product), vcap);
        // Floor is sufficient away from integer boundaries. The guard covers
        // multiplication rounding near those boundaries; saturation removes
        // the upper boundary. Zero magnitudes always keep code zero.
        __m256d interior = _mm256_and_pd(
            _mm256_cmp_pd(product, _mm256_add_pd(c, guard), _CMP_GT_OQ),
            _mm256_or_pd(
                _mm256_cmp_pd(c, vcap, _CMP_EQ_OQ),
                _mm256_cmp_pd(
                    product, _mm256_sub_pd(_mm256_add_pd(c, one), guard), _CMP_LT_OQ
                )
            )
        );
        unsigned boundary =
            (~static_cast<unsigned>(_mm256_movemask_pd(interior))) &
            static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(v, z, _CMP_GT_OQ))) & 15;
        // Repair only flagged lanes using the heap's division-based thresholds.
        // This ensures an event at k/a_i belongs to code k in both algorithms.
        if (boundary) {
            alignas(32) double corrected[4];
            _mm256_store_pd(corrected, c);
            for (unsigned mask = boundary; mask; mask &= mask - 1) {
                unsigned k = __builtin_ctz(mask);
                corrected[k] = quantized_code_at_scale(magnitudes[i + k], scale, max_code);
            }
            c = _mm256_load_pd(corrected);
        }
        // vn/vd hold lane-local contributions to N/S. Accumulate in double;
        // the shared search encloses rounding in N, while S remains exact.
        __m256d q = _mm256_add_pd(c, half);
        vn = _mm256_add_pd(vn, _mm256_mul_pd(v, q));
        vd = _mm256_add_pd(vd, _mm256_mul_pd(q, q));
        // A coordinate enters its current code at c_i/a_i (only if c_i>0)
        // and leaves at (c_i+1)/a_i (only if a_i>0 and not saturated).
        // Inactive coordinates contribute neutral values: zero to max(first)
        // and infinity to min(next). Reciprocals avoid per-coordinate division.
        __m256d af = _mm256_cmp_pd(c, z, _CMP_GT_OQ),
                an = _mm256_and_pd(
                    _mm256_cmp_pd(c, vcap, _CMP_LT_OQ), _mm256_cmp_pd(v, z, _CMP_GT_OQ)
                );
        __m256d f = _mm256_blendv_pd(z, _mm256_mul_pd(c, vi), af),
                n = _mm256_blendv_pd(
                    _mm256_set1_pd(std::numeric_limits<double>::infinity()),
                    _mm256_mul_pd(_mm256_add_pd(c, one), vi),
                    an
                );
        __m256d fm = _mm256_and_pd(af, _mm256_cmp_pd(f, vfirst, _CMP_GT_OQ)),
                nm = _mm256_and_pd(an, _mm256_cmp_pd(n, vnext, _CMP_LT_OQ));
        // Retain original coordinate indices along with each lane's extrema,
        // so the final boundary can be recomputed by division below.
        first_idx = _mm256_blendv_pd(first_idx, positions, fm);
        next_idx = _mm256_blendv_pd(next_idx, positions, nm);
        vfirst = _mm256_max_pd(vfirst, f);
        vnext = _mm256_min_pd(vnext, n);
        positions = _mm256_add_pd(positions, _mm256_set1_pd(4));
    }
    // Reduce the four partial sums and select the global boundary coordinates.
    // Index -1 means this lane never had an active boundary candidate.
    alignas(32) double ns[4], ds[4], fs[4], ts[4];
    _mm256_store_pd(ns, vn);
    _mm256_store_pd(ds, vd);
    _mm256_store_pd(fs, vfirst);
    _mm256_store_pd(ts, vnext);
    RescaleSearchState s{
        ns[0] + ns[1] + ns[2] + ns[3], ds[0] + ds[1] + ds[2] + ds[3], t_start, t_end};
    alignas(32) double first_positions[4], next_positions[4];
    _mm256_store_pd(first_positions, first_idx);
    _mm256_store_pd(next_positions, next_idx);
    size_t fi[4], ni[4];
    for (size_t k = 0; k < 4; ++k) {
        fi[k] = first_positions[k] < 0 ? std::numeric_limits<size_t>::max()
                                       : static_cast<size_t>(first_positions[k]);
        ni[k] = next_positions[k] < 0 ? std::numeric_limits<size_t>::max()
                                      : static_cast<size_t>(next_positions[k]);
    }
    size_t fk = std::max_element(fs, fs + 4) - fs, nk = std::min_element(ts, ts + 4) - ts;
    // Reciprocal products selected the coordinates; division now recovers the
    // exact heap event scales, clipped to the configured search interval.
    if (fi[fk] != std::numeric_limits<size_t>::max())
        s.first = std::max(
            s.first,
            static_cast<double>(quantized_code_at_scale(magnitudes[fi[fk]], scale, max_code)
            ) / magnitudes[fi[fk]]
        );
    if (ni[nk] != std::numeric_limits<size_t>::max())
        s.next = std::min(
            s.next,
            (quantized_code_at_scale(magnitudes[ni[nk]], scale, max_code) + 1.0) /
                magnitudes[ni[nk]]
        );
    // Handle the remaining zero to three coordinates with the same definitions.
    for (; i < dim; ++i) {
        int c = quantized_code_at_scale(magnitudes[i], scale, max_code);
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

}  // namespace rabitqlib::simd::detail

namespace rabitqlib::simd {
double best_rescale_factor_avx2(
    const float* magnitudes, size_t dim, int max_code, double start, double end
) {
    // A negative result asks the caller to rerun the scalar heap search.
    double best_t = 0;
    return detail::try_find_best_scale_by_interval_search(
               magnitudes,
               dim,
               max_code,
               start,
               end,
               detail::evaluate_scale_state_avx2,
               best_t
           )
               ? best_t
               : -1;
}
}  // namespace rabitqlib::simd
