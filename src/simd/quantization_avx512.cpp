#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <limits>

#include "rabitqlib/simd/quantization_dispatch.hpp"
#include "rescale_search.hpp"
namespace rabitqlib::simd::detail {

// Evaluate one candidate scale, processing eight coordinates per iteration.
// For a_i = magnitudes[i], c_i = min(floor(scale*a_i), max_code), q_i = c_i+0.5,
// return N = sum(a_i*q_i), S = sum(q_i*q_i), and the constant-code interval
// [first, next). The shared search uses N^2/S to compare and bound squared cosine.
// No code vector is stored here; the caller emits codes once the search finishes.
//
// Magnitudes are float32 values promoted exactly to double. For levels <= 255,
// unequal comparable level/magnitude ratios differ by at least 2^-33 relatively.
// Reciprocal multiplication has much less error, so its extreme indices remain
// valid. Recompute those two endpoints by division to retain exact event scales.
static RescaleSearchState evaluate_scale_state_avx512(
    const double* magnitudes,
    const double* reciprocals,
    size_t dim,
    int max_code,
    double t_start,
    double t_end,
    double scale
) {
    __m512d z = _mm512_setzero_pd(), one = _mm512_set1_pd(1), half = _mm512_set1_pd(.5),
            vt = _mm512_set1_pd(scale), vcap = _mm512_set1_pd(max_code);
    __m512d guard = _mm512_set1_pd(
                2 * std::numeric_limits<double>::epsilon() * (max_code + 1.0)
            ),
            vn = z, vd = z, vfirst = z,
            vnext = _mm512_set1_pd(std::numeric_limits<double>::infinity());
    __m512i positions = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7),
            first_idx = _mm512_set1_epi64(-1), next_idx = first_idx;
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m512d v = _mm512_loadu_pd(magnitudes + i), vi = _mm512_loadu_pd(reciprocals + i),
                product = _mm512_mul_pd(v, vt),
                c = _mm512_min_pd(_mm512_floor_pd(product), vcap);
        // Floor is sufficient away from integer boundaries. The guard covers
        // multiplication rounding near those boundaries; saturation removes
        // the upper boundary. Zero magnitudes always keep code zero.
        __mmask8 interior =
            _mm512_cmp_pd_mask(product, _mm512_add_pd(c, guard), _CMP_GT_OQ) &
            (_mm512_cmp_pd_mask(c, vcap, _CMP_EQ_OQ) |
             _mm512_cmp_pd_mask(
                 product, _mm512_sub_pd(_mm512_add_pd(c, one), guard), _CMP_LT_OQ
             ));
        __mmask8 boundary = ~interior & _mm512_cmp_pd_mask(v, z, _CMP_GT_OQ);
        // Repair only flagged lanes using the heap's division-based thresholds.
        // This ensures an event at k/a_i belongs to code k in both algorithms.
        if (boundary) {
            alignas(64) double corrected[8];
            _mm512_store_pd(corrected, c);
            for (unsigned mask = boundary; mask; mask &= mask - 1) {
                unsigned k = __builtin_ctz(mask);
                corrected[k] = quantized_code_at_scale(magnitudes[i + k], scale, max_code);
            }
            c = _mm512_load_pd(corrected);
        }
        // vn/vd hold lane-local contributions to N/S. Accumulate in double;
        // the shared search encloses rounding in N, while S remains exact.
        __m512d q = _mm512_add_pd(c, half);
        vn = _mm512_add_pd(vn, _mm512_mul_pd(v, q));
        vd = _mm512_add_pd(vd, _mm512_mul_pd(q, q));
        // A coordinate enters its current code at c_i/a_i (only if c_i>0)
        // and leaves at (c_i+1)/a_i (only if a_i>0 and not saturated).
        // Inactive coordinates contribute neutral values: zero to max(first)
        // and infinity to min(next). Reciprocals avoid per-coordinate division.
        __mmask8 activefirst = _mm512_cmp_pd_mask(c, z, _CMP_GT_OQ),
                 activenext = _mm512_cmp_pd_mask(c, vcap, _CMP_LT_OQ) &
                              _mm512_cmp_pd_mask(v, z, _CMP_GT_OQ);
        __m512d f = _mm512_maskz_mul_pd(activefirst, c, vi),
                n = _mm512_mask_mul_pd(
                    _mm512_set1_pd(std::numeric_limits<double>::infinity()),
                    activenext,
                    _mm512_add_pd(c, one),
                    vi
                );
        __mmask8 fm = activefirst & _mm512_cmp_pd_mask(f, vfirst, _CMP_GT_OQ),
                 nm = activenext & _mm512_cmp_pd_mask(n, vnext, _CMP_LT_OQ);
        // Retain original coordinate indices along with each lane's extrema,
        // so the final boundary can be recomputed by division below.
        first_idx = _mm512_mask_mov_epi64(first_idx, fm, positions);
        next_idx = _mm512_mask_mov_epi64(next_idx, nm, positions);
        vfirst = _mm512_max_pd(vfirst, f);
        vnext = _mm512_min_pd(vnext, n);
        positions = _mm512_add_epi64(positions, _mm512_set1_epi64(8));
    }
    // Reduce the eight partial sums and select the global boundary coordinates.
    // Index -1 becomes size_t::max(), marking lanes without an active candidate.
    RescaleSearchState s{
        _mm512_reduce_add_pd(vn), _mm512_reduce_add_pd(vd), t_start, t_end};
    alignas(64) size_t fi[8], ni[8];
    _mm512_store_si512(fi, first_idx);
    _mm512_store_si512(ni, next_idx);
    double f = _mm512_reduce_max_pd(vfirst), n = _mm512_reduce_min_pd(vnext);
    // At least one lane equals each reduced extremum. Select its first set bit;
    // tied boundary ratios describe the same event scale.
    unsigned fk = __builtin_ctz(static_cast<unsigned>(
                 _mm512_cmp_pd_mask(vfirst, _mm512_set1_pd(f), _CMP_EQ_OQ)
             )),
             nk = __builtin_ctz(static_cast<unsigned>(
                 _mm512_cmp_pd_mask(vnext, _mm512_set1_pd(n), _CMP_EQ_OQ)
             ));
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
    // Handle the remaining zero to seven coordinates with the same definitions.
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
double best_rescale_factor_avx512(
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
               detail::evaluate_scale_state_avx512,
               best_t
           )
               ? best_t
               : -1;
}
}  // namespace rabitqlib::simd
