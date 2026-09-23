// AVX intrinsics FHT for RaBitQ rotation, inspired by FFHT:
// https://github.com/FALCONN-LIB/FFHT

#pragma once

#include <immintrin.h>

#include <cstddef>

namespace rabitqlib::simd {
namespace {

// Each stage forms [a+b, a-b] by swapping the paired lanes, negating the
// original upper lane, and adding. XOR changes only the sign bit, replacing
// the duplicated-lane permutations, subtraction from zero, and blends.
// The masks negate odd lanes, upper lane pairs, then the upper 128-bit half.
template <int stage>
inline __m256 fht_lane_stage(__m256 x) {
    if constexpr (stage == 0) {
        const __m256 sign =
            _mm256_setr_ps(0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F);
        return _mm256_add_ps(_mm256_permute_ps(x, 0xB1), _mm256_xor_ps(x, sign));
    } else if constexpr (stage == 1) {
        const __m256 sign =
            _mm256_setr_ps(0.0F, 0.0F, -0.0F, -0.0F, 0.0F, 0.0F, -0.0F, -0.0F);
        return _mm256_add_ps(_mm256_permute_ps(x, 0x4E), _mm256_xor_ps(x, sign));
    } else {
        const __m256 sign =
            _mm256_setr_ps(0.0F, 0.0F, 0.0F, 0.0F, -0.0F, -0.0F, -0.0F, -0.0F);
        return _mm256_add_ps(_mm256_permute2f128_ps(x, x, 0x01), _mm256_xor_ps(x, sign));
    }
}

inline void fht_butterfly(__m256& a, __m256& b) {
    const __m256 sum = _mm256_add_ps(a, b);
    b = _mm256_sub_ps(a, b);
    a = sum;
}

// Fuse three stages so eight registers are loaded and stored only once.
// Stride is eight for the 64-point leaf, or N/8 for a larger transform.
template <bool leaf>
inline void fht_eight_registers(float* data, size_t stride) {
    __m256 a = _mm256_loadu_ps(data);
    __m256 b = _mm256_loadu_ps(data + stride);
    __m256 c = _mm256_loadu_ps(data + 2 * stride);
    __m256 d = _mm256_loadu_ps(data + 3 * stride);
    __m256 e = _mm256_loadu_ps(data + 4 * stride);
    __m256 f = _mm256_loadu_ps(data + 5 * stride);
    __m256 g = _mm256_loadu_ps(data + 6 * stride);
    __m256 h = _mm256_loadu_ps(data + 7 * stride);
    if constexpr (leaf) {
        // As in FFHT's assembly, finish each stage across all eight registers
        // before starting the next, exposing independent work to the scheduler.
        a = fht_lane_stage<0>(a);
        b = fht_lane_stage<0>(b);
        c = fht_lane_stage<0>(c);
        d = fht_lane_stage<0>(d);
        e = fht_lane_stage<0>(e);
        f = fht_lane_stage<0>(f);
        g = fht_lane_stage<0>(g);
        h = fht_lane_stage<0>(h);
        a = fht_lane_stage<1>(a);
        b = fht_lane_stage<1>(b);
        c = fht_lane_stage<1>(c);
        d = fht_lane_stage<1>(d);
        e = fht_lane_stage<1>(e);
        f = fht_lane_stage<1>(f);
        g = fht_lane_stage<1>(g);
        h = fht_lane_stage<1>(h);
        a = fht_lane_stage<2>(a);
        b = fht_lane_stage<2>(b);
        c = fht_lane_stage<2>(c);
        d = fht_lane_stage<2>(d);
        e = fht_lane_stage<2>(e);
        f = fht_lane_stage<2>(f);
        g = fht_lane_stage<2>(g);
        h = fht_lane_stage<2>(h);
    }
    fht_butterfly(a, b);
    fht_butterfly(c, d);
    fht_butterfly(e, f);
    fht_butterfly(g, h);
    fht_butterfly(a, c);
    fht_butterfly(b, d);
    fht_butterfly(e, g);
    fht_butterfly(f, h);
    fht_butterfly(a, e);
    fht_butterfly(b, f);
    fht_butterfly(c, g);
    fht_butterfly(d, h);
    _mm256_storeu_ps(data, a);
    _mm256_storeu_ps(data + stride, b);
    _mm256_storeu_ps(data + 2 * stride, c);
    _mm256_storeu_ps(data + 3 * stride, d);
    _mm256_storeu_ps(data + 4 * stride, e);
    _mm256_storeu_ps(data + 5 * stride, f);
    _mm256_storeu_ps(data + 6 * stride, g);
    _mm256_storeu_ps(data + 7 * stride, h);
}

template <size_t log_dim>
void fht_intrinsics(float* data) {
    static_assert(log_dim >= 6 && log_dim <= 16);
    constexpr size_t kDim = size_t{1} << log_dim;
    if constexpr (log_dim == 6) {
        fht_eight_registers<true>(data, 8);
    } else if constexpr (log_dim < 9) {
        fht_intrinsics<log_dim - 1>(data);
        fht_intrinsics<log_dim - 1>(data + kDim / 2);
        for (size_t i = 0; i < kDim / 2; i += 8) {
            __m256 a = _mm256_loadu_ps(data + i);
            __m256 b = _mm256_loadu_ps(data + kDim / 2 + i);
            fht_butterfly(a, b);
            _mm256_storeu_ps(data + i, a);
            _mm256_storeu_ps(data + kDim / 2 + i, b);
        }
    } else {
        // Recurse into contiguous blocks before merging, keeping the working
        // set cache-local even for the largest supported rotation.
        constexpr size_t kStride = kDim / 8;
        for (size_t block = 0; block < kDim; block += kStride) {
            fht_intrinsics<log_dim - 3>(data + block);
        }
        for (size_t i = 0; i < kStride; i += 8) {
            fht_eight_registers<false>(data + i, kStride);
        }
    }
}

}  // namespace
}  // namespace rabitqlib::simd
