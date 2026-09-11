#pragma once

#include <immintrin.h>

#include <cstddef>

namespace rabitqlib::simd {
namespace {
#ifdef __AVX512F__
using Vec = __m512;
constexpr size_t kWidth = 16;
inline Vec zero() { return _mm512_setzero_ps(); }
inline Vec load(const float* p) { return _mm512_loadu_ps(p); }
inline Vec add(Vec a, Vec b) { return _mm512_add_ps(a, b); }
inline Vec sub(Vec a, Vec b) { return _mm512_sub_ps(a, b); }
inline Vec fma(Vec a, Vec b, Vec c) { return _mm512_fmadd_ps(a, b, c); }
inline float reduce(Vec a) { return _mm512_reduce_add_ps(a); }
#else
using Vec = __m256;
constexpr size_t kWidth = 8;
inline Vec zero() { return _mm256_setzero_ps(); }
inline Vec load(const float* p) { return _mm256_loadu_ps(p); }
inline Vec add(Vec a, Vec b) { return _mm256_add_ps(a, b); }
inline Vec sub(Vec a, Vec b) { return _mm256_sub_ps(a, b); }
inline Vec fma(Vec a, Vec b, Vec c) { return _mm256_fmadd_ps(a, b, c); }
inline float reduce(Vec a) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_movehdup_ps(h)));
}
#endif
inline Vec tail_load(const float* p, size_t remaining) {
#ifdef __AVX512F__
    return _mm512_maskz_loadu_ps(static_cast<__mmask16>((1U << remaining) - 1), p);
#else
    __m256i mask = _mm256_cmpgt_epi32(
        _mm256_set1_epi32(static_cast<int>(remaining)),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
    );
    return _mm256_maskload_ps(p, mask);
#endif
}
// Four independent accumulation chains avoid a single FMA dependency bottleneck.
// Partial loads preserve the unpadded-input contract.
enum class FloatOperation { SquaredL2, Dot, InnerProductDistance, SquaredNorm };

template <FloatOperation Op>
inline float raw_float(const float* a, const float* b, size_t n) {
    Vec s0 = zero(), s1 = zero(), s2 = zero(), s3 = zero();
    size_t i = 0;
    auto accumulate = [](Vec sum, Vec x, Vec y) {
        if constexpr (Op == FloatOperation::SquaredL2) {
            Vec d = sub(x, y);
            return fma(d, d, sum);
        } else if constexpr (Op == FloatOperation::SquaredNorm)
            return fma(x, x, sum);
        else
            return fma(x, y, sum);
    };
    auto step = [&](Vec sum, size_t at) {
        Vec x = load(a + at);
        if constexpr (Op == FloatOperation::SquaredNorm)
            return accumulate(sum, x, x);
        else
            return accumulate(sum, x, load(b + at));
    };
    for (; n - i >= kWidth * 4; i += kWidth * 4) {
        s0 = step(s0, i);
        s1 = step(s1, i + kWidth);
        s2 = step(s2, i + kWidth * 2);
        s3 = step(s3, i + kWidth * 3);
    }
    Vec sum = add(add(s0, s1), add(s2, s3));
    for (; n - i >= kWidth; i += kWidth)
        sum = step(sum, i);
    if (i < n) {
        Vec x = tail_load(a + i, n - i);
        if constexpr (Op == FloatOperation::SquaredNorm)
            sum = accumulate(sum, x, x);
        else
            sum = accumulate(sum, x, tail_load(b + i, n - i));
    }
    float result = reduce(sum);
    if constexpr (Op == FloatOperation::InnerProductDistance)
        return 1.0F - result;
    return result;
}
}  // namespace
}  // namespace rabitqlib::simd
