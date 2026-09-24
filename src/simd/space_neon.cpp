#include <arm_neon.h>

#include <cstddef>

#include "rabitqlib/simd/space_dispatch.hpp"

namespace rabitqlib::simd {
namespace {
// Four independent float accumulation chains, matching the x86 kernel strategy.
template <bool SquaredDistance>
float product(const float* a, const float* b, size_t dim) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    float32x4_t s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
    auto step = [&](float32x4_t sum, size_t at) {
        const float32x4_t x = vld1q_f32(a + at);
        const float32x4_t y = vld1q_f32(b + at);
        if constexpr (SquaredDistance) {
            const float32x4_t diff = vsubq_f32(x, y);
            return vfmaq_f32(sum, diff, diff);
        } else {
            return vfmaq_f32(sum, x, y);
        }
    };
    size_t i = 0;
    for (; dim - i >= 16; i += 16) {
        s0 = step(s0, i);
        s1 = step(s1, i + 4);
        s2 = step(s2, i + 8);
        s3 = step(s3, i + 12);
    }
    float32x4_t sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    for (; dim - i >= 4; i += 4)
        sum = step(sum, i);
    float result = vaddvq_f32(sum);
    for (; i < dim; ++i) {
        if constexpr (SquaredDistance) {
            const float diff = a[i] - b[i];
            result += diff * diff;
        } else {
            result += a[i] * b[i];
        }
    }
    return result;
}
}  // namespace
float euclidean_sqr_neon(const float* a, const float* b, size_t dim) {
    return product<true>(a, b, dim);
}
float dot_product_neon(const float* a, const float* b, size_t dim) {
    return product<false>(a, b, dim);
}
float dot_product_dis_neon(const float* a, const float* b, size_t dim) {
    return 1.0F - dot_product_neon(a, b, dim);
}
float l2norm_sqr_neon(const float* a, size_t dim) { return dot_product_neon(a, a, dim); }
}  // namespace rabitqlib::simd
