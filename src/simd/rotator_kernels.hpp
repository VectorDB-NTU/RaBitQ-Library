#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "fht_kernels.hpp"

namespace rabitqlib::simd {
namespace {
// Preserve the final Kac arithmetic and scale, storing each result only once.
inline void kacs_walk_rescale(float* data, size_t dim, float factor) {
    const size_t half = dim / 2;
#if defined(__AVX512F__)
    const __m512 multiplier512 = _mm512_set1_ps(factor);
    for (size_t i = 0; i < half; i += 16) {
        const __m512 left = _mm512_loadu_ps(data + i);
        const __m512 right = _mm512_loadu_ps(data + half + i);
        _mm512_storeu_ps(
            data + i, _mm512_mul_ps(_mm512_add_ps(left, right), multiplier512)
        );
        _mm512_storeu_ps(
            data + half + i, _mm512_mul_ps(_mm512_sub_ps(left, right), multiplier512)
        );
    }
#else
    const __m256 multiplier = _mm256_set1_ps(factor);
    for (size_t i = 0; i < half; i += 8) {
        const __m256 left = _mm256_loadu_ps(data + i);
        const __m256 right = _mm256_loadu_ps(data + half + i);
        _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_add_ps(left, right), multiplier));
        _mm256_storeu_ps(
            data + half + i, _mm256_mul_ps(_mm256_sub_ps(left, right), multiplier)
        );
    }
#endif
}
// Both backends and all compilers use the same AVX butterfly ordering.
template <auto flip_sign, auto kacs_walk, auto kacs_flip = nullptr>
void fht_rotate_impl(
    const float* data,
    float* rotated_vec,
    size_t dim,
    size_t padded_dim,
    size_t trunc_dim,
    float fac,
    const uint8_t* flip
) {
    void (*fht)(float*, float) = nullptr;
#define RABITQ_FHT_CASE(log_dim)             \
    case size_t{1} << (log_dim):             \
        fht = fht_intrinsics<log_dim, true>; \
        break;
    switch (trunc_dim) {
        RABITQ_FHT_CASE(6)
        RABITQ_FHT_CASE(7)
        RABITQ_FHT_CASE(8)
        RABITQ_FHT_CASE(9)
        RABITQ_FHT_CASE(10)
        RABITQ_FHT_CASE(11)
        RABITQ_FHT_CASE(12)
        RABITQ_FHT_CASE(13)
        RABITQ_FHT_CASE(14)
        RABITQ_FHT_CASE(15)
        RABITQ_FHT_CASE(16)
        default:
            throw std::invalid_argument("Unsupported dimension for FhtKacRotator");
    }
#undef RABITQ_FHT_CASE

    std::memcpy(rotated_vec, data, sizeof(float) * dim);
    std::fill(rotated_vec + dim, rotated_vec + padded_dim, 0.0F);

    if (trunc_dim == padded_dim) {
        for (size_t round = 0; round < 4; ++round) {
            flip_sign(flip + round * (padded_dim / 8), rotated_vec, padded_dim);
            fht(rotated_vec, fac);
        }
        return;
    }

    const auto walk_and_flip = [&](const uint8_t* signs) {
        if constexpr (kacs_flip != nullptr) {
            kacs_flip(signs, rotated_vec, padded_dim);
        } else {
            kacs_walk(rotated_vec, padded_dim);
            flip_sign(signs, rotated_vec, padded_dim);
        }
    };
    size_t start = padded_dim - trunc_dim;

    flip_sign(flip, rotated_vec, padded_dim);
    fht(rotated_vec, fac);
    walk_and_flip(flip + (padded_dim / 8));
    fht(rotated_vec + start, fac);
    walk_and_flip(flip + (2 * padded_dim / 8));
    fht(rotated_vec, fac);
    walk_and_flip(flip + (3 * padded_dim / 8));
    fht(rotated_vec + start, fac);
    kacs_walk_rescale(rotated_vec, padded_dim, 0.25F);
}
}  // namespace
}  // namespace rabitqlib::simd
