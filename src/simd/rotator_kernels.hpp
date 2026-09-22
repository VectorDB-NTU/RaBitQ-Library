#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "fht_kernels.hpp"

namespace rabitqlib::simd {
namespace {
inline void rescale(float* data, size_t dim, float factor) {
    for (size_t i = 0; i < dim; ++i) {
        data[i] *= factor;
    }
}
// Both backends and all compilers use the same AVX butterfly ordering.
template <auto flip_sign, auto kacs_walk>
void fht_rotate_impl(
    const float* data,
    float* rotated_vec,
    size_t dim,
    size_t padded_dim,
    size_t trunc_dim,
    float fac,
    const uint8_t* flip
) {
    void (*fht)(float*) = nullptr;
#define RABITQ_FHT_CASE(log_dim)       \
    case size_t{1} << (log_dim):       \
        fht = fht_intrinsics<log_dim>; \
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
        flip_sign(flip, rotated_vec, padded_dim);
        fht(rotated_vec);
        rescale(rotated_vec, trunc_dim, fac);

        flip_sign(flip + (padded_dim / 8), rotated_vec, padded_dim);
        fht(rotated_vec);
        rescale(rotated_vec, trunc_dim, fac);

        flip_sign(flip + (2 * padded_dim / 8), rotated_vec, padded_dim);
        fht(rotated_vec);
        rescale(rotated_vec, trunc_dim, fac);

        flip_sign(flip + (3 * padded_dim / 8), rotated_vec, padded_dim);
        fht(rotated_vec);
        rescale(rotated_vec, trunc_dim, fac);

        return;
    }

    size_t start = padded_dim - trunc_dim;

    flip_sign(flip, rotated_vec, padded_dim);
    fht(rotated_vec);
    rescale(rotated_vec, trunc_dim, fac);
    kacs_walk(rotated_vec, padded_dim);

    flip_sign(flip + (padded_dim / 8), rotated_vec, padded_dim);
    fht(rotated_vec + start);
    rescale(rotated_vec + start, trunc_dim, fac);
    kacs_walk(rotated_vec, padded_dim);

    flip_sign(flip + (2 * padded_dim / 8), rotated_vec, padded_dim);
    fht(rotated_vec);
    rescale(rotated_vec, trunc_dim, fac);
    kacs_walk(rotated_vec, padded_dim);

    flip_sign(flip + (3 * padded_dim / 8), rotated_vec, padded_dim);
    fht(rotated_vec + start);
    rescale(rotated_vec + start, trunc_dim, fac);
    kacs_walk(rotated_vec, padded_dim);

    // This can be removed if we don't care about the absolute value of
    // similarities.
    rescale(rotated_vec, padded_dim, 0.25F);
}
}  // namespace
}  // namespace rabitqlib::simd
