#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace rabitqlib::simd {
namespace {
// The imported header has no includes. Keep its inline helpers private so the
// linker cannot merge compiler-generated code from different ISA backends.
#include "rabitqlib/utils/fht_avx.hpp"
inline void rescale(float* data, size_t dim, float factor) {
    for (size_t i = 0; i < dim; ++i) {
        data[i] *= factor;
    }
}
// FFHT retains its imported AVX butterflies in both ISA backends. Surrounding
// C++ is compiled at the selected ISA; no vendor implementation is rewritten.
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
    switch (trunc_dim) {
        case 64:
            fht = helper_float_6;
            break;
        case 128:
            fht = helper_float_7;
            break;
        case 256:
            fht = helper_float_8;
            break;
        case 512:
            fht = helper_float_9;
            break;
        case 1024:
            fht = helper_float_10;
            break;
        case 2048:
            fht = helper_float_11;
            break;
        default:
            throw std::invalid_argument("Unsupported dimension for FhtKacRotator");
    }

    std::memcpy(rotated_vec, data, sizeof(float) * dim);
    std::fill(rotated_vec + dim, rotated_vec + padded_dim, 0);

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
