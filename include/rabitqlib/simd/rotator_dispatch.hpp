#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::simd {

void flip_sign(const uint8_t* flip, float* data, size_t dim);

void kacs_walk(float* data, size_t len);

void fht_rotate(
    const float* data,
    float* rotated_vec,
    size_t dim,
    size_t padded_dim,
    size_t trunc_dim,
    float fac,
    const uint8_t* flip
);

void flip_sign_avx2(const uint8_t* flip, float* data, size_t dim);

void kacs_walk_avx2(float* data, size_t len);

void fht_rotate_avx2(
    const float* data,
    float* rotated_vec,
    size_t dim,
    size_t padded_dim,
    size_t trunc_dim,
    float fac,
    const uint8_t* flip
);

void flip_sign_avx512(const uint8_t* flip, float* data, size_t dim);

void kacs_walk_avx512(float* data, size_t len);

void fht_rotate_avx512(
    const float* data,
    float* rotated_vec,
    size_t dim,
    size_t padded_dim,
    size_t trunc_dim,
    float fac,
    const uint8_t* flip
);

}  // namespace rabitqlib::simd
