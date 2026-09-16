#pragma once

#include <cstddef>

namespace rabitqlib::simd {
// Dense row-major float32 kernels; inputs and outputs must not overlap.
// These preserve the caller's Eigen operations and surrounding thread scheduling.
void matrix_product(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
);

void matrix_product_generic(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
);

void matrix_product_avx2(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
);

void matrix_product_avx512(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
);

void matrix_product_transposed(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
);

void matrix_product_transposed_generic(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
);

void matrix_product_transposed_avx2(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
);

void matrix_product_transposed_avx512(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
);

void row_norms(const float* data, float* result, size_t rows, size_t dim);

void row_norms_generic(const float* data, float* result, size_t rows, size_t dim);

void row_norms_avx2(const float* data, float* result, size_t rows, size_t dim);

void row_norms_avx512(const float* data, float* result, size_t rows, size_t dim);

void pairwise_distances_lower(
    const float* data,
    float* result,
    float* norms_data,
    size_t size,
    size_t dim,
    bool inner_product
);

void pairwise_distances_lower_generic(
    const float* data,
    float* result,
    float* norms_data,
    size_t size,
    size_t dim,
    bool inner_product
);

void pairwise_distances_lower_avx2(
    const float* data,
    float* result,
    float* norms_data,
    size_t size,
    size_t dim,
    bool inner_product
);

void pairwise_distances_lower_avx512(
    const float* data,
    float* result,
    float* norms_data,
    size_t size,
    size_t dim,
    bool inner_product
);
}  // namespace rabitqlib::simd
