#include <cstddef>
#include <cstdint>

#include "rabitqlib/core/detail/simd/matrix_dispatch.hpp"

#define RABITQ_MATRIX_EIGEN_NAMESPACE rabitqlib_eigen_avx2
#include "matrix_kernels.hpp"

namespace rabitqlib::simd {
void accumulate_cluster_sums_avx2(
    const float* data,
    const size_t* point_ids,
    size_t count,
    size_t dim,
    const uint32_t* assignments,
    double* sums,
    size_t* counts
) {
    accumulate_cluster_sums_impl(data, point_ids, count, dim, assignments, sums, counts);
}

void matrix_product_avx2(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
) {
    matrix_product_impl(left, right, result, rows, inner, cols);
}

void matrix_product_transposed_avx2(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
) {
    matrix_product_transposed_impl(left, right, result, rows, inner, cols);
}

void row_norms_avx2(const float* data, float* result, size_t rows, size_t dim) {
    row_norms_impl(data, result, rows, dim);
}

void pairwise_distances_lower_avx2(
    const float* data,
    float* result,
    float* norms_data,
    size_t size,
    size_t dim,
    bool inner_product
) {
    pairwise_distances_lower_impl(data, result, norms_data, size, dim, inner_product);
}
}  // namespace rabitqlib::simd
