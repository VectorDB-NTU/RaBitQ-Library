#pragma once

#include <algorithm>
#include <cstddef>

// Keep Eigen's ISA-dependent template helpers private to each backend. The
// default also lets this header compile on its own for static analysis.
#ifndef RABITQ_MATRIX_EIGEN_NAMESPACE
#define RABITQ_MATRIX_EIGEN_NAMESPACE rabitqlib_eigen_generic
#endif
#define Eigen RABITQ_MATRIX_EIGEN_NAMESPACE
#include "rabitqlib/third/Eigen/Dense"
#undef Eigen

namespace rabitqlib::simd {
namespace {
namespace kernel_eigen = RABITQ_MATRIX_EIGEN_NAMESPACE;
#undef RABITQ_MATRIX_EIGEN_NAMESPACE

using Index = kernel_eigen::Index;
using Matrix = kernel_eigen::
    Matrix<float, kernel_eigen::Dynamic, kernel_eigen::Dynamic, kernel_eigen::RowMajor>;
using ConstRowMajorMatrixMap = kernel_eigen::Map<const Matrix>;
using RowMajorMatrixMap = kernel_eigen::Map<Matrix>;
using VectorMap = kernel_eigen::Map<kernel_eigen::Matrix<float, kernel_eigen::Dynamic, 1>>;

inline void matrix_product_impl(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
) {
    ConstRowMajorMatrixMap a(left, static_cast<Index>(rows), static_cast<Index>(inner));
    ConstRowMajorMatrixMap b(right, static_cast<Index>(inner), static_cast<Index>(cols));
    RowMajorMatrixMap out(result, static_cast<Index>(rows), static_cast<Index>(cols));
    out.noalias() = a * b;
}

inline void matrix_product_transposed_impl(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
) {
    ConstRowMajorMatrixMap a(left, static_cast<Index>(rows), static_cast<Index>(inner));
    ConstRowMajorMatrixMap b(right, static_cast<Index>(cols), static_cast<Index>(inner));
    RowMajorMatrixMap out(result, static_cast<Index>(rows), static_cast<Index>(cols));
    out.noalias() = a * b.transpose();
}

inline void row_norms_impl(const float* data, float* result, size_t rows, size_t dim) {
    ConstRowMajorMatrixMap points(data, static_cast<Index>(rows), static_cast<Index>(dim));
    VectorMap norms(result, static_cast<Index>(rows));
    norms = points.rowwise().squaredNorm();
}

inline void pairwise_distances_lower_impl(
    const float* data,
    float* result,
    float* norms_data,
    size_t size,
    size_t dim,
    bool inner_product
) {
    const auto count = static_cast<Index>(size);
    ConstRowMajorMatrixMap points(data, count, static_cast<Index>(dim));
    RowMajorMatrixMap distances(result, count, count);
    VectorMap norms(norms_data, count);
    distances.setZero();
    distances.selfadjointView<kernel_eigen::Lower>().rankUpdate(points);
    norms = points.rowwise().squaredNorm();
    for (kernel_eigen::Index i = 0; i < count; ++i) {
        for (kernel_eigen::Index j = 0; j <= i; ++j) {
            const float dot = distances(i, j);
            distances(i, j) =
                !inner_product ? std::max(0.0F, norms[i] + norms[j] - 2 * dot) : -dot;
        }
    }
}
}  // namespace
}  // namespace rabitqlib::simd
