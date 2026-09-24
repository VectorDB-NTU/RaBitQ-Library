#include "rabitqlib/simd/matrix_dispatch.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "rabitqlib/utils/cpu_features.hpp"

namespace rabitqlib::simd {
namespace {
struct MatrixBackend {
    decltype(&matrix_product) product;
    decltype(&matrix_product_transposed) transposed;
    decltype(&row_norms) norms;
    decltype(&pairwise_distances_lower) pairwise;
};

TEST(MatrixDispatchTest, AllBackendsMatchScalarForRectangularUnalignedInputs) {
    std::vector<MatrixBackend> backends{
        {matrix_product, matrix_product_transposed, row_norms, pairwise_distances_lower},
        {matrix_product_generic,
         matrix_product_transposed_generic,
         row_norms_generic,
         pairwise_distances_lower_generic}};
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        backends.push_back(
            {matrix_product_avx2,
             matrix_product_transposed_avx2,
             row_norms_avx2,
             pairwise_distances_lower_avx2}
        );
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx512_core()) {
        backends.push_back(
            {matrix_product_avx512,
             matrix_product_transposed_avx512,
             row_norms_avx512,
             pairwise_distances_lower_avx512}
        );
    }
#endif
    for (size_t dim : {1U, 7U, 16U, 17U, 65U, 128U}) {
        for (size_t rows : {1U, 33U}) {
            constexpr size_t kCols = 19;
            std::vector<float> a(rows * dim + 1), b(dim * kCols + 1), bt(kCols * dim + 1);
            for (size_t i = 0; i < rows * dim; ++i) {
                a[i + 1] = std::sin(static_cast<float>(i) * 0.13F);
            }
            for (size_t i = 0; i < dim; ++i) {
                for (size_t j = 0; j < kCols; ++j) {
                    b[1 + i * kCols + j] = std::cos(static_cast<float>(i + j * 3) * 0.27F);
                    bt[1 + j * dim + i] = b[1 + i * kCols + j];
                }
            }
            for (const auto& backend : backends) {
                std::vector<float> product(rows * kCols + 2, 12345), transposed(product),
                    norms(rows + 2, 12345), pairwise(rows * rows + 2, 12345);
                backend.product(
                    a.data() + 1, b.data() + 1, product.data() + 1, rows, dim, kCols
                );
                backend.transposed(
                    a.data() + 1, bt.data() + 1, transposed.data() + 1, rows, dim, kCols
                );
                backend.norms(a.data() + 1, norms.data() + 1, rows, dim);
                const double tolerance = 2e-6 * static_cast<double>(dim);
                for (size_t i = 0; i < rows; ++i) {
                    double norm = 0;
                    for (size_t k = 0; k < dim; ++k) {
                        norm +=
                            static_cast<double>(a[1 + i * dim + k]) * a[1 + i * dim + k];
                    }
                    EXPECT_NEAR(norms[i + 1], norm, tolerance);
                    for (size_t j = 0; j < kCols; ++j) {
                        double expected = 0;
                        for (size_t k = 0; k < dim; ++k) {
                            expected += static_cast<double>(a[1 + i * dim + k]) *
                                        b[1 + k * kCols + j];
                        }
                        EXPECT_NEAR(product[1 + i * kCols + j], expected, tolerance);
                        EXPECT_NEAR(transposed[1 + i * kCols + j], expected, tolerance);
                    }
                }
                for (bool ip : {false, true}) {
                    backend.pairwise(
                        a.data() + 1, pairwise.data() + 1, norms.data() + 1, rows, dim, ip
                    );
                    for (size_t i = 0; i < rows; ++i) {
                        for (size_t j = 0; j <= i; ++j) {
                            double expected = 0;
                            for (size_t k = 0; k < dim; ++k) {
                                const double x = a[1 + i * dim + k], y = a[1 + j * dim + k];
                                expected += ip ? -x * y : (x - y) * (x - y);
                            }
                            EXPECT_NEAR(pairwise[1 + i * rows + j], expected, tolerance);
                        }
                    }
                }
                for (const auto* output : {&product, &transposed, &norms, &pairwise}) {
                    EXPECT_EQ(output->front(), 12345);
                    EXPECT_EQ(output->back(), 12345);
                }
            }
        }
    }
}
}  // namespace
}  // namespace rabitqlib::simd
