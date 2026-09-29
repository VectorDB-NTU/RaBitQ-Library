#include "rabitqlib/simd/matrix_dispatch.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <utility>
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
using AccumulationBackend = std::pair<const char*, decltype(&accumulate_cluster_sums)>;

std::vector<AccumulationBackend> accumulation_backends() {
    std::vector<AccumulationBackend> backends{
        {"dispatch", accumulate_cluster_sums},
        {"generic", accumulate_cluster_sums_generic}};
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        backends.emplace_back("avx2", accumulate_cluster_sums_avx2);
    }
    if (cpu::has_avx512_core()) {
        backends.emplace_back("avx512", accumulate_cluster_sums_avx512);
    }
#endif
    return backends;
}

TEST(MatrixDispatchTest, IndexedAccumulationMatchesOrderedReferenceWithUnalignedTails) {
    constexpr size_t kPoints = 97;
    constexpr size_t kClusters = 7;
    const auto backends = accumulation_backends();
    for (const size_t dim :
         {1U,
          3U,
          4U,
          7U,
          8U,
          15U,
          16U,
          31U,
          32U,
          63U,
          64U,
          65U,
          127U,
          128U,
          129U,
          1536U,
          1537U}) {
        SCOPED_TRACE(dim);
        std::vector<float> data(kPoints * dim + 2, 12345.0F);
        std::vector<uint32_t> labels(kPoints + 2, 99);
        for (size_t point = 0; point < kPoints; ++point) {
            labels[point + 1] = static_cast<uint32_t>((point * 3) % kClusters);
        }
        for (const bool zero : {false, true}) {
            SCOPED_TRACE(zero);
            for (size_t index = 0; index < kPoints * dim; ++index) {
                data[index + 1] =
                    zero ? (index % 2 == 0 ? 0.0F : -0.0F)
                         : std::ldexp(
                               static_cast<float>(static_cast<int>(index % 37) - 18),
                               static_cast<int>(index % 21) - 10
                           );
            }
            for (const size_t stride : {1U, 3U, 100U}) {
                SCOPED_TRACE(stride);
                std::vector<size_t> point_ids;
                for (size_t point = 0; point < kPoints; point += stride) {
                    point_ids.push_back(point);
                }
                if (point_ids.size() > 2) {
                    point_ids.push_back(point_ids[1]);
                    std::reverse(point_ids.begin(), point_ids.end());
                }
                std::vector<double> initial(kClusters * dim + 2, 12345.0);
                std::vector<size_t> initial_counts(kClusters + 2, 12345);
                for (size_t index = 0; index < kClusters * dim; ++index) {
                    initial[index + 1] = zero ? (index % 2 == 0 ? 0.0 : -0.0)
                                              : static_cast<double>(index % 11) * 0.125;
                }
                for (size_t cluster = 0; cluster < kClusters; ++cluster) {
                    initial_counts[cluster + 1] = cluster + 2;
                }
                auto expected = initial;
                auto expected_counts = initial_counts;
                for (const size_t point : point_ids) {
                    const size_t cluster = labels[point + 1];
                    ++expected_counts[cluster + 1];
                    for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
                        expected[1 + cluster * dim + coordinate] +=
                            static_cast<double>(data[1 + point * dim + coordinate]);
                    }
                }
                for (const auto& backend : backends) {
                    SCOPED_TRACE(backend.first);
                    auto actual = initial;
                    auto actual_counts = initial_counts;
                    const size_t split = point_ids.size() / 2;
                    backend.second(
                        data.data() + 1,
                        point_ids.data(),
                        split,
                        dim,
                        labels.data() + 1,
                        actual.data() + 1,
                        actual_counts.data() + 1
                    );
                    backend.second(
                        data.data() + 1,
                        point_ids.data() + split,
                        point_ids.size() - split,
                        dim,
                        labels.data() + 1,
                        actual.data() + 1,
                        actual_counts.data() + 1
                    );
                    EXPECT_EQ(actual_counts, expected_counts);
                    EXPECT_EQ(
                        std::memcmp(
                            actual.data(), expected.data(), actual.size() * sizeof(double)
                        ),
                        0
                    );
                }
            }
        }
    }
}

TEST(MatrixDispatchTest, IndexedAccumulationPreservesPointOrderUnderCancellation) {
    constexpr size_t kDimension = 65;
    const std::vector<float> values{1e7F, 1e7F, 1e-10F, -1e7F, -1e7F, 1e-10F};
    const std::vector<uint32_t> labels{0, 1, 0, 1, 0, 1};
    std::vector<float> data(values.size() * kDimension);
    std::vector<size_t> point_ids(values.size());
    std::iota(point_ids.begin(), point_ids.end(), size_t{0});
    for (size_t point = 0; point < values.size(); ++point) {
        std::fill_n(data.data() + point * kDimension, kDimension, values[point]);
    }
    for (const auto& backend : accumulation_backends()) {
        SCOPED_TRACE(backend.first);
        std::vector<double> sums(2 * kDimension);
        std::vector<size_t> counts(2);
        backend.second(
            data.data(),
            point_ids.data(),
            3,
            kDimension,
            labels.data(),
            sums.data(),
            counts.data()
        );
        backend.second(
            data.data(),
            point_ids.data() + 3,
            3,
            kDimension,
            labels.data(),
            sums.data(),
            counts.data()
        );
        EXPECT_EQ(counts, (std::vector<size_t>{3, 3}));
        for (size_t coordinate = 0; coordinate < kDimension; ++coordinate) {
            EXPECT_DOUBLE_EQ(sums[coordinate], 0.0);
            EXPECT_DOUBLE_EQ(sums[kDimension + coordinate], static_cast<double>(1e-10F));
        }
    }
}

TEST(MatrixDispatchTest, EmptyIndexedAccumulationLeavesOutputsUntouched) {
    for (const auto& backend : accumulation_backends()) {
        SCOPED_TRACE(backend.first);
        double sum = -7.5;
        size_t count = 23;
        backend.second(nullptr, nullptr, 0, 65, nullptr, &sum, &count);
        EXPECT_DOUBLE_EQ(sum, -7.5);
        EXPECT_EQ(count, 23U);
    }
}

}  // namespace
}  // namespace rabitqlib::simd
