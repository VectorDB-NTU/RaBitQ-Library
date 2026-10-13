#include "rabitqlib/fastscan/encoder.hpp"  // IWYU pragma: keep (public header coverage)

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/utils/openmp.hpp"

namespace rabitqlib::quant {
namespace {
template <typename T, bool Parallel>
void check_one_bit_batches() {
    // Alternate dimensions to exercise scratch growth and shrinking on each thread.
    for (size_t dim : {64U, 8U, 128U, 16U, 768U, 24U, 192U, 56U, 72U, 64U}) {
        for (size_t num : {0U, 1U, 15U, 16U, 17U, 31U, 32U, 33U, 64U, 65U}) {
            for (MetricType metric : {METRIC_L2, METRIC_IP}) {
                SCOPED_TRACE(
                    ::testing::Message()
                    << "dim=" << dim << " num=" << num << " metric=" << metric
                );
                std::vector<T> centroid(dim);
                std::vector<T> data(num * dim);
                for (size_t d = 0; d < dim; ++d) {
                    centroid[d] = static_cast<T>(static_cast<int>(d % 7) - 3) / 8;
                    for (size_t i = 0; i < num; ++i) {
                        const T residual =
                            i % 3 == 0
                                ? T{0}
                                : static_cast<T>(static_cast<int>((i + d) % 11) - 5) / 4;
                        data[i * dim + d] = centroid[d] + residual;
                    }
                }
                // A guard byte/factor also verifies empty inputs and tail writes.
                const size_t packed_bytes = ((num + 31) / 32) * 32 * (dim / 8);
                std::vector<uint8_t> expected(packed_bytes + 1, 0xA5);
                std::vector<uint8_t> actual = expected;
                std::vector<uint8_t> compact(num * dim / 8);
                std::vector<T> expected_add(num + 1, T{123});
                std::vector<T> expected_rescale = expected_add;
                std::vector<T> expected_error = expected_add;
                std::vector<T> actual_add = expected_add;
                std::vector<T> actual_rescale = expected_add;
                std::vector<T> actual_error = expected_add;
                for (size_t i = 0; i < num; ++i) {
                    rabitq_impl::one_bit::one_bit_compact_code(
                        data.data() + i * dim,
                        centroid.data(),
                        dim,
                        compact.data() + i * dim / 8,
                        expected_add[i],
                        expected_rescale[i],
                        expected_error[i],
                        metric
                    );
                }
                fastscan::pack_codes(dim, compact.data(), num, expected.data());
                rabitq_impl::one_bit::one_bit_batch_code<T, Parallel>(
                    data.data(),
                    centroid.data(),
                    num,
                    dim,
                    actual.data(),
                    actual_add.data(),
                    actual_rescale.data(),
                    actual_error.data(),
                    metric
                );
                EXPECT_EQ(actual, expected);
                EXPECT_EQ(actual_add, expected_add);
                EXPECT_EQ(actual_rescale, expected_rescale);
                EXPECT_EQ(actual_error, expected_error);
            }
        }
    }
}

TEST(RabitqOneBitBatchTest, MatchesIndividualEncoding) {
    check_one_bit_batches<float, false>();
    check_one_bit_batches<double, false>();
}

TEST(RabitqOneBitBatchTest, ParallelMatchesIndividualEncoding) {
#if RABITQ_USE_OPENMP
    const int previous_threads = omp_get_max_threads();
    omp_set_num_threads(2);
#endif
    check_one_bit_batches<float, true>();
    check_one_bit_batches<double, true>();
#if RABITQ_USE_OPENMP
    omp_set_num_threads(previous_threads);
#endif
}

}  // namespace
}  // namespace rabitqlib::quant
