#include "rabitqlib/fastscan/data_layout.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/fastscan/encoder.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"

namespace rabitqlib {
namespace {
float load_float(const char* data) {
    float value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

TEST(FastScanDataLayoutTest, QgBatchSizeRejectsOverflow) {
    const size_t max_dim = std::numeric_limits<size_t>::max() / fastscan::kBatchSize;
    EXPECT_THROW(QGBatchDataMap<float>::data_bytes(max_dim + 1), std::length_error);
    EXPECT_THROW(ConstQGBatchDataMap<float>::data_bytes(max_dim + 1), std::length_error);
    EXPECT_EQ(
        QGBatchDataMap<float>::data_bytes(max_dim),
        max_dim * fastscan::kBatchSize / 8 + sizeof(float) * fastscan::kBatchSize * 2
    );
}

TEST(FastScanDataLayoutTest, BatchFactorsKeepPackedOffsetsOnUnalignedStorage) {
    constexpr size_t kDim = 64;
    constexpr size_t kCodeBytes = kDim * fastscan::kBatchSize / 8;
    constexpr size_t kDataBytes = kCodeBytes + (3 * fastscan::kBatchSize * sizeof(float));
    const size_t data_bytes = BatchDataMap<float>::data_bytes(kDim);
    ASSERT_EQ(data_bytes, kCodeBytes + (3 * fastscan::kBatchSize * sizeof(float)));

    alignas(float) std::array<char, kDataBytes + 2> storage;
    storage.fill(char{0x5a});
    char* data = storage.data() + 1;
    ASSERT_NE(
        reinterpret_cast<uintptr_t>(data + kCodeBytes) % alignof(float), uintptr_t{0}
    );
    BatchDataMap<float> batch(data, kDim);
    batch.f_add()[0] = 1.25F;
    batch.f_rescale()[0] = -2.5F;
    batch.f_error()[fastscan::kBatchSize - 1] = 3.75F;

    ConstBatchDataMap<float> stored(data, kDim);
    EXPECT_FLOAT_EQ(stored.f_add()[0], 1.25F);
    EXPECT_FLOAT_EQ(stored.f_rescale()[0], -2.5F);
    EXPECT_FLOAT_EQ(stored.f_error()[fastscan::kBatchSize - 1], 3.75F);
    EXPECT_FLOAT_EQ(load_float(data + kCodeBytes), 1.25F);
    EXPECT_FLOAT_EQ(
        load_float(data + kCodeBytes + (fastscan::kBatchSize * sizeof(float))), -2.5F
    );
    EXPECT_FLOAT_EQ(
        load_float(data + kCodeBytes + ((3 * fastscan::kBatchSize - 1) * sizeof(float))),
        3.75F
    );
    EXPECT_EQ(storage.front(), char{0x5a});
    EXPECT_EQ(storage.back(), char{0x5a});
}

TEST(FastScanDataLayoutTest, QgBatchFactorsKeepPackedOffsetsOnUnalignedStorage) {
    constexpr size_t kDim = 64;
    constexpr size_t kCodeBytes = kDim * fastscan::kBatchSize / 8;
    constexpr size_t kDataBytes = kCodeBytes + (2 * fastscan::kBatchSize * sizeof(float));
    const size_t data_bytes = QGBatchDataMap<float>::data_bytes(kDim);
    ASSERT_EQ(data_bytes, kCodeBytes + (2 * fastscan::kBatchSize * sizeof(float)));

    alignas(float) std::array<char, kDataBytes + 2> storage{};
    char* data = storage.data() + 1;
    ASSERT_NE(
        reinterpret_cast<uintptr_t>(data + kCodeBytes) % alignof(float), uintptr_t{0}
    );
    QGBatchDataMap<float> batch(data, kDim);
    std::fill_n(batch.f_add(), fastscan::kBatchSize, 4.25F);
    std::fill_n(batch.f_rescale(), fastscan::kBatchSize, -5.5F);

    ConstQGBatchDataMap<float> stored(data, kDim);
    EXPECT_FLOAT_EQ(stored.f_add()[fastscan::kBatchSize - 1], 4.25F);
    EXPECT_FLOAT_EQ(stored.f_rescale()[fastscan::kBatchSize - 1], -5.5F);
    EXPECT_FLOAT_EQ(load_float(data + kCodeBytes), 4.25F);
    EXPECT_FLOAT_EQ(
        load_float(data + kCodeBytes + (fastscan::kBatchSize * sizeof(float))), -5.5F
    );
}

template <typename T>
void check_qg_encoding() {
    for (size_t dim : {32U, 64U, 96U}) {
        for (size_t count : {1U, 17U, 32U}) {
            for (MetricType metric : {METRIC_L2, METRIC_IP}) {
                SCOPED_TRACE(::testing::Message() << dim << ',' << count << ',' << metric);
                std::vector<T> data(count * dim), centroid(dim);
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i] = static_cast<T>(static_cast<int>(i % 19) - 9) / T{11};
                }
                for (size_t i = 0; i < dim; ++i) {
                    centroid[i] = static_cast<T>(static_cast<int>(i % 7) - 3) / T{13};
                }
                for (bool zero_centroid : {false, true}) {
                    if (zero_centroid) {
                        std::fill(centroid.begin(), centroid.end(), T{0});
                    }
                    // The first record has zero residual, including the sign convention.
                    std::copy(centroid.begin(), centroid.end(), data.begin());
                    std::vector<char> reference(BatchDataMap<T>::data_bytes(dim), 0);
                    const size_t bytes = QGBatchDataMap<T>::data_bytes(dim);
                    std::vector<char> compact(bytes + 2, char{0x5a});
                    quant::quantize_one_batch(
                        data.data(), centroid.data(), count, dim, reference.data(), metric
                    );
                    if (zero_centroid) {
                        quant::quantize_qg_batch(
                            data.data(), count, dim, compact.data() + 1, metric
                        );
                    } else {
                        quant::quantize_qg_batch(
                            data.data(),
                            centroid.data(),
                            count,
                            dim,
                            compact.data() + 1,
                            metric
                        );
                    }
                    const ConstBatchDataMap<T> expected(reference.data(), dim);
                    const ConstQGBatchDataMap<T> actual(compact.data() + 1, dim);
                    EXPECT_EQ(
                        std::memcmp(
                            actual.bin_code(),
                            expected.bin_code(),
                            dim * fastscan::kBatchSize / 8
                        ),
                        0
                    );
                    for (size_t row = 0; row < count; ++row) {
                        EXPECT_EQ(actual.f_add()[row], expected.f_add()[row]);
                        EXPECT_EQ(actual.f_rescale()[row], expected.f_rescale()[row]);
                    }
                    EXPECT_EQ(compact.front(), char{0x5a});
                    EXPECT_EQ(compact.back(), char{0x5a});
                }
            }
        }
    }
}

TEST(FastScanQgBatchEncodingTest, PreservesCodesAndFactorsWithExplicitAndZeroCentroids) {
    check_qg_encoding<float>();
    check_qg_encoding<double>();
}

}  // namespace
}  // namespace rabitqlib
