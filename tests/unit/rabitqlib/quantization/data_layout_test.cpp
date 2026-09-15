#include "rabitqlib/quantization/data_layout.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/quantization/rabitq.hpp"

namespace rabitqlib {
namespace {

float load_float(const char* data) {
    float value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

TEST(DataLayoutTest, BatchFactorsKeepPackedOffsetsOnUnalignedStorage) {
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

TEST(DataLayoutTest, QgBatchFactorsKeepPackedOffsetsOnUnalignedStorage) {
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

TEST(DataLayoutTest, SingleCodeFactorsKeepPackedOffsetsOnUnalignedStorage) {
    constexpr size_t kDim = 64;
    constexpr size_t kBits = 3;
    constexpr size_t kCodeBytes = kDim * kBits / 8;

    constexpr size_t kExBytes = kCodeBytes + (2 * sizeof(float));
    alignas(float) std::array<char, kExBytes + 2> ex_storage{};
    char* ex_data = ex_storage.data() + 1;
    ASSERT_NE(
        reinterpret_cast<uintptr_t>(ex_data + kCodeBytes) % alignof(float), uintptr_t{0}
    );
    ExDataMap<float> ex(ex_data, kDim, kBits);
    ex.f_add_ex() = 6.25F;
    ex.f_rescale_ex() = -7.5F;
    ConstExDataMap<float> stored_ex(ex_data, kDim, kBits);
    EXPECT_FLOAT_EQ(stored_ex.f_add_ex(), 6.25F);
    EXPECT_FLOAT_EQ(stored_ex.f_rescale_ex(), -7.5F);
    EXPECT_FLOAT_EQ(load_float(ex_data + kCodeBytes), 6.25F);
    EXPECT_FLOAT_EQ(load_float(ex_data + kCodeBytes + sizeof(float)), -7.5F);

    constexpr size_t kBaseBytes = kCodeBytes + (3 * sizeof(float));
    alignas(float) std::array<char, kBaseBytes + 2> base_storage{};
    char* base_data = base_storage.data() + 1;
    ASSERT_NE(
        reinterpret_cast<uintptr_t>(base_data + kCodeBytes) % alignof(float), uintptr_t{0}
    );
    BaseDataMap<float> base(base_data, kDim, kBits);
    base.f_add() = 8.25F;
    base.f_rescale() = -9.5F;
    base.f_error() = 10.75F;
    ConstBaseDataMap<float> stored_base(base_data, kDim, kBits);
    EXPECT_FLOAT_EQ(stored_base.f_add(), 8.25F);
    EXPECT_FLOAT_EQ(stored_base.f_rescale(), -9.5F);
    EXPECT_FLOAT_EQ(stored_base.f_error(), 10.75F);
    EXPECT_FLOAT_EQ(load_float(base_data + kCodeBytes), 8.25F);
    EXPECT_FLOAT_EQ(load_float(base_data + kCodeBytes + sizeof(float)), -9.5F);
    EXPECT_FLOAT_EQ(load_float(base_data + kCodeBytes + (2 * sizeof(float))), 10.75F);

    constexpr size_t kBinCodeBytes = kDim / 8;
    constexpr size_t kBinBytes = kBinCodeBytes + (3 * sizeof(float));
    alignas(uint64_t) std::array<char, kBinBytes + 2> bin_storage{};
    char* bin_data = bin_storage.data() + 1;
    ASSERT_NE(reinterpret_cast<uintptr_t>(bin_data) % alignof(uint64_t), uintptr_t{0});
    ASSERT_NE(
        reinterpret_cast<uintptr_t>(bin_data + kBinCodeBytes) % alignof(float), uintptr_t{0}
    );
    BinDataMap<float> bin(bin_data, kDim);
    static_assert(std::is_same_v<decltype(bin.bin_code()), uint8_t*>);
    bin.bin_code()[0] = 0xa5;
    bin.f_add() = 11.25F;
    bin.f_rescale() = -12.5F;
    bin.f_error() = 13.75F;
    ConstBinDataMap<float> stored_bin(bin_data, kDim);
    static_assert(std::is_same_v<decltype(stored_bin.bin_code()), const uint8_t*>);
    EXPECT_EQ(stored_bin.bin_code()[0], uint8_t{0xa5});
    EXPECT_FLOAT_EQ(stored_bin.f_add(), 11.25F);
    EXPECT_FLOAT_EQ(stored_bin.f_rescale(), -12.5F);
    EXPECT_FLOAT_EQ(stored_bin.f_error(), 13.75F);
    EXPECT_FLOAT_EQ(load_float(bin_data + kBinCodeBytes), 11.25F);
    EXPECT_FLOAT_EQ(load_float(bin_data + kBinCodeBytes + sizeof(float)), -12.5F);
    EXPECT_FLOAT_EQ(load_float(bin_data + kBinCodeBytes + (2 * sizeof(float))), 13.75F);
}

TEST(DataLayoutTest, UnalignedSingleCodePreservesUint64PackedRepresentation) {
    constexpr size_t kDim = 64;
    constexpr size_t kCodeBytes = kDim / 8;
    constexpr size_t kDataBytes = kCodeBytes + (3 * sizeof(float));
    std::array<float, kDim> data{};
    std::array<float, kDim> centroid{};
    for (size_t i = 0; i < kDim; ++i) {
        data[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
        centroid[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.25F;
    }

    uint64_t expected_code = 0;
    float expected_add = 0;
    float expected_rescale = 0;
    float expected_error = 0;
    quant::quantize_compact_one_bit(
        data.data(),
        centroid.data(),
        kDim,
        &expected_code,
        expected_add,
        expected_rescale,
        expected_error
    );

    alignas(uint64_t) std::array<char, kDataBytes + 1> storage{};
    char* packed = storage.data() + 1;
    ASSERT_NE(reinterpret_cast<uintptr_t>(packed) % alignof(uint64_t), uintptr_t{0});
    quant::quantize_compact_one_bit(data.data(), centroid.data(), kDim, packed);

    ConstBinDataMap<float> stored(packed, kDim);
    EXPECT_EQ(std::memcmp(stored.bin_code(), &expected_code, sizeof(expected_code)), 0);
    EXPECT_FLOAT_EQ(stored.f_add(), expected_add);
    EXPECT_FLOAT_EQ(stored.f_rescale(), expected_rescale);
    EXPECT_FLOAT_EQ(stored.f_error(), expected_error);
}

}  // namespace
}  // namespace rabitqlib
