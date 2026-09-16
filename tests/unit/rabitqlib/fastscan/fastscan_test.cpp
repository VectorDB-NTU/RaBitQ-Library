#include "rabitqlib/fastscan/fastscan.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/lut.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/simd/estimator_dispatch.hpp"
#include "rabitqlib/simd/fastscan_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

namespace rabitqlib::fastscan {
namespace {

template <typename T>
void check_lut_subset_sums(void (*pack_fn)(size_t, const T*, T*) = pack_lut<T>) {
    const T large = std::numeric_limits<T>::max() / 4;
    const T tiny = std::numeric_limits<T>::denorm_min();
    const std::array<std::array<T, 4>, 5> patterns = {
        {{1, 2, 4, 8},
         {-0.0, -0.0, -0.0, -0.0},
         {large, tiny, -large, 1},
         {tiny, tiny, -tiny, -tiny},
         {-3.5, 0.25, 2, -0.125}}};
    for (size_t dim : {0U, 4U, 8U, 12U, 16U, 64U, 128U, 768U}) {
        for (size_t pattern = 0; pattern < patterns.size(); ++pattern) {
            SCOPED_TRACE(::testing::Message() << "dim=" << dim << " pattern=" << pattern);
            // Offset by one element to avoid requiring SIMD-aligned pointers.
            std::vector<T> query(dim + 1);
            for (size_t d = 0; d < dim; ++d) {
                query[d + 1] = patterns[(pattern + d / 4) % patterns.size()][d % 4];
            }
            const auto original = query;
            std::vector<T> expected(dim * 4 + 2, T{123});
            std::vector<T> actual = expected;
            for (size_t group = 0; group < dim / 4; ++group) {
                for (size_t mask = 0; mask < 16; ++mask) {
                    T sum = 0;
                    // Independently enumerate selected coordinates in input order.
                    for (size_t d = 0; d < 4; ++d) {
                        if ((mask & (8U >> d)) != 0) {
                            sum += query[1 + group * 4 + d];
                        }
                    }
                    expected[1 + group * 16 + mask] = sum;
                }
            }
            pack_fn(dim, query.data() + 1, actual.data() + 1);
            // Compare bits to detect signed-zero changes as well as rounding changes.
            EXPECT_EQ(
                std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)), 0
            );
            EXPECT_EQ(
                std::memcmp(query.data(), original.data(), query.size() * sizeof(T)), 0
            );
        }
    }
}

TEST(FastScanLutTest, MatchesOrderedSubsetSumsExactly) {
    check_lut_subset_sums<float>();
    check_lut_subset_sums<double>();
}

TEST(FastScanLutTest, EverySupportedBackendMatchesOrderedSubsetSumsExactly) {
    check_lut_subset_sums<float>(simd::pack_lut_generic);
    if (cpu::has_avx2()) {
        SCOPED_TRACE("AVX2");
        check_lut_subset_sums<float>(simd::pack_lut_avx2);
    }
    if (cpu::has_avx512_core()) {
        SCOPED_TRACE("AVX512");
        check_lut_subset_sums<float>(simd::pack_lut_avx512);
    }
}

TEST(FastScanPackingTest, MatchesBitReferenceIncludingTailsAndUnalignedBuffers) {
    for (size_t dim : {8U, 16U, 24U, 56U, 64U, 72U, 128U, 768U}) {
        for (size_t num = 0; num <= 65; ++num) {
            SCOPED_TRACE(::testing::Message() << "dim=" << dim << " num=" << num);
            std::vector<uint8_t> input(num * dim / 8 + 1);
            uint32_t state = 42;
            for (auto& value : input) {
                state = state * 1664525U + 1013904223U;
                value = static_cast<uint8_t>(state >> 24);
            }
            const auto original = input;
            const size_t bytes = ((num + 31) / 32) * 32 * (dim / 8);
            std::vector<uint8_t> expected(bytes + 2, 0);
            expected.front() = expected.back() = 0xA5;
            // Scatter individual sign bits to the documented FastScan lane layout.
            // This reference neither gathers columns nor uses the packing permutation.
            for (size_t row = 0; row < num; ++row) {
                const size_t lane = 2 * (row % 8) + (row % 16) / 8;
                for (size_t d = 0; d < dim; ++d) {
                    const uint8_t bit =
                        (input[1 + row * dim / 8 + d / 8] >> (7 - d % 8)) & 1;
                    const size_t offset = (row / 32) * (32 * dim / 8) + (d / 4) * 16 + lane;
                    expected[1 + offset] |= bit << (3 - d % 4 + 4 * ((row % 32) / 16));
                }
            }
            std::vector<uint8_t> actual(bytes + 2, 0xA5);
            pack_codes(dim, input.data() + 1, num, actual.data() + 1);
            EXPECT_EQ(actual, expected);
            EXPECT_EQ(input, original);
        }
    }
}

TEST(FastScanPackingTest, AccumulatesReferenceLutValuesOnEverySupportedBackend) {
    constexpr size_t kMaxDim = 768;
    for (size_t dim : {16U, 64U, 128U, 768U}) {
        for (size_t num : {1U, 17U, 31U, 32U}) {
            SCOPED_TRACE(::testing::Message() << "dim=" << dim << " num=" << num);
            std::vector<uint8_t> input(num * dim / 8);
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = static_cast<uint8_t>(i * 73 + i / 7);
            }
            std::vector<uint8_t> packed(32 * dim / 8);
            pack_codes(dim, input.data(), num, packed.data());
            std::vector<uint8_t> lut(dim * 4);
            std::vector<uint16_t> lut_hacc(dim * 4);
            for (size_t i = 0; i < lut.size(); ++i) {
                lut[i] = static_cast<uint8_t>((i * 7 + i / 16) % 32);
                lut_hacc[i] = static_cast<uint16_t>((i * 37 + i / 16) % 4096);
            }
            std::array<uint16_t, 32> expected{};
            std::array<int32_t, 32> expected_hacc{};
            for (size_t row = 0; row < 32; ++row) {
                for (size_t group = 0; group < dim / 4; ++group) {
                    const uint8_t byte = row < num ? input[row * dim / 8 + group / 2] : 0;
                    const size_t code = (byte >> (group % 2 == 0 ? 4 : 0)) & 15;
                    expected[row] += lut[group * 16 + code];
                    expected_hacc[row] += lut_hacc[group * 16 + code];
                }
            }
            auto check_backend = [&](auto accumulate_fn, auto transfer_fn, auto hacc_fn) {
                std::array<uint16_t, 32> actual{};
                std::array<int32_t, 32> actual_hacc{};
                alignas(64) std::array<uint8_t, kMaxDim * 8> packed_lut{};
                accumulate_fn(packed.data(), lut.data(), actual.data(), dim);
                transfer_fn(lut_hacc.data(), dim, packed_lut.data());
                hacc_fn(packed.data(), packed_lut.data(), actual_hacc.data(), dim);
                EXPECT_EQ(actual, expected);
                EXPECT_EQ(actual_hacc, expected_hacc);
            };
            if (cpu::has_avx2()) {
                SCOPED_TRACE("AVX2");
                check_backend(
                    simd::accumulate_avx2,
                    simd::transfer_lut_hacc_avx2,
                    simd::accumulate_hacc_avx2
                );
            }
            if (cpu::has_avx512_core()) {
                SCOPED_TRACE("AVX512");
                check_backend(
                    simd::accumulate_avx512,
                    simd::transfer_lut_hacc_avx512,
                    simd::accumulate_hacc_avx512
                );
            }
        }
    }
}

TEST(FastScanHighAccuracyTest, RejectsDimensionsThatCannotFillASimdBlock) {
    std::array<uint16_t, 16 * 3> lut{};
    std::array<uint8_t, 16 * 3 * 2> packed_lut{};
    std::array<uint8_t, 16 * 3> codes{};
    std::array<uint16_t, 32> low_result{};
    std::array<int32_t, 32> result{};

    for (size_t dim : {0U, 4U, 8U, 12U, 20U}) {
        EXPECT_THROW(
            accumulate(codes.data(), packed_lut.data(), low_result.data(), dim),
            std::invalid_argument
        );
        EXPECT_THROW(
            transfer_lut_hacc(lut.data(), dim, packed_lut.data()), std::invalid_argument
        );
        EXPECT_THROW(
            accumulate_hacc(codes.data(), packed_lut.data(), result.data(), dim),
            std::invalid_argument
        );
        std::vector<float> query(dim);
        EXPECT_THROW((Lut<float>(query.data(), dim, true)), std::invalid_argument);
        EXPECT_THROW((Lut<float>(query.data(), dim, false)), std::invalid_argument);
    }
}

TEST(FastScanHighAccuracyTest, Avx512TransferAcceptsUnalignedOutput) {
    if (!cpu::has_avx512_core()) {
        GTEST_SKIP() << "AVX512 is not supported on this CPU";
    }

    constexpr size_t kDim = 64;
    std::array<uint16_t, kDim * 4> lut{};
    for (size_t i = 0; i < lut.size(); ++i) {
        lut[i] = static_cast<uint16_t>((i * 977U) & 0xffffU);
    }

    std::array<uint8_t, kDim * 8> expected{};
    constexpr size_t kCodebooksPerRegister = 4;
    constexpr size_t kBytesPerRegisterPair = 128;
    constexpr size_t kBytesPerCodebook = 16;
    constexpr size_t kHighByteOffset = 64;
    for (size_t codebook = 0; codebook < kDim / 4; ++codebook) {
        const size_t low_offset =
            (codebook / kCodebooksPerRegister * kBytesPerRegisterPair) +
            (codebook % kCodebooksPerRegister * kBytesPerCodebook);
        for (size_t entry = 0; entry < 16; ++entry) {
            const uint16_t value = lut[codebook * 16 + entry];
            expected[low_offset + entry] = static_cast<uint8_t>(value);
            expected[low_offset + kHighByteOffset + entry] =
                static_cast<uint8_t>(value >> 8);
        }
    }

    alignas(16) std::array<uint8_t, (kDim * 8) + 1> storage{};
    ASSERT_NE(reinterpret_cast<uintptr_t>(storage.data() + 1) % 16, uintptr_t{0});
    simd::transfer_lut_hacc_avx512(lut.data(), kDim, storage.data() + 1);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), storage.begin() + 1));
}

TEST(BatchEstimatorTest, BackendsMatchScalarCorrectionAcrossChunksAndTailBatches) {
    if (!cpu::has_avx2() && !cpu::has_avx512_core()) {
        GTEST_SKIP() << "FastScan requires AVX2/FMA or AVX512";
    }
    using Estimator = decltype(&rabitqlib::simd::split_batch_estdist);
    std::vector<Estimator> backends{
        rabitqlib::simd::split_batch_estdist, rabitqlib::simd::split_batch_estdist_generic};
    if (cpu::has_avx2()) {
        backends.push_back(rabitqlib::simd::split_batch_estdist_avx2);
    }
    if (cpu::has_avx512_core()) {
        backends.push_back(rabitqlib::simd::split_batch_estdist_avx512);
    }
    for (size_t dim : {16U, 64U, 128U, 1024U, 1040U, 4096U}) {
        for (bool hacc : {false, true}) {
            for (auto metric : {METRIC_L2, METRIC_IP}) {
                SCOPED_TRACE(::testing::Message() << dim << " " << hacc << " " << metric);
                std::vector<float> query(dim, 1.0F);
                SplitBatchQuery<float> q(query.data(), dim, 3, metric, hacc);
                q.set_g_add(3.0F, 2.0F);
                // An unaligned batch with only 17 logical rows still has 32 physical lanes.
                std::vector<char> storage(BatchDataMap<float>::data_bytes(dim) + 1);
                BatchDataMap<float> batch(storage.data() + 1, dim);
                std::vector<uint8_t> codes(17 * dim / 8);
                for (size_t i = 0; i < codes.size(); ++i) {
                    codes[i] = static_cast<uint8_t>(i * 73 + i / 7);
                }
                pack_codes(dim, codes.data(), 17, batch.bin_code());
                for (size_t lane = 0; lane < kBatchSize; ++lane) {
                    batch.f_add()[lane] = static_cast<float>(lane) + 10.0F;
                    batch.f_rescale()[lane] = lane % 2 == 0 ? 0.125F : -0.25F;
                    batch.f_error()[lane] = static_cast<float>(lane) / 32.0F;
                }
                // With a constant-one query, LUT entries depend only on sign-bit count.
                std::array<int32_t, 5> lut{};
                const float inverse_delta = 1.0F / q.delta();
                for (size_t count = 0; count < lut.size(); ++count) {
                    lut[count] = static_cast<int32_t>(
                        std::nearbyint(static_cast<float>(count) * inverse_delta)
                    );
                }
                std::array<double, kBatchSize> expected_ip{}, expected_dist{},
                    expected_low{};
                for (size_t lane = 0; lane < kBatchSize; ++lane) {
                    int32_t sum = 0;
                    for (size_t group = 0; group < dim / 4; ++group) {
                        const uint8_t byte =
                            lane < 17 ? codes[lane * dim / 8 + group / 2] : 0;
                        const unsigned code = (byte >> (group % 2 == 0 ? 4 : 0)) & 15;
                        unsigned count = 0;
                        for (unsigned bit = 0; bit < 4; ++bit) {
                            count += (code >> bit) & 1U;
                        }
                        sum += lut[count];
                    }
                    expected_ip[lane] =
                        static_cast<double>(q.delta()) * sum + q.sum_vl_lut();
                    expected_dist[lane] = static_cast<float>(batch.f_add()[lane]) +
                                          q.g_add() +
                                          static_cast<float>(batch.f_rescale()[lane]) *
                                              (expected_ip[lane] + q.k1xsumq());
                    expected_low[lane] =
                        expected_dist[lane] -
                        static_cast<float>(batch.f_error()[lane]) * q.g_error();
                }
                for (auto backend : backends) {
                    std::array<float, kBatchSize + 2> dist, low, ip;
                    dist.fill(12345.0F);
                    low.fill(12345.0F);
                    ip.fill(12345.0F);
                    backend(
                        storage.data() + 1,
                        q,
                        dim,
                        dist.data() + 1,
                        low.data() + 1,
                        ip.data() + 1,
                        hacc
                    );
                    for (size_t lane = 0; lane < kBatchSize; ++lane) {
                        const double tolerance = 2e-6 * static_cast<double>(dim);
                        EXPECT_NEAR(ip[lane + 1], expected_ip[lane], tolerance);
                        EXPECT_NEAR(dist[lane + 1], expected_dist[lane], tolerance);
                        EXPECT_NEAR(low[lane + 1], expected_low[lane], tolerance);
                    }
                    for (const auto* output : {&dist, &low, &ip}) {
                        EXPECT_EQ(output->front(), 12345.0F);
                        EXPECT_EQ(output->back(), 12345.0F);
                    }
                }
            }
        }
    }
}

TEST(FastScanHighAccuracyTest, AccumulatesAcrossChunks) {
    if (!cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 is not supported on this CPU";
    }

    constexpr size_t kDim = 4096;
    std::vector<float> query(kDim, 1.0F);
    SplitBatchQuery<float> q_obj(query.data(), kDim, 3, METRIC_L2, true);

    std::vector<char> batch_data(BatchDataMap<float>::data_bytes(kDim));
    BatchDataMap<float> batch(batch_data.data(), kDim);
    std::fill_n(batch.bin_code(), kDim * fastscan::kBatchSize / 8, uint8_t{0xff});
    std::fill_n(batch.f_add(), fastscan::kBatchSize, 0.0F);
    std::fill_n(batch.f_rescale(), fastscan::kBatchSize, 1.0F);
    std::fill_n(batch.f_error(), fastscan::kBatchSize, 0.0F);

    const int32_t scalar_accumulator = int32_t{65535} * static_cast<int32_t>(kDim / 4);
    const float expected_ip =
        q_obj.delta() * static_cast<float>(scalar_accumulator) + q_obj.sum_vl_lut();
    const float expected_distance = expected_ip + q_obj.k1xsumq();

    std::array<float, fastscan::kBatchSize> estimated{};
    std::array<float, fastscan::kBatchSize> lower{};
    std::array<float, fastscan::kBatchSize> inner_products{};
    split_batch_estdist(
        batch_data.data(),
        q_obj,
        kDim,
        estimated.data(),
        lower.data(),
        inner_products.data(),
        true
    );

    for (size_t lane = 0; lane < fastscan::kBatchSize; ++lane) {
        EXPECT_TRUE(std::isfinite(estimated[lane]));
        EXPECT_FLOAT_EQ(inner_products[lane], expected_ip);
        EXPECT_FLOAT_EQ(estimated[lane], expected_distance);
        EXPECT_FLOAT_EQ(lower[lane], expected_distance);
    }
}

}  // namespace
}  // namespace rabitqlib::fastscan
