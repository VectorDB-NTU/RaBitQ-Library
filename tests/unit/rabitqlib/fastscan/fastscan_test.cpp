#include "rabitqlib/fastscan/fastscan.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

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

}  // namespace
}  // namespace rabitqlib::fastscan
