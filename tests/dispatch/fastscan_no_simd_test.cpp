#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

// Link-time substitutes, isolated from both production detection and the main suite.
namespace rabitqlib::cpu {
bool has_avx2() { return false; }
bool has_neon() { return false; }
bool has_avx512_core() { return false; }
bool has_avx512_popcnt() { return false; }
}  // namespace rabitqlib::cpu

namespace rabitqlib::fastscan {
namespace {

TEST(FastScanDispatchTest, UnsupportedCpuThrowsWithoutAccessingInputsOrOutput) {
    std::array<int32_t, kBatchSize> result;
    result.fill(-123);
    const auto original = result;
    try {
        accumulate(nullptr, nullptr, result.data(), 16);
        FAIL() << "Standard FastScan must reject an unsupported SIMD backend";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(
            error.what(),
            "Standard FastScan accumulation requires AVX2/FMA, AVX-512, or ARM NEON; "
            "no supported SIMD backend is available"
        );
    }
    EXPECT_EQ(result, original);
}

TEST(FastScanDispatchTest, InvalidDimensionStillRaisesInvalidArgument) {
    EXPECT_THROW(accumulate(nullptr, nullptr, nullptr, 0), std::invalid_argument);
    EXPECT_THROW(accumulate(nullptr, nullptr, nullptr, 15), std::invalid_argument);
}

TEST(FastScanDispatchTest, HighAccuracyUsesScalarFallbackWithoutSimd) {
    constexpr size_t kDim = 16;
    std::array<uint8_t, kDim * 4> codes{};
    std::array<uint16_t, kDim * 4> lut;
    lut.fill(1000);
    std::array<uint8_t, kDim * 8> high_accuracy_lut{};
    transfer_lut_hacc(lut.data(), kDim, high_accuracy_lut.data());
    std::array<int32_t, kBatchSize> result{};
    accumulate_hacc(codes.data(), high_accuracy_lut.data(), result.data(), kDim);
    for (const auto value : result) {
        EXPECT_EQ(value, 4000);
    }
}

}  // namespace
}  // namespace rabitqlib::fastscan
