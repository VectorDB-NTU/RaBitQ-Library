#include "rabitqlib/utils/tools.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rabitqlib {
namespace {

TEST(ToolsTest, FloorLog2BaseCasesAndPowersOfTwo) {
    EXPECT_EQ(floor_log2(0), 0U);
    EXPECT_EQ(floor_log2(1), 0U);
    EXPECT_EQ(floor_log2(2), 1U);
    EXPECT_EQ(floor_log2(3), 1U);
    EXPECT_EQ(floor_log2(4), 2U);
    EXPECT_EQ(floor_log2(7), 2U);
    EXPECT_EQ(floor_log2(8), 3U);
}

TEST(ToolsTest, CeilLog2BaseCasesAndSmallInputs) {
    EXPECT_EQ(ceil_log2(0), 0U);
    EXPECT_EQ(ceil_log2(1), 0U);
    EXPECT_EQ(ceil_log2(2), 1U);
    EXPECT_EQ(ceil_log2(3), 2U);
    EXPECT_EQ(ceil_log2(4), 2U);
    EXPECT_EQ(ceil_log2(5), 3U);
    EXPECT_EQ(ceil_log2(7), 3U);
    EXPECT_EQ(ceil_log2(8), 3U);
    EXPECT_EQ(ceil_log2(9), 4U);
}

TEST(ToolsTest, CeilLog2PowersOfTwoAndAdjacentBoundaries) {
    constexpr size_t digits = std::numeric_limits<size_t>::digits;
    for (size_t bit = 1; bit < digits; ++bit) {
        const size_t power = size_t{1} << bit;

        // Exactly at power of two: ceil_log2(2^k) == k
        EXPECT_EQ(ceil_log2(power), bit) << "Failed at 2^" << bit;

        // Just below power of two: ceil_log2(2^k - 1) == k
        EXPECT_EQ(ceil_log2(power - 1), bit) << "Failed at 2^" << bit << " - 1";

        // Just above power of two (if representable without overflow)
        if (bit + 1 < digits) {
            EXPECT_EQ(ceil_log2(power + 1), bit + 1)
                << "Failed at 2^" << bit << " + 1";
        }
    }
}

TEST(ToolsTest, CeilLog2WindowsLLP64Boundaries) {
    // Verify boundaries around 2^31 and 2^32, which previously triggered 32-bit
    // 1UL shift overflow/UB on Windows x64 (LLP64).
    constexpr size_t p31 = size_t{1} << 31;
    EXPECT_EQ(ceil_log2(p31 - 1), 31U);
    EXPECT_EQ(ceil_log2(p31), 31U);
    EXPECT_EQ(ceil_log2(p31 + 1), 32U);

    if constexpr (std::numeric_limits<size_t>::digits >= 64) {
        constexpr size_t p32 = size_t{1} << 32;
        EXPECT_EQ(ceil_log2(p32 - 1), 32U);
        EXPECT_EQ(ceil_log2(p32), 32U);
        EXPECT_EQ(ceil_log2(p32 + 1), 33U);

        constexpr size_t p33 = size_t{1} << 33;
        EXPECT_EQ(ceil_log2(p33 - 1), 33U);
        EXPECT_EQ(ceil_log2(p33), 33U);
        EXPECT_EQ(ceil_log2(p33 + 1), 34U);
    }
}

TEST(ToolsTest, CeilLog2HighestRepresentableSizeTValues) {
    constexpr size_t max_val = std::numeric_limits<size_t>::max();
    constexpr size_t digits = std::numeric_limits<size_t>::digits;
    constexpr size_t highest_power = size_t{1} << (digits - 1);

    EXPECT_EQ(ceil_log2(highest_power - 1), digits - 1);
    EXPECT_EQ(ceil_log2(highest_power), digits - 1);
    EXPECT_EQ(ceil_log2(highest_power + 1), digits);
    EXPECT_EQ(ceil_log2(max_val), digits);
}

}  // namespace
}  // namespace rabitqlib
