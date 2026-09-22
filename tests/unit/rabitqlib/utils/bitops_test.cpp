#include "rabitqlib/utils/bitops.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace rabitqlib::bitops {
namespace {

TEST(BitopsTest, CountsBitsAndTrailingZeros) {
    EXPECT_EQ(popcount32(0), 0U);
    EXPECT_EQ(popcount32(0xFFFFFFFFU), 32U);
    EXPECT_EQ(popcount32(0x80000001U), 2U);
    EXPECT_EQ(popcount64(0), 0U);
    EXPECT_EQ(popcount64(0xFFFFFFFFFFFFFFFFULL), 64U);
    EXPECT_EQ(popcount64(0x8000000000000001ULL), 2U);
    for (unsigned bit = 0; bit < 32; ++bit) {
        EXPECT_EQ(countr_zero32(uint32_t{1} << bit), bit);
    }
}

}  // namespace
}  // namespace rabitqlib::bitops
