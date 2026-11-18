#include "rabitqlib/utils/tools.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>

namespace rabitqlib {
namespace {

TEST(ToolsTest, CeilLog2RoundsUpAcrossSizeTWidth) {
    EXPECT_EQ(ceil_log2(0), 0U);
    EXPECT_EQ(ceil_log2(1), 0U);
    EXPECT_EQ(ceil_log2(3), 2U);

    // Exponents of 32 and above exceed the width of a 32-bit shift operand
    // (unsigned long on LLP64).
    constexpr size_t kDigits = std::numeric_limits<size_t>::digits;
    for (size_t bit = 1; bit < kDigits; ++bit) {
        const size_t power = size_t{1} << bit;
        EXPECT_EQ(ceil_log2(power), bit) << bit;
        EXPECT_EQ(ceil_log2(power + 1), bit + 1) << bit;
    }
    EXPECT_EQ(ceil_log2(std::numeric_limits<size_t>::max()), kDigits);
}

}  // namespace
}  // namespace rabitqlib
