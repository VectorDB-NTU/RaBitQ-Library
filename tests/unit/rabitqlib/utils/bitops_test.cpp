#include "rabitqlib/utils/bitops.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rabitqlib::bitops {
namespace {

TEST(BitopsTest, CountsBitsAndTrailingZeros) {
    EXPECT_EQ(popcount32(0), 0U);
    EXPECT_EQ(popcount32(0xFFFFFFFFU), 32U);
    EXPECT_EQ(popcount32(0x80000001U), 2U);
    for (unsigned bit = 0; bit < 32; ++bit) {
        EXPECT_EQ(countr_zero32(uint32_t{1} << bit), bit);
    }
}

TEST(BitopsTest, Popcount64MatchesBitReference) {
    const auto check = [](uint64_t value) {
        unsigned expected = 0;
        for (size_t bit = 0; bit < 64; ++bit) {
            expected += static_cast<unsigned>((value >> bit) & 1U);
        }
        EXPECT_EQ(popcount64(value), expected) << value;
    };
    check(0);
    check(std::numeric_limits<uint64_t>::max());
    check(0x8000000000000001ULL);
    for (size_t bit = 0; bit < 64; ++bit) {
        check(uint64_t{1} << bit);
        check(~(uint64_t{1} << bit));
    }
    uint64_t value = 17;
    for (size_t i = 0; i < 1024; ++i) {
        value = value * 6364136223846793005ULL + 1442695040888963407ULL;
        check(value);
    }
}

TEST(BitopsTest, CompactWordLoadsZeroExtendWithoutReadingPastInput) {
    for (size_t bytes = 1; bytes <= 8; ++bytes) {
        for (size_t offset : {0U, 1U}) {
            SCOPED_TRACE(bytes);
            SCOPED_TRACE(offset);
            std::vector<uint8_t> storage(bytes + offset);
            uint64_t expected = 0;
            for (size_t i = 0; i < bytes; ++i) {
                const auto value = static_cast<uint8_t>(0x80U + i * 17U);
                storage[offset + i] = value;
                expected |= uint64_t{value} << (8 * i);
            }
            EXPECT_EQ(load_word(storage.data() + offset, bytes), expected);
        }
    }
}

}  // namespace
}  // namespace rabitqlib::bitops
