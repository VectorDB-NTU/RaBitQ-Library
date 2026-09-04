#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/utils/rotator.hpp"

TEST(FastScanDispatch, MatchesUnpackedRowsWithTailsAndWrapping) {
    std::mt19937 gen(42);
    for (size_t dim : {16, 64, 512, 1040, 4096}) {
        for (size_t count : {1, 31, 32, 33, 97}) {
            SCOPED_TRACE(::testing::Message() << "dim=" << dim << " count=" << count);
            const size_t batches = (count + 31) / 32;
            std::vector<uint8_t> rows(count * dim / 8), lut(dim * 4 + 1),
                packed(batches * dim * 4 + 1);
            for (auto& value : rows)
                value = static_cast<uint8_t>(gen());
            for (auto& value : lut)
                value = static_cast<uint8_t>(gen());
            rabitqlib::fastscan::pack_codes(dim, rows.data(), count, packed.data() + 1);
            for (size_t batch = 0; batch < batches; ++batch) {
                std::array<uint16_t, 34> actual{};
                actual.front() = actual.back() = 0xcafe;
                rabitqlib::fastscan::accumulate(
                    packed.data() + 1 + batch * dim * 4,
                    lut.data() + 1,
                    actual.data() + 1,
                    dim
                );
                for (size_t lane = 0; lane < 32; ++lane) {
                    const size_t row = batch * 32 + lane;
                    uint32_t expected = 0;
                    for (size_t col = 0; col < dim / 8; ++col) {
                        const uint8_t code = row < count ? rows[row * dim / 8 + col] : 0;
                        expected += lut[1 + col * 32 + (code >> 4)] +
                                    lut[1 + col * 32 + 16 + (code & 15)];
                    }
                    EXPECT_EQ(actual[lane + 1], static_cast<uint16_t>(expected));
                }
                EXPECT_EQ(actual.front(), 0xcafe);
                EXPECT_EQ(actual.back(), 0xcafe);
            }
        }
    }
}

TEST(HadamardBackend, MatchesIndependentSylvesterMatrix) {
    std::array<float, 512> original{}, actual{};
    for (size_t i = 0; i < original.size(); ++i)
        original[i] = static_cast<float>(i % 17) - 8;
    actual = original;
    helper_float_9(actual.data());
    for (size_t row = 0; row < actual.size(); ++row) {
        float expected = 0;
        for (size_t col = 0; col < original.size(); ++col) {
            size_t overlap = row & col;
            unsigned parity = 0;
            while (overlap) {
                parity ^= (overlap & 1);
                overlap >>= 1;
            }
            expected += (parity ? -1.0F : 1.0F) * original[col];
        }
        EXPECT_EQ(actual[row], expected);
    }
}
