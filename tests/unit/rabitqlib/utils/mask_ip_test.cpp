#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/space.hpp"

TEST(MaskIpX0Q, BackendsMatchScalarAcrossBlocksAndAlignments) {
    using namespace rabitqlib;
    using Function = float (*)(const float*, const uint8_t*, size_t);
    std::vector<Function> functions{
        static_cast<Function>(simd::mask_ip_x0_q_generic),
        static_cast<Function>(mask_ip_x0_q)};
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        functions.push_back(static_cast<Function>(simd::mask_ip_x0_q_avx2));
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx512_core()) {
        functions.push_back(static_cast<Function>(simd::mask_ip_x0_q_avx512));
    }
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    if (cpu::has_neon()) {
        functions.push_back(static_cast<Function>(simd::mask_ip_x0_q_neon));
    }
#endif

    for (size_t dim : {0, 64, 128, 192, 256, 448, 576, 1024, 4096}) {
        SCOPED_TRACE(dim);
        for (size_t pattern = 0; pattern < 5; ++pattern) {
            SCOPED_TRACE(pattern);
            std::vector<float> query(dim + 1);
            std::vector<uint64_t> words(dim / 64, 0);
            double expected = 0;
            double sum_abs = 0;
            for (size_t i = 0; i < dim; ++i) {
                query[i + 1] = static_cast<float>(static_cast<int>(i % 23) - 11) / 7.0F;
                if (pattern == 4) {
                    query[i + 1] = (i % 2 == 0) ? 65536.0F : -65535.0F;
                }
                const bool selected = pattern == 1 || pattern == 4 ||
                                      (pattern == 2 && i % 3 == 0) ||
                                      (pattern == 3 && (i % 32 == 0 || i % 32 == 31));
                if (selected) {
                    words[i / 64] |= uint64_t{1} << (63 - i % 64);
                    expected += query[i + 1];
                    sum_abs += std::abs(static_cast<double>(query[i + 1]));
                }
            }
            for (size_t offset : {0, 1, 3, 7}) {
                SCOPED_TRACE(offset);
                std::vector<uint8_t> storage(dim / 8 + 8, 0);
                if (dim != 0) {
                    std::memcpy(storage.data() + offset, words.data(), dim / 8);
                }
                const auto* codes = storage.data() + offset;
                for (auto function : functions) {
                    const float result = function(query.data() + 1, codes, dim);
                    EXPECT_NEAR(result, expected, 2e-6 * std::max(1.0, sum_abs));
                    if (pattern == 0 || pattern == 4) {
                        EXPECT_EQ(result, expected);
                    }
                }
            }
        }
    }
}

TEST(MaskIpX0Q, DispatchPreservesEveryStoredBitPosition) {
    using namespace rabitqlib;
    constexpr size_t dim = 192;
    std::vector<float> query(dim);
    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(i + 1);
    }
    for (size_t bit = 0; bit < dim; ++bit) {
        SCOPED_TRACE(bit);
        std::vector<uint64_t> words(dim / 64, 0);
        words[bit / 64] = uint64_t{1} << (63 - bit % 64);
        EXPECT_EQ(mask_ip_x0_q(query.data(), words.data(), dim), query[bit]);
    }
}
