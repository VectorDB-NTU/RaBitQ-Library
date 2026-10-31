#include "rabitqlib/utils/space.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "rabitqlib/core/detail/simd/space_dispatch.hpp"
#include "rabitqlib/core/detail/simd/warmup_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/warmup_space.hpp"

using namespace rabitqlib;

TEST(SpaceHelpersTest, CentroidSupportsRequestedWorkerCounts) {
    const std::array<float, 12> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    for (size_t workers : {0U, 1U, 2U, 64U}) {
        const auto centroid = compute_centroid(data.data(), 4, 3, workers);
        ASSERT_EQ(centroid.size(), 3U);
        EXPECT_FLOAT_EQ(centroid[0], 5.5F);
        EXPECT_FLOAT_EQ(centroid[1], 6.5F);
        EXPECT_FLOAT_EQ(centroid[2], 7.5F);
    }
}

TEST(PackBinaryTest, SupportsUnalignedOutput) {
    constexpr size_t kDim = 128;
    std::array<int, kDim> binary_code{};
    for (size_t i = 0; i < kDim; ++i) {
        binary_code[i] = (i % 3) == 0;
    }

    alignas(uint64_t) std::array<uint8_t, (2 * sizeof(uint64_t)) + 1> storage{};
    auto* output = storage.data() + 1;
    ASSERT_NE(reinterpret_cast<uintptr_t>(output) % alignof(uint64_t), 0U);
    pack_binary_to_bytes<uint64_t>(binary_code.data(), output, kDim);

    const std::array<uint64_t, 2> packed{
        load_unaligned_u64(storage.data() + 1),
        load_unaligned_u64(storage.data() + 1 + sizeof(uint64_t)),
    };
    for (size_t i = 0; i < kDim; ++i) {
        const size_t word = i / 64;
        const auto bit = static_cast<int>((packed[word] >> (63 - (i % 64))) & 1U);
        EXPECT_EQ(bit, binary_code[i]) << "bit " << i;
    }
}

TEST(PackBinaryTest, AllWordSizesPreserveFullAndPartialWordLayout) {
    const auto check = [](auto word_type) {
        using Word = decltype(word_type);
        constexpr size_t kWordBits = sizeof(Word) * 8;
        for (size_t dim : {8U, 16U, 24U, 32U, 48U, 64U, 80U, 96U, 112U, 128U}) {
            SCOPED_TRACE(kWordBits);
            SCOPED_TRACE(dim);
            std::vector<int> signs(dim);
            // All zeros, all ones, and each isolated bit catch order and tail errors.
            for (size_t pattern = 0; pattern < dim + 2; ++pattern) {
                for (size_t i = 0; i < dim; ++i) {
                    signs[i] = pattern == dim + 1 || i == pattern;
                }
                std::vector<uint8_t> output(dim / 8 + 2, 0xA5);
                pack_binary_to_bytes<Word>(signs.data(), output.data() + 1, dim);
                EXPECT_EQ(output.front(), 0xA5);
                EXPECT_EQ(output.back(), 0xA5);
                for (size_t start = 0; start < dim; start += kWordBits) {
                    const size_t width = std::min(kWordBits, dim - start);
                    Word actual = 0;
                    std::memcpy(&actual, output.data() + 1 + start / 8, width / 8);
                    for (size_t bit = 0; bit < width; ++bit) {
                        EXPECT_EQ((actual >> (width - 1 - bit)) & 1U, signs[start + bit]);
                    }
                }
            }
        }
    };
    check(uint8_t{});
    check(uint16_t{});
    check(uint32_t{});
    check(uint64_t{});
}

TEST(MaskIpX0QTest, SupportsUnalignedCodes) {
    constexpr size_t kDim = 128;
    std::array<int, kDim> binary_code{};
    std::array<float, kDim> query{};
    float expected = 0;
    for (size_t i = 0; i < kDim; ++i) {
        binary_code[i] = (i % 3) == 0;
        query[i] = static_cast<float>(i + 1);
        expected += binary_code[i] != 0 ? query[i] : 0;
    }

    alignas(uint64_t) std::array<uint8_t, (2 * sizeof(uint64_t)) + 1> storage{};
    auto* codes = storage.data() + 1;
    ASSERT_NE(reinterpret_cast<uintptr_t>(codes) % alignof(uint64_t), 0U);
    pack_binary_to_bytes<uint64_t>(binary_code.data(), codes, kDim);

#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        EXPECT_FLOAT_EQ(simd::mask_ip_x0_q_avx2(query.data(), codes, kDim), expected);
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx512_core()) {
        EXPECT_FLOAT_EQ(simd::mask_ip_x0_q_avx512(query.data(), codes, kDim), expected);
    }
#endif
    EXPECT_FLOAT_EQ(mask_ip_x0_q(query.data(), codes, kDim), expected);
}

TEST(WarmupIpX0QTest, SupportsUnalignedCodes) {
    constexpr float kDelta = 0.5F;
    constexpr float kVl = -0.25F;
    for (size_t dim :
         {0UL,
          64UL,
          96UL,
          128UL,
          160UL,
          192UL,
          224UL,
          256UL,
          288UL,
          448UL,
          512UL,
          544UL,
          576UL}) {
        SCOPED_TRACE(dim);
        std::vector<int> data_bits(dim);
        size_t data_popcount = 0;
        for (size_t i = 0; i < dim; ++i) {
            data_bits[i] = (i % 3) == 0;
            data_popcount += data_bits[i] != 0;
        }

        std::vector<uint8_t> storage((dim / 8) + 1);
        auto* codes = storage.data() + 1;
        ASSERT_NE(reinterpret_cast<uintptr_t>(codes) % alignof(uint64_t), 0U);
        pack_binary_to_bytes<uint64_t>(data_bits.data(), codes, dim);

        for (size_t b_query = 0; b_query <= 8; ++b_query) {
            SCOPED_TRACE(b_query);
            std::vector<uint8_t> query_values(dim);
            size_t weighted_intersection = 0;
            for (size_t i = 0; i < dim; ++i) {
                query_values[i] =
                    static_cast<uint8_t>((i * 37 + 11) & ((1U << b_query) - 1));
                if (data_bits[i] != 0) {
                    weighted_intersection += query_values[i];
                }
            }

            std::vector<uint64_t> query(((dim + 63) / 64) * b_query, 0);
            size_t query_offset = 0;
            for (size_t block = 0; block < dim; block += 512) {
                const size_t block_dim = (dim - block < 512) ? dim - block : 512;
                const size_t chunks = (block_dim + 63) / 64;
                for (size_t bit = 0; bit < b_query; ++bit) {
                    for (size_t chunk = 0; chunk < chunks; ++chunk) {
                        uint64_t packed = 0;
                        const size_t width = std::min(size_t{64}, block_dim - chunk * 64);
                        for (size_t k = 0; k < width; ++k) {
                            const size_t i = block + chunk * 64 + k;
                            packed |= static_cast<uint64_t>((query_values[i] >> bit) & 1U)
                                      << (width - 1 - k);
                        }
                        query[query_offset + bit * chunks + chunk] = packed;
                    }
                }
                query_offset += chunks * b_query;
            }

            const float expected = kDelta * static_cast<float>(weighted_intersection) +
                                   kVl * static_cast<float>(data_popcount);
            EXPECT_FLOAT_EQ(
                warmup_ip_x0_q_512(codes, query.data(), kDelta, kVl, dim, b_query), expected
            );
#if defined(__x86_64__) || defined(_M_X64)
            if (cpu::has_avx2()) {
                EXPECT_FLOAT_EQ(
                    simd::warmup_ip_x0_q_512_avx2(
                        codes, query.data(), kDelta, kVl, dim, b_query
                    ),
                    expected
                );
            }
#endif
#if defined(__x86_64__) || defined(_M_X64)
            if (cpu::has_avx512_popcnt()) {
                EXPECT_FLOAT_EQ(
                    simd::warmup_ip_x0_q_512_avx512(
                        codes, query.data(), kDelta, kVl, dim, b_query
                    ),
                    expected
                );
            }
#endif
        }
    }
}

TEST(WarmupIpX0QTest, RejectsQueryWidthsBeyondByte) {
    const std::array<uint8_t, 8> data{};
    const std::array<uint64_t, 9> query{};
    EXPECT_THROW(
        warmup_ip_x0_q_512(data.data(), query.data(), 1.0F, 0.0F, 64, 9),
        std::invalid_argument
    );
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        EXPECT_THROW(
            simd::warmup_ip_x0_q_512_avx2(data.data(), query.data(), 1.0F, 0.0F, 64, 9),
            std::invalid_argument
        );
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx512_popcnt()) {
        EXPECT_THROW(
            simd::warmup_ip_x0_q_512_avx512(data.data(), query.data(), 1.0F, 0.0F, 64, 9),
            std::invalid_argument
        );
    }
#endif
}

TEST(SelectExcodeIpFuncTest, ReturnsStableFunctionPointer) {
    for (size_t bits = 0; bits <= 8; ++bits) {
        SCOPED_TRACE(bits);
        const auto ip_func = select_excode_ipfunc(bits);
        ASSERT_NE(ip_func, nullptr);
        ASSERT_EQ(ip_func, select_excode_ipfunc(bits));
#if defined(__x86_64__) || defined(_M_X64)
        if (bits == 8) {
            if (cpu::has_avx512_core()) {
                ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_avx512);
            } else if (cpu::has_avx2()) {
                ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_avx2);
            }
        }
#endif
    }
}

TEST(SelectExcodeIpFuncTest, ZeroExBitsContributesNothing) {
    constexpr size_t kDim = 64;
    std::vector<float> query(kDim);
    std::vector<uint8_t> codes(kDim / 8);

    for (size_t i = 0; i < kDim; ++i) {
        query[i] = static_cast<float>(i) + 1.0F;
    }
    for (size_t i = 0; i < codes.size(); ++i) {
        codes[i] = 0xFF;
    }

    ex_ipfunc ip_func = select_excode_ipfunc(0);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func(query.data(), codes.data(), kDim), 0.0F);
}

TEST(SelectExcodeIpFuncTest, RejectsExBitsAboveEight) {
    try {
        select_excode_ipfunc(9);
        FAIL() << "ex_bits above 8 must be rejected";
    } catch (const std::invalid_argument& error) {
        EXPECT_STREQ(error.what(), "ex_bits must be in [0, 8]");
    }
}

TEST(LegacyExcodeIpTest, Ip16Fxu1AvxMatchesReference) {
    constexpr size_t kDim = 64;
    float query[kDim];
    uint8_t codes[kDim / 8];

    for (size_t i = 0; i < kDim; ++i) {
        query[i] = static_cast<float>((i * 37 + 11) % 101) * 0.25F;
    }

    for (size_t i = 0; i < kDim / 8; ++i) {
        codes[i] = static_cast<uint8_t>(i * 53 + 19);
    }

    float expected = 0.0F;
    for (size_t i = 0; i < kDim; ++i) {
        expected += query[i] * static_cast<float>((codes[i / 8] >> (i % 8)) & 1U);
    }
    ASSERT_NEAR(
        rabitqlib::excode_ipimpl::ip16_fxu1_avx(query, codes, kDim), expected, 0.1F
    );
}

TEST(LegacyExcodeIpTest, Ip64Fxu2AvxMatchesReference) {
    constexpr size_t kDim = 64 * 4;
    float query[kDim];
    uint8_t codes[kDim / 4];

    for (size_t i = 0; i < kDim; ++i) {
        query[i] = static_cast<float>((i * 37 + 11) % 101) * 0.25F;
    }

    for (size_t i = 0; i < kDim / 4; ++i) {
        codes[i] = static_cast<uint8_t>(i * 53 + 19);
    }

    float expected = 0.0F;
    for (size_t i = 0; i < kDim; ++i) {
        const uint8_t packed = codes[(i / 64) * 16 + (i % 16)];
        const auto code = static_cast<uint8_t>((packed >> (2 * ((i % 64) / 16))) & 3U);
        expected += query[i] * static_cast<float>(code);
    }
    ASSERT_NEAR(
        rabitqlib::excode_ipimpl::ip64_fxu2_avx(query, codes, kDim), expected, 0.1F
    );
}

TEST(PackBinaryTest, CompactTailWordsMatchBinaryArithmetic) {
    for (size_t dim : {64U, 96U, 128U, 160U}) {
        std::vector<int> signs(dim);
        std::vector<uint16_t> values(dim);
        std::vector<uint8_t> code(dim / 8 + 1, 0xA5);
        std::vector<uint64_t> transposed((dim + 63) / 64 * 4);
        unsigned expected_count = 0, expected_ip = 0;
        for (size_t i = 0; i < dim; ++i) {
            signs[i] = i % 3 != 0;
            values[i] = (i * 7) % 16;
            expected_count += signs[i];
            expected_ip += signs[i] * values[i];
        }
        pack_binary_to_bytes<uint64_t>(signs.data(), code.data(), dim);
        EXPECT_EQ(code.back(), 0xA5);
        const auto* words = reinterpret_cast<const uint64_t*>(code.data());
        EXPECT_EQ(popcount(words, dim), expected_count);
        EXPECT_EQ(ip_bin_bin(words, words, dim), expected_count);
        new_transpose_bin(values.data(), transposed.data(), dim, 4);
        EXPECT_EQ(ip_x0_q(words, transposed.data(), 1, 0, dim, 4), expected_ip);
        EXPECT_EQ(
            rabitqlib::warmup_ip_x0_q<4>(words, transposed.data(), 1, 0, dim), expected_ip
        );
    }
}
