#include "rabitqlib/utils/space.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rabitqlib/simd/pack_excode_dispatch.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/simd/warmup_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/warmup_space.hpp"

using namespace rabitqlib;

TEST(PackBinary, SupportsUnalignedOutput) {
    constexpr size_t dim = 128;
    std::array<int, dim> binary_code{};
    for (size_t i = 0; i < dim; ++i) {
        binary_code[i] = (i % 3) == 0;
    }

    alignas(uint64_t) std::array<uint8_t, (2 * sizeof(uint64_t)) + 1> storage{};
    auto* output = storage.data() + 1;
    ASSERT_NE(reinterpret_cast<uintptr_t>(output) % alignof(uint64_t), 0U);
    pack_binary_to_bytes<uint64_t>(binary_code.data(), output, dim);

    const std::array<uint64_t, 2> packed{
        load_unaligned_u64(storage.data() + 1),
        load_unaligned_u64(storage.data() + 1 + sizeof(uint64_t)),
    };
    for (size_t i = 0; i < dim; ++i) {
        const size_t word = i / 64;
        const auto bit = static_cast<int>((packed[word] >> (63 - (i % 64))) & 1U);
        EXPECT_EQ(bit, binary_code[i]) << "bit " << i;
    }
}

TEST(MaskIpX0Q, SupportsUnalignedCodes) {
    constexpr size_t dim = 128;
    std::array<int, dim> binary_code{};
    std::array<float, dim> query{};
    float expected = 0;
    for (size_t i = 0; i < dim; ++i) {
        binary_code[i] = (i % 3) == 0;
        query[i] = static_cast<float>(i + 1);
        expected += binary_code[i] != 0 ? query[i] : 0;
    }

    alignas(uint64_t) std::array<uint8_t, (2 * sizeof(uint64_t)) + 1> storage{};
    auto* codes = storage.data() + 1;
    ASSERT_NE(reinterpret_cast<uintptr_t>(codes) % alignof(uint64_t), 0U);
    pack_binary_to_bytes<uint64_t>(binary_code.data(), codes, dim);

#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        EXPECT_FLOAT_EQ(simd::mask_ip_x0_q_avx2(query.data(), codes, dim), expected);
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx512_core()) {
        EXPECT_FLOAT_EQ(simd::mask_ip_x0_q_avx512(query.data(), codes, dim), expected);
    }
#endif
    EXPECT_FLOAT_EQ(mask_ip_x0_q(query.data(), codes, dim), expected);
}

TEST(WarmupIpX0Q, SupportsUnalignedCodes) {
    constexpr float delta = 0.5F;
    constexpr float vl = -0.25F;
    for (size_t dim : {64UL, 128UL, 192UL, 448UL, 512UL, 576UL}) {
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

        for (size_t b_query : {1UL, 4UL, 8UL}) {
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

            std::vector<uint64_t> query((dim / 64) * b_query, 0);
            size_t query_offset = 0;
            for (size_t block = 0; block < dim; block += 512) {
                const size_t block_dim = (dim - block < 512) ? dim - block : 512;
                const size_t chunks = block_dim / 64;
                for (size_t bit = 0; bit < b_query; ++bit) {
                    for (size_t chunk = 0; chunk < chunks; ++chunk) {
                        uint64_t packed = 0;
                        for (size_t k = 0; k < 64; ++k) {
                            const size_t i = block + chunk * 64 + k;
                            packed |= static_cast<uint64_t>((query_values[i] >> bit) & 1U)
                                      << (63 - k);
                        }
                        query[query_offset + bit * chunks + chunk] = packed;
                    }
                }
                query_offset += chunks * b_query;
            }

            const float expected = delta * static_cast<float>(weighted_intersection) +
                                   vl * static_cast<float>(data_popcount);
#if defined(__x86_64__) || defined(_M_X64)
            if (cpu::has_avx2()) {
                EXPECT_FLOAT_EQ(
                    simd::warmup_ip_x0_q_512_avx2(
                        codes, query.data(), delta, vl, dim, b_query
                    ),
                    expected
                );
                EXPECT_FLOAT_EQ(
                    warmup_ip_x0_q_512(codes, query.data(), delta, vl, dim, b_query),
                    expected
                );
            }
#endif
#if defined(__x86_64__) || defined(_M_X64)
            if (cpu::has_avx512_popcnt()) {
                EXPECT_FLOAT_EQ(
                    simd::warmup_ip_x0_q_512_avx512(
                        codes, query.data(), delta, vl, dim, b_query
                    ),
                    expected
                );
            }
#endif
        }
    }
}

TEST(WarmupIpX0Q, RejectsQueryWidthsBeyondByte) {
    const std::array<uint8_t, 8> data{};
    const std::array<uint64_t, 9> query{};
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

TEST(Select_IP_Func, returns_stable_function_pointer) {
    auto ip_func = select_excode_ipfunc(0);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(0));

    ip_func = select_excode_ipfunc(1);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(1));

    ip_func = select_excode_ipfunc(2);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(2));

    ip_func = select_excode_ipfunc(3);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(3));

    ip_func = select_excode_ipfunc(4);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(4));

    ip_func = select_excode_ipfunc(5);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(5));

    ip_func = select_excode_ipfunc(6);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(6));

    ip_func = select_excode_ipfunc(7);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(7));

    ip_func = select_excode_ipfunc(8);
    ASSERT_NE(ip_func, nullptr);
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx512_core()) {
        ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_avx512);
    } else if (cpu::has_avx2()) {
        ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_avx2);
    }
#endif
}

TEST(Select_IP_Func, zero_ex_bits_contributes_nothing) {
    constexpr size_t dim = 64;
    std::vector<float> query(dim);
    std::vector<uint8_t> codes(dim / 8);

    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(i) + 1.0F;
    }
    for (size_t i = 0; i < codes.size(); ++i) {
        codes[i] = 0xFF;
    }

    ex_ipfunc ip_func = select_excode_ipfunc(0);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func(query.data(), codes.data(), dim), 0.0F);
}

TEST(ScalarQuantize, Uint8MatchesRoundedScalar) {
    constexpr size_t dim = 37;
    constexpr float lo = -3.0F;
    constexpr float delta = 0.25F;
    std::vector<float> input(dim);
    std::vector<uint8_t> result(dim);
    std::vector<uint8_t> expected(dim);

    for (size_t i = 0; i < dim; ++i) {
        float quantized = static_cast<float>((i * 7) % 251) +
                          static_cast<float>(static_cast<int>(i % 3) - 1) * 0.2F;
        input[i] = lo + delta * quantized;
        expected[i] = static_cast<uint8_t>(std::round((input[i] - lo) / delta));
    }

    scalar_quantize<uint8_t>(result.data(), input.data(), dim, lo, delta);

    ASSERT_EQ(result, expected);
}

TEST(ScalarQuantize, Uint16MatchesRoundedScalar) {
    constexpr size_t dim = 41;
    constexpr float lo = 2.0F;
    constexpr float delta = 0.125F;
    std::vector<float> input(dim);
    std::vector<uint16_t> result(dim);
    std::vector<uint16_t> expected(dim);

    for (size_t i = 0; i < dim; ++i) {
        float quantized = static_cast<float>(1000 + i * 317) +
                          static_cast<float>(static_cast<int>(i % 5) - 2) * 0.1F;
        input[i] = lo + delta * quantized;
        expected[i] = static_cast<uint16_t>(std::round((input[i] - lo) / delta));
    }

    scalar_quantize<uint16_t>(result.data(), input.data(), dim, lo, delta);

    ASSERT_EQ(result, expected);
}

constexpr size_t kScalarQuantizeTestDim = 33;

TEST(ScalarQuantize, HalfValuesMatchScalarAcrossVectorBoundaries) {
    constexpr size_t dim = kScalarQuantizeTestDim;
    const std::array<float, 9> values{
        0.5F,
        1.5F,
        2.5F,
        3.5F,
        4.5F,
        std::nextafter(2.5F, 0.0F),
        std::nextafter(2.5F, std::numeric_limits<float>::infinity()),
        125.5F,
        126.5F,
    };
    std::array<float, dim> input{};
    std::array<uint8_t, dim> expected8{};
    std::array<uint16_t, dim> expected16{};
    for (size_t i = 0; i < dim; ++i) {
        input[i] = (i == 8 || i == 16 || i == 32) ? 2.5F : values[i % values.size()];
        expected8[i] = static_cast<uint8_t>(std::round(input[i]));
        expected16[i] = static_cast<uint16_t>(std::round(input[i]));
    }

    const auto check = [&](auto quantize8, auto quantize16) {
        std::array<uint8_t, kScalarQuantizeTestDim> actual8{};
        std::array<uint16_t, kScalarQuantizeTestDim> actual16{};
        quantize8(actual8.data(), input.data(), dim, 0.0F, 1.0F);
        quantize16(actual16.data(), input.data(), dim, 0.0F, 1.0F);
        EXPECT_EQ(actual8, expected8);
        EXPECT_EQ(actual16, expected16);
    };

    check(simd::scalar_quantize_uint8, simd::scalar_quantize_uint16);
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        check(simd::scalar_quantize_uint8_avx2, simd::scalar_quantize_uint16_avx2);
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx512_core()) {
        check(simd::scalar_quantize_uint8_avx512, simd::scalar_quantize_uint16_avx512);
    }
#endif
}

TEST(ip16_fxu1_avx, ip_works) {
    constexpr size_t dim = 64;
    float query[dim];
    uint8_t codes[dim / 8];

    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>((i * 37 + 11) % 101) * 0.25F;
    }

    for (size_t i = 0; i < dim / 8; ++i) {
        codes[i] = static_cast<uint8_t>(i * 53 + 19);
    }

    float expected = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        expected += query[i] * static_cast<float>((codes[i / 8] >> (i % 8)) & 1U);
    }
    ASSERT_NEAR(rabitqlib::excode_ipimpl::ip16_fxu1_avx(query, codes, dim), expected, 0.1F);
}

TEST(ip64_fxu2_avx, ip_works) {
    constexpr size_t dim = 64 * 4;
    float query[dim];
    uint8_t codes[dim / 4];

    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>((i * 37 + 11) % 101) * 0.25F;
    }

    for (size_t i = 0; i < dim / 4; ++i) {
        codes[i] = static_cast<uint8_t>(i * 53 + 19);
    }

    float expected = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        const uint8_t packed = codes[(i / 64) * 16 + (i % 16)];
        const auto code = static_cast<uint8_t>((packed >> (2 * ((i % 64) / 16))) & 3U);
        expected += query[i] * static_cast<float>(code);
    }
    ASSERT_NEAR(rabitqlib::excode_ipimpl::ip64_fxu2_avx(query, codes, dim), expected, 0.1F);
}

TEST(OddBitExcodeIp, MatchesScalarInnerProduct) {
    constexpr size_t dim = 64 * 4;
    std::vector<float> query(dim);
    std::vector<uint8_t> codes(dim);

    for (size_t bits : std::array<size_t, 3>{3, 5, 7}) {
        const uint8_t max_code = static_cast<uint8_t>((1U << bits) - 1U);
        for (size_t i = 0; i < dim; ++i) {
            query[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 7.0F;
            codes[i] = static_cast<uint8_t>((i * 37U + 19U) & max_code);
        }
        // Exercise the high bit of each packed 64-value block, including bit 63 of the
        // scalar word used by the SIMD unpacking path.
        for (size_t i = 63; i < dim; i += 64) {
            codes[i] = max_code;
        }

        std::vector<uint8_t> compact(dim * bits / 8);
        if (bits == 3) {
            simd::packing_3bit_excode(codes.data(), compact.data(), dim);
        } else if (bits == 5) {
            simd::packing_5bit_excode(codes.data(), compact.data(), dim);
        } else {
            simd::packing_7bit_excode(codes.data(), compact.data(), dim);
        }

        double expected = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            expected += static_cast<double>(query[i]) * static_cast<double>(codes[i]);
        }
        const float expected_float = static_cast<float>(expected);

#if defined(__x86_64__) || defined(_M_X64)
        if (cpu::has_avx2()) {
            const std::array<ex_ipfunc, 8> avx2_functions{
                nullptr,
                simd::excode_ipimpl::ip16_fxu1_avx2,
                simd::excode_ipimpl::ip64_fxu2_avx2,
                simd::excode_ipimpl::ip64_fxu3_avx2,
                simd::excode_ipimpl::ip16_fxu4_avx2,
                simd::excode_ipimpl::ip64_fxu5_avx2,
                simd::excode_ipimpl::ip64_fxu6_avx2,
                simd::excode_ipimpl::ip64_fxu7_avx2,
            };
            ASSERT_NEAR(
                avx2_functions[bits](query.data(), compact.data(), dim),
                expected_float,
                0.1F
            );
        }
#endif
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu::has_avx512_core()) {
            const std::array<ex_ipfunc, 8> avx512_functions{
                nullptr,
                simd::excode_ipimpl::ip16_fxu1_avx512,
                simd::excode_ipimpl::ip64_fxu2_avx512,
                simd::excode_ipimpl::ip64_fxu3_avx512,
                simd::excode_ipimpl::ip16_fxu4_avx512,
                simd::excode_ipimpl::ip64_fxu5_avx512,
                simd::excode_ipimpl::ip64_fxu6_avx512,
                simd::excode_ipimpl::ip64_fxu7_avx512,
            };
            ASSERT_NEAR(
                avx512_functions[bits](query.data(), compact.data(), dim),
                expected_float,
                0.1F
            );
        }
#endif
    }
}

TEST(ip_fxu8_avx, ip_works) {
    constexpr size_t dim = 1024;
    std::vector<float> query(dim);
    std::vector<uint8_t> codes(dim);
    double expected = 0.0;

    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(i % 97) / 17.0F;
        codes[i] = static_cast<uint8_t>(i % 251);
        expected += static_cast<double>(query[i]) * static_cast<double>(codes[i]);
    }

    const float expected_float = static_cast<float>(expected);
    ex_ipfunc ip_func = select_excode_ipfunc(8);
    ASSERT_NEAR(ip_func(query.data(), codes.data(), dim), expected_float, 0.1F);
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        ASSERT_NEAR(
            simd::excode_ipimpl::ip16_fxu8_avx2(query.data(), codes.data(), dim),
            expected_float,
            0.1F
        );
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx512_core()) {
        ASSERT_NEAR(
            simd::excode_ipimpl::ip16_fxu8_avx512(query.data(), codes.data(), dim),
            expected_float,
            0.1F
        );
    }
#endif
}
