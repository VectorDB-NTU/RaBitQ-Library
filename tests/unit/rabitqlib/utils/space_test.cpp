#include "rabitqlib/utils/space.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "rabitqlib/simd/pack_excode_dispatch.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

using namespace rabitqlib;

namespace {
bool has_avx2_backend() {
#if defined(__aarch64__)
    return true;  // The AVX2 implementation is compiled through SIMDe.
#else
    return cpu::has_avx2();
#endif
}
}  // namespace

TEST(ScalarQuantizeUint8, SupportsUnalignedOutputAcrossVectorBlocks) {
    constexpr size_t dim = 137;
    std::array<float, dim> query{};
    std::array<uint8_t, dim + 2> output{};
    output.front() = output.back() = 0xa5;
    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>((i * 13) % 256);
    }
    simd::scalar_quantize_uint8(output.data() + 1, query.data(), dim, 0.0F, 1.0F);
    for (size_t i = 0; i < dim; ++i) {
        EXPECT_EQ(output[i + 1], static_cast<uint8_t>(query[i]));
    }
    EXPECT_EQ(output.front(), 0xa5);
    EXPECT_EQ(output.back(), 0xa5);
}

TEST(PackBinary, SupportsUnalignedOutput) {
    constexpr size_t dim = 128;
    std::array<int, dim> binary_code{};
    for (size_t i = 0; i < dim; ++i) {
        binary_code[i] = (i % 3) == 0;
    }

    alignas(uint64_t) std::array<uint8_t, (2 * sizeof(uint64_t)) + 1> storage{};
    auto* output = reinterpret_cast<uint64_t*>(storage.data() + 1);
    ASSERT_NE(reinterpret_cast<uintptr_t>(output) % alignof(uint64_t), 0U);
    pack_binary(binary_code.data(), output, dim);

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
    auto* codes = reinterpret_cast<uint64_t*>(storage.data() + 1);
    ASSERT_NE(reinterpret_cast<uintptr_t>(codes) % alignof(uint64_t), 0U);
    pack_binary(binary_code.data(), codes, dim);

    if (has_avx2_backend()) {
        EXPECT_FLOAT_EQ(simd::mask_ip_x0_q_avx2(query.data(), codes, dim), expected);
    }
#if !defined(__aarch64__)
    if (cpu::has_avx512_core()) {
        EXPECT_FLOAT_EQ(simd::mask_ip_x0_q_avx512(query.data(), codes, dim), expected);
    }
#endif
    EXPECT_FLOAT_EQ(mask_ip_x0_q(query.data(), codes, dim), expected);
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
#if !defined(__aarch64__)
    if (cpu::has_avx512_core()) {
        ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_avx512);
    } else {
        ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_avx2);
    }
#else
    ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_avx2);
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

TEST(ip16_fxu1_avx, ip_works) {
    srand(42);
    constexpr size_t dim = 64;
    float query[dim];
    uint8_t codes[dim / 8];

    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 1000.0f;
    }

    for (size_t i = 0; i < dim / 8; ++i) {
        codes[i] = static_cast<uint8_t>(rand() % 256);
    }

    double expected = 0;
    for (size_t i = 0; i < dim; ++i) {
        expected += ((codes[i / 8] >> (i % 8)) & 1U) * static_cast<double>(query[i]);
    }
    ASSERT_NEAR(rabitqlib::excode_ipimpl::ip16_fxu1_avx(query, codes, dim), expected, 0.1f);
}

TEST(ip64_fxu2_avx, ip_works) {
    srand(42);
    constexpr size_t dim = 64 * 4;
    float query[dim];
    uint8_t codes[dim / 4];

    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 1000.0f;
    }

    for (size_t i = 0; i < dim / 4; ++i) {
        codes[i] = static_cast<uint8_t>(rand() % 256);
    }
    double expected = 0;
    for (size_t i = 0; i < dim; ++i) {
        const size_t block = i / 64, lane = i % 64;
        const uint8_t code = (codes[block * 16 + lane % 16] >> (2 * (lane / 16))) & 3U;
        expected += static_cast<double>(query[i]) * code;
    }
    ASSERT_NEAR(rabitqlib::excode_ipimpl::ip64_fxu2_avx(query, codes, dim), expected, 0.1f);
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

        if (has_avx2_backend()) {
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
#if !defined(__aarch64__)
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
    if (has_avx2_backend()) {
        ASSERT_NEAR(
            simd::excode_ipimpl::ip16_fxu8_avx2(query.data(), codes.data(), dim),
            expected_float,
            0.1F
        );
    }
#if !defined(__aarch64__)
    if (cpu::has_avx512_core()) {
        ASSERT_NEAR(
            simd::excode_ipimpl::ip16_fxu8_avx512(query.data(), codes.data(), dim),
            expected_float,
            0.1F
        );
    }
#endif
}
