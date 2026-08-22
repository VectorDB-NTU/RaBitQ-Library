#include "rabitqlib/utils/space.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "test_data.hpp"
#include "test_helpers.hpp"

using namespace rabitqlib;
using namespace rabitq_test;

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
    if (cpu::has_avx512_core()) {
        ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_avx512);
    } else {
        ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_avx2);
    }
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
    size_t dim = 64;
    float query[dim];
    uint8_t codes[dim / 8];

    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(rand()) / RAND_MAX * 1000.0f;
    }

    for (size_t i = 0; i < dim / 8; ++i) {
        codes[i] = static_cast<uint8_t>(rand() % 256);
    }

    ASSERT_NEAR(
        rabitqlib::excode_ipimpl::ip16_fxu1_avx(query, codes, dim), 15055.81f, 0.1f
    );
}

TEST(ip64_fxu2_avx, ip_works) {
    srand(42);
    size_t dim = 64 * 4;
    float query[dim];
    uint8_t codes[dim / 4];

    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(rand()) / RAND_MAX * 1000.0f;
    }

    for (size_t i = 0; i < dim / 4; ++i) {
        codes[i] = static_cast<uint8_t>(rand() % 256);
    }
    ASSERT_NEAR(
        rabitqlib::excode_ipimpl::ip64_fxu2_avx(query, codes, dim), 217584.15f, 0.1f
    );
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
    if (cpu::has_avx2()) {
        ASSERT_NEAR(
            simd::excode_ipimpl::ip16_fxu8_avx2(query.data(), codes.data(), dim),
            expected_float,
            0.1F
        );
    }
    if (cpu::has_avx512_core()) {
        ASSERT_NEAR(
            simd::excode_ipimpl::ip16_fxu8_avx512(query.data(), codes.data(), dim),
            expected_float,
            0.1F
        );
    }
}
