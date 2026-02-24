#include <gtest/gtest.h>
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/defines.hpp"
#include "test_helpers.hpp"
#include "test_data.hpp"
#include <vector>
#include <cmath>

using namespace rabitqlib;
using namespace rabitq_test;

TEST(Select_IP_Func, returns_correct_function_pointer) {
    auto ip_func = select_excode_ipfunc(0);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip16_fxu1_avx);

    ip_func = select_excode_ipfunc(1);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip16_fxu1_avx);

    ip_func = select_excode_ipfunc(2);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip64_fxu2_avx);
    
    ip_func = select_excode_ipfunc(3);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip64_fxu3_avx);

    ip_func = select_excode_ipfunc(4);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip16_fxu4_avx);

    ip_func = select_excode_ipfunc(5);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip64_fxu5_avx);

    ip_func = select_excode_ipfunc(6);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip64_fxu6_avx);

    ip_func = select_excode_ipfunc(7);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip64_fxu7_avx);

    ip_func = select_excode_ipfunc(8);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, (excode_ipimpl::ip_fxi<float, uint8_t>));
}

TEST(ip16_fxu1_avx, ip_works) {
    srand(42);
    size_t dim = 64;
    float query[dim];
    uint8_t codes[dim/8];
    
    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(rand()) / RAND_MAX * 1000.0f;
    }

    for (size_t i = 0; i < dim / 8; ++i) {
        codes[i] = static_cast<uint8_t>(rand() % 256);
    }

    ASSERT_NEAR(rabitqlib::excode_ipimpl::ip16_fxu1_avx(query, codes, dim), 15055.81f, 0.1f);
}

TEST(ip64_fxu2_avx, ip_works) {
    srand(42);
    size_t dim = 64*4;
    float query[dim];
    uint8_t codes[dim/4];
    
    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(rand()) / RAND_MAX * 1000.0f;
    }

    for (size_t i = 0; i < dim / 4; ++i) {
        codes[i] = static_cast<uint8_t>(rand() % 256);
    }
    ASSERT_NEAR(rabitqlib::excode_ipimpl::ip64_fxu2_avx(query, codes, dim), 217584.15f, 0.1f);
}