#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/space.hpp"

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
using BinaryFn = float (*)(const float*, const float*, size_t);
using NormFn = float (*)(const float*, size_t);
struct Backend {
    const char* name;
    BinaryFn l2;
    BinaryFn dot;
    BinaryFn ip;
    NormFn norm;
};

std::vector<Backend> backends() {
    using namespace rabitqlib;
    std::vector<Backend> result{
        {"public",
         euclidean_sqr<float>,
         dot_product<float>,
         dot_product_dis<float>,
         l2norm_sqr<float>},
        {"generic",
         simd::euclidean_sqr_generic,
         simd::dot_product_generic,
         simd::dot_product_dis_generic,
         simd::l2norm_sqr_generic}};
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        result.push_back(
            {"avx2",
             simd::euclidean_sqr_avx2,
             simd::dot_product_avx2,
             simd::dot_product_dis_avx2,
             simd::l2norm_sqr_avx2}
        );
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx512_core()) {
        result.push_back(
            {"avx512",
             simd::euclidean_sqr_avx512,
             simd::dot_product_avx512,
             simd::dot_product_dis_avx512,
             simd::l2norm_sqr_avx512}
        );
    }
#endif
#if defined(__aarch64__)
    if (cpu::has_neon()) {
        result.push_back(
            {"neon",
             simd::euclidean_sqr_neon,
             simd::dot_product_neon,
             simd::dot_product_dis_neon,
             simd::l2norm_sqr_neon}
        );
    }
#endif
    return result;
}

}  // namespace

TEST(FloatDistance, EmptyAndAliasedInputsPreserveDistanceConventions) {
    const std::array<float, 5> a{1, -2, 3, -4, 5};
    for (const auto& backend : backends()) {
        SCOPED_TRACE(backend.name);
        EXPECT_EQ(backend.l2(nullptr, nullptr, 0), 0);
        EXPECT_EQ(backend.dot(nullptr, nullptr, 0), 0);
        EXPECT_EQ(backend.ip(nullptr, nullptr, 0), 1);
        EXPECT_EQ(backend.norm(nullptr, 0), 0);
        EXPECT_EQ(backend.l2(a.data(), a.data(), a.size()), 0);
        EXPECT_EQ(backend.dot(a.data(), a.data(), a.size()), 55);
        EXPECT_EQ(backend.ip(a.data(), a.data(), a.size()), -54);
        EXPECT_EQ(backend.norm(a.data(), a.size()), 55);
    }
}

TEST(FloatDistance, SubtractionUsesFloatPrecisionInVectorBlocksAndTails) {
    // In float, 1 - 2^-25 rounds to 1 before squaring. Widening the subtraction
    // to double would instead return the float immediately below 1.
    for (size_t dim : {1U, 4U, 16U, 17U, 33U}) {
        for (size_t at : {size_t{0}, dim - 1}) {
            std::vector<float> a(dim, 0), b(dim, 0);
            a[at] = 1.0F;
            b[at] = std::ldexp(1.0F, -25);
            for (const auto& backend : backends()) {
                SCOPED_TRACE(
                    ::testing::Message() << backend.name << " dim=" << dim << " at=" << at
                );
                EXPECT_EQ(backend.l2(a.data(), b.data(), dim), 1.0F);
            }
        }
    }
}

TEST(FloatDistance, AccumulationRetainsFloatPrecision) {
    // Each small square is below half an ulp at 1. Place them in the same
    // accumulation lane for all backends. Double accumulation would preserve
    // their combined 2^-23 contribution and return the float above 1.
    std::array<float, 513> a{}, zero{};
    a[0] = 1.0F;
    for (size_t i = 64; i < a.size(); i += 64)
        a[i] = std::ldexp(1.0F, -13);
    for (const auto& backend : backends()) {
        SCOPED_TRACE(backend.name);
        EXPECT_EQ(backend.l2(a.data(), zero.data(), a.size()), 1.0F);
        EXPECT_EQ(backend.dot(a.data(), a.data(), a.size()), 1.0F);
        EXPECT_EQ(backend.ip(a.data(), a.data(), a.size()), 0.0F);
        EXPECT_EQ(backend.norm(a.data(), a.size()), 1.0F);
    }
}

TEST(FloatDistance, DoubleTemplatesRetainDoublePrecision) {
    const std::array<double, 3> a{1.0000000001, 2.0000000001, 3.0000000001};
    const std::array<double, 3> b{1, 2, 3};
    EXPECT_NEAR(rabitqlib::euclidean_sqr(a.data(), b.data(), a.size()), 3e-20, 1e-26);
    EXPECT_DOUBLE_EQ(rabitqlib::dot_product(a.data(), b.data(), a.size()), 14.0000000006);
    EXPECT_DOUBLE_EQ(
        rabitqlib::dot_product_dis(a.data(), b.data(), a.size()), -13.0000000006
    );
    EXPECT_DOUBLE_EQ(rabitqlib::l2norm_sqr(a.data(), a.size()), 14.0000000012);
}

#if defined(__linux__)
TEST(FloatDistance, TailsDoNotReadPastGuardPage) {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    void* memory =
        mmap(nullptr, page * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(memory, MAP_FAILED);
    struct Mapping {
        void* address;
        size_t bytes;
        ~Mapping() { munmap(address, bytes); }
    } mapping{memory, page * 2};
    auto* end = static_cast<char*>(memory) + page;
    ASSERT_EQ(mprotect(end, page, PROT_NONE), 0);
    for (size_t dim :
         {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 420, 960}) {
        SCOPED_TRACE(dim);
        ASSERT_LE(dim * sizeof(float), page);
        auto* data = reinterpret_cast<float*>(end) - dim;
        float expected = 0;
        for (size_t i = 0; i < dim; ++i) {
            data[i] = static_cast<float>(i % 7);
            expected += data[i] * data[i];
        }
        // Small integer inputs make these sums exactly representable in float.
        // Keep this a bounds-safety check, independent of precision tolerances.
        for (const auto& backend : backends()) {
            SCOPED_TRACE(backend.name);
            EXPECT_EQ(backend.l2(data, data, dim), 0.0F);
            EXPECT_EQ(backend.dot(data, data, dim), expected);
            EXPECT_EQ(backend.ip(data, data, dim), 1.0F - expected);
            EXPECT_EQ(backend.norm(data, dim), expected);
        }
    }
}
#endif
