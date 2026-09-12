#include <gtest/gtest.h>

#include <algorithm>
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
    if (cpu::has_avx2()) {
        result.push_back(
            {"avx2",
             simd::euclidean_sqr_avx2,
             simd::dot_product_avx2,
             simd::dot_product_dis_avx2,
             simd::l2norm_sqr_avx2}
        );
    }
    if (cpu::has_avx512_core()) {
        result.push_back(
            {"avx512",
             simd::euclidean_sqr_avx512,
             simd::dot_product_avx512,
             simd::dot_product_dis_avx512,
             simd::l2norm_sqr_avx512}
        );
    }
    return result;
}

void check_reference(const float* a, const float* b, size_t dim) {
    long double l2 = 0, dot = 0, norm = 0, absolute_products = 0;
    for (size_t i = 0; i < dim; ++i) {
        const long double x = a[i], y = b[i];
        l2 += (x - y) * (x - y);
        dot += x * y;
        norm += x * x;
        absolute_products += std::abs(x * y);
    }
    // Longer float reductions accumulate more rounding error.
    const long double tolerance = 2e-6L * std::max(1.0L, dim / 4096.0L);
    for (const auto& backend : backends()) {
        SCOPED_TRACE(backend.name);
        EXPECT_NEAR(backend.l2(a, b, dim), l2, tolerance * std::max(1.0L, l2));
        EXPECT_NEAR(
            backend.dot(a, b, dim), dot, tolerance * std::max(1.0L, absolute_products)
        );
        EXPECT_NEAR(backend.ip(a, b, dim), 1 - dot, tolerance * (1 + absolute_products));
        EXPECT_NEAR(backend.norm(a, dim), norm, tolerance * std::max(1.0L, norm));
    }
}
}  // namespace

TEST(FloatDistance, BackendsMatchReferenceAcrossDimensionsAndAlignments) {
    std::vector<size_t> dimensions;
    for (size_t dim = 0; dim <= 80; ++dim) {
        dimensions.push_back(dim);
    }
    for (size_t dim :
         {127,
          128,
          129,
          255,
          256,
          257,
          419,
          420,
          421,
          959,
          960,
          961,
          1023,
          1024,
          1025,
          4096,
          65536}) {
        dimensions.push_back(dim);
    }
    for (size_t dim : dimensions) {
        SCOPED_TRACE(dim);
        for (size_t offset : {0, 1, 7, 15}) {
            std::vector<float> a(std::max(size_t{1}, dim + offset));
            std::vector<float> b(std::max(size_t{1}, dim + offset));
            for (int pattern = 0; pattern < 6; ++pattern) {
                SCOPED_TRACE(pattern);
                for (size_t i = 0; i < dim; ++i) {
                    float x = static_cast<float>(static_cast<int>(i % 23) - 11) / 7;
                    float y = static_cast<float>(static_cast<int>(i % 17) - 8) / 9;
                    if (pattern == 1) {
                        y = x;
                    } else if (pattern == 2) {
                        y = -x;
                    } else if (pattern == 3) {
                        x = (i % 4 < 2) ? 4096.0F : -4096.0F;
                        y = 1;
                    } else if (pattern == 4) {
                        x = std::ldexp(x, 30);
                        y = std::ldexp(y, 30);
                    } else if (pattern == 5) {
                        y = std::nextafter(x, 2.0F);
                    }
                    a[i + offset] = x;
                    b[i + offset] = y;
                }
                check_reference(a.data() + offset, b.data() + offset, dim);
            }
        }
    }
}

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
        for (size_t i = 0; i < dim; ++i) {
            data[i] = static_cast<float>(i % 7);
        }
        check_reference(data, data, dim);
    }
}
#endif
