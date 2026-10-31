#include "rabitqlib/fastscan/detail/lut_prepare.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "rabitqlib/core/detail/simd/space_dispatch.hpp"
#include "rabitqlib/fastscan/detail/lut.hpp"
#include "rabitqlib/fastscan/detail/simd.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

namespace rabitqlib::fastscan {
namespace {
using Pack = void (*)(size_t, const float*, float*);
using PackRange = void (*)(size_t, const float*, float*, float&, float&);
using Quantize = void (*)(uint8_t*, const float*, size_t, float, float);

struct BackendCase {
    const char* name;
    Pack pack;
    PackRange pack_range;
    Quantize quantize;
    Quantize reference_quantize;
};

std::vector<BackendCase> available_backends() {
    std::vector<BackendCase> backends{
        {"generic",
         simd::pack_lut_generic,
         simd::pack_lut_range_generic,
         rabitqlib::simd::scalar_quantize_uint8_generic,
         rabitqlib::simd::scalar_quantize_uint8_generic}};
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        backends.push_back(
            {"avx2",
             simd::pack_lut_avx2,
             simd::pack_lut_range_avx2,
             rabitqlib::simd::scalar_quantize_nonnegative_uint8_avx2,
             rabitqlib::simd::scalar_quantize_uint8_avx2}
        );
    }
    if (cpu::has_avx512_core()) {
        backends.push_back(
            {"avx512",
             simd::pack_lut_avx512,
             simd::pack_lut_range_avx512,
             rabitqlib::simd::scalar_quantize_nonnegative_uint8_avx512,
             rabitqlib::simd::scalar_quantize_uint8_avx512}
        );
    }
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    if (cpu::has_neon()) {
        backends.push_back(
            {"neon",
             simd::pack_lut_neon,
             simd::pack_lut_range_neon,
             rabitqlib::simd::scalar_quantize_uint8_neon,
             rabitqlib::simd::scalar_quantize_uint8_neon}
        );
    }
#endif
    return backends;
}

struct SavedEnvironment {
    std::fenv_t environment{};
    SavedEnvironment() { std::fegetenv(&environment); }
    ~SavedEnvironment() { std::fesetenv(&environment); }
};

template <typename T>
void expect_bits_equal(T a, T b) {
    EXPECT_EQ(std::memcmp(&a, &b, sizeof(T)), 0);
}

TEST(FastScanLutPreparationTest, FusedBackendsPreserveOrderedTablesAndFiniteExtrema) {
    const float tiny = std::numeric_limits<float>::denorm_min();
    const std::array<std::array<float, 4>, 6> patterns{
        {{1, 2, 4, 8},
         {-1, -2, -4, -8},
         {-0.0F, -0.0F, 0, 0},
         {1e20F, tiny, -1e20F, 1},
         {tiny, tiny, -tiny, -tiny},
         {-3.5F, .25F, 2, -.125F}}};
    for (const auto& backend : available_backends()) {
        SCOPED_TRACE(backend.name);
        for (size_t dim : {4U, 12U, 16U, 64U, 960U, 1536U}) {
            SCOPED_TRACE(dim);
            for (const auto& pattern : patterns) {
                std::vector<float> query(dim + 1);
                for (size_t i = 0; i < dim; ++i) {
                    query[i + 1] = pattern[i % 4];
                }
                const auto original = query;
                std::vector<float> expected(dim * 4 + 2, 123), actual = expected;
                backend.pack(dim, query.data() + 1, expected.data() + 1);
                float lo, hi, expected_lo, expected_hi;
                data_range(expected.data() + 1, dim * 4, expected_lo, expected_hi);
                backend.pack_range(dim, query.data() + 1, actual.data() + 1, lo, hi);
                EXPECT_EQ(
                    std::memcmp(
                        actual.data(), expected.data(), actual.size() * sizeof(float)
                    ),
                    0
                );
                EXPECT_EQ(
                    std::memcmp(
                        query.data(), original.data(), query.size() * sizeof(float)
                    ),
                    0
                );
                // Signed-zero selection is handled by the wrapper, not the raw kernel.
                EXPECT_EQ(lo, expected_lo);
                EXPECT_EQ(hi, expected_hi);
            }
        }
    }
}

TEST(FastScanLutPreparationTest, WrapperPreservesExceptionalAndSignedZeroBounds) {
    const SavedEnvironment saved;
    for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(rounding), 0);
        for (float value :
             {0.0F,
              -0.0F,
              std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::quiet_NaN()}) {
            std::array<float, 16> query{};
            query.fill(value);
            std::array<float, 66> expected{}, actual{};
            expected.front() = actual.front() = 19;
            expected.back() = actual.back() = 23;
            pack_lut(query.size(), query.data(), expected.data() + 1);
            float lo, hi, expected_lo, expected_hi;
            data_range(expected.data() + 1, size_t{64}, expected_lo, expected_hi);
            const bool finite = detail::pack_lut_with_range(
                query.size(), query.data(), actual.data() + 1, lo, hi
            );
            EXPECT_EQ(
                finite, std::all_of(expected.begin() + 1, expected.end() - 1, [](float x) {
                    return std::isfinite(x);
                })
            );
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), sizeof(actual)), 0);
            expect_bits_equal(lo, expected_lo);
            expect_bits_equal(hi, expected_hi);
        }
    }
}

TEST(FastScanLutPreparationTest, Uint8PathPreservesHalfBoundariesFlagsAndZeroRange) {
    const SavedEnvironment saved;
    std::vector<float> boundaries{0, -0.0F, 255};
    for (int value = 0; value < 255; ++value) {
        const float half = static_cast<float>(value) + .5F;
        boundaries.push_back(std::nextafter(half, -std::numeric_limits<float>::infinity()));
        boundaries.push_back(half);
        boundaries.push_back(std::nextafter(half, std::numeric_limits<float>::infinity()));
    }
    for (const auto& backend : available_backends()) {
        SCOPED_TRACE(backend.name);
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
            ASSERT_EQ(std::fesetround(rounding), 0);
            for (size_t count : {0U, 1U, 7U, 8U, 15U, 16U, 17U, 768U}) {
                std::vector<float> values(count + 1);
                for (size_t i = 0; i < count; ++i) {
                    values[i + 1] = boundaries[i % boundaries.size()];
                }
                std::vector<uint8_t> expected(count + 2, 71), actual = expected;
                std::feclearexcept(FE_ALL_EXCEPT);
                backend.reference_quantize(
                    expected.data() + 1, values.data() + 1, count, 0, 1
                );
                const int expected_flags = std::fetestexcept(FE_ALL_EXCEPT);
                std::feclearexcept(FE_ALL_EXCEPT);
                backend.quantize(actual.data() + 1, values.data() + 1, count, 0, 1);
                const int actual_flags = std::fetestexcept(FE_ALL_EXCEPT);
                EXPECT_EQ(actual, expected);
                EXPECT_EQ(actual_flags, expected_flags);
                // Round-half-away results are promised only in the documented
                // round-to-nearest environment: Clang inlines the scalar-tail
                // std::round in SIMD translation units with a mode-dependent addition.
                if (rounding == FE_TONEAREST) {
                    for (size_t i = 0; i < count; ++i) {
                        EXPECT_EQ(
                            actual[i + 1], static_cast<uint8_t>(std::round(values[i + 1]))
                        );
                    }
                }
                for (float delta : {0.0F, -0.0F}) {
                    backend.quantize(
                        actual.data() + 1, values.data() + 1, count, -0.0F, delta
                    );
                    EXPECT_EQ(actual.front(), 71);
                    EXPECT_EQ(actual.back(), 71);
                    EXPECT_TRUE(
                        std::all_of(actual.begin() + 1, actual.end() - 1, [](uint8_t x) {
                            return x == 0;
                        })
                    );
                }
            }
        }
    }
}

template <typename T>
void check_prepared_lut() {
    for (size_t dim : {16U, 128U, 960U, 1536U, 4096U}) {
        for (int pattern = 0; pattern < 4; ++pattern) {
            std::vector<T> query(dim), table(dim * 4);
            for (size_t i = 0; i < dim; ++i) {
                query[i] = pattern == 0   ? T{0}
                           : pattern == 1 ? -T{0}
                           : pattern == 2
                               ? static_cast<T>(static_cast<int>(i % 19) - 9) / T{7}
                               : std::numeric_limits<T>::denorm_min() *
                                     static_cast<T>(static_cast<int>(i % 3) - 1);
            }
            pack_lut(dim, query.data(), table.data());
            T lo, hi;
            data_range(table.data(), table.size(), lo, hi);
            for (bool hacc : {false, true}) {
                const T delta = (hi - lo) / static_cast<T>(hacc ? 65535 : 255);
                const T reciprocal = delta == 0 ? T{0} : T{1} / delta;
                std::vector<uint8_t> expected(table.size() * (hacc ? 2 : 1));
                std::vector<uint16_t> codes(table.size());
                for (size_t i = 0; i < table.size(); ++i) {
                    const T value =
                        delta == 0 ? T{0} : std::round((table[i] - lo) * reciprocal);
                    codes[i] = static_cast<uint16_t>(value);
                    if (!hacc) {
                        expected[i] = static_cast<uint8_t>(value);
                    }
                }
                if (hacc) {
                    transfer_lut_hacc(codes.data(), dim, expected.data());
                }
                Lut<T> actual(query.data(), dim, hacc);
                expect_bits_equal(actual.delta(), delta);
                expect_bits_equal(actual.sum_vl(), lo * static_cast<T>(dim / 4));
                EXPECT_EQ(std::memcmp(actual.lut(), expected.data(), expected.size()), 0);
            }
        }
    }
}

TEST(FastScanLutPreparationTest, FloatLutPreservesStandardAndHaccCodes) {
    check_prepared_lut<float>();
}
TEST(FastScanLutPreparationTest, DoubleLutRetainsOriginalPreparation) {
    check_prepared_lut<double>();
}
}  // namespace
}  // namespace rabitqlib::fastscan
