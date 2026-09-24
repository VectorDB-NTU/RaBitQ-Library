#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib {
namespace {
template <typename T>
void check_quantization(void (*quantize)(T*, const float*, size_t, float, float)) {
    const auto max = static_cast<float>(std::numeric_limits<T>::max());
    for (const auto& params : {std::array<float, 2>{0, 1}, {-3.25F, 0.125F}, {-7, 0.1F}}) {
        const float lo = params[0], delta = params[1];
        std::vector<float> input(1, 0);  // Offset SIMD loads by one float.
        input.push_back(lo);
        input.push_back(lo + max * delta);
        for (size_t code = 0; code < static_cast<size_t>(max); ++code) {
            const float half = lo + (static_cast<float>(code) + 0.5F) * delta;
            input.push_back(std::nextafter(half, -std::numeric_limits<float>::infinity()));
            input.push_back(half);
            input.push_back(std::nextafter(half, std::numeric_limits<float>::infinity()));
        }
        for (size_t dim : std::array<size_t, 14>{
                 0UL,
                 1UL,
                 3UL,
                 4UL,
                 7UL,
                 8UL,
                 9UL,
                 15UL,
                 16UL,
                 17UL,
                 31UL,
                 32UL,
                 33UL,
                 input.size() - 1}) {
            SCOPED_TRACE(dim);
            SCOPED_TRACE(delta);
            std::vector<T> actual(dim + 2, 99), expected(actual);
            const float reciprocal = 1.0F / delta;
            for (size_t i = 0; i < dim; ++i) {
                expected[i + 1] =
                    static_cast<T>(std::round((input[i + 1] - lo) * reciprocal));
            }
            quantize(actual.data() + 1, input.data() + 1, dim, lo, delta);
            EXPECT_EQ(actual, expected);
        }
    }
}

template <typename T>
void check_transpose(
    void (*transpose)(const T*, uint64_t*, size_t, size_t), bool blocked512
) {
    for (size_t dim : {0U, 64U, 128U, 448U, 512U, 576U, 960U, 1024U, 1088U}) {
        std::vector<T> input(dim + 1);
        for (size_t i = 0; i < dim; ++i) {
            input[i + 1] =
                static_cast<T>((i * 977 + i / 7) ^ (1U << (i % (sizeof(T) * 8))));
        }
        for (size_t bits = 0; bits <= sizeof(T) * 8; ++bits) {
            SCOPED_TRACE(dim);
            SCOPED_TRACE(bits);
            std::vector<uint64_t> actual(dim / 64 * bits + 2, UINT64_MAX);
            std::vector<uint64_t> expected(actual.size(), 0);
            expected.front() = expected.back() = UINT64_MAX;
            size_t offset = 1;
            const size_t block_size = blocked512 ? 512 : 64;
            for (size_t block = 0; block < dim; block += block_size) {
                const size_t chunks = std::min(block_size, dim - block) / 64;
                for (size_t b = 0; b < bits; ++b) {
                    for (size_t c = 0; c < chunks; ++c) {
                        for (size_t i = 0; i < 64; ++i) {
                            expected[offset + b * chunks + c] |=
                                uint64_t{(input[block + c * 64 + i + 1] >> b) & 1U}
                                << (63 - i);
                        }
                    }
                }
                offset += chunks * bits;
            }
            transpose(input.data() + 1, actual.data() + 1, dim, bits);
            EXPECT_EQ(actual, expected);
        }
    }
}
}  // namespace

TEST(QueryPreparation, QuantizationMatchesHalfAwayRoundingAndTails) {
    check_quantization(simd::scalar_quantize_uint8_generic);
    check_quantization(simd::scalar_quantize_uint16_generic);
    check_quantization(simd::scalar_quantize_uint8);
    check_quantization(simd::scalar_quantize_uint16);
#if defined(__aarch64__) || defined(_M_ARM64)
    if (cpu::has_neon()) {
        check_quantization(simd::scalar_quantize_uint8_neon);
        check_quantization(simd::scalar_quantize_uint16_neon);
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        check_quantization(simd::scalar_quantize_uint8_avx2);
        check_quantization(simd::scalar_quantize_uint16_avx2);
    }
    if (cpu::has_avx512_core()) {
        check_quantization(simd::scalar_quantize_uint8_avx512);
        check_quantization(simd::scalar_quantize_uint16_avx512);
    }
#endif
}

TEST(QueryPreparation, TranspositionMatchesCoordinateReferenceAndBounds) {
    check_transpose(simd::new_transpose_bin_generic, false);
    check_transpose(simd::new_transpose_bin_512_generic, true);
    check_transpose(new_transpose_bin, false);
    check_transpose(new_transpose_bin_512, true);
#if defined(__aarch64__) || defined(_M_ARM64)
    if (cpu::has_neon()) {
        check_transpose(simd::new_transpose_bin_neon, false);
        check_transpose(simd::new_transpose_bin_512_neon, true);
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        check_transpose(simd::new_transpose_bin_avx2, false);
        check_transpose(simd::new_transpose_bin_512_avx2, true);
    }
    if (cpu::has_avx512_core()) {
        check_transpose(simd::new_transpose_bin_avx512, false);
        check_transpose(simd::new_transpose_bin_512_avx512, true);
    }
#endif
}
}  // namespace rabitqlib
