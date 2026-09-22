#include "rabitqlib/utils/rotator.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "rabitqlib/utils/cpu_features.hpp"
#include "test_data.hpp"

using namespace rabitqlib;
using namespace rabitq_test;

class RotatorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        dim = 128;
        test_data = TestDataGenerator::GenerateRandomVector(dim, -1.0f, 1.0f, 42);
    }

    void TearDown() override {
        // Clean up any temporary files
        std::remove("test_rotator.bin");
    }

    size_t dim;
    std::vector<float> test_data;
};

// Test that FhtKacRotator is chosen by default
TEST_F(RotatorTest, DefaultRotatorType) {
    std::unique_ptr<Rotator<float>> rotator(choose_rotator<float>(dim));
    ASSERT_NE(rotator, nullptr);

    // FhtKacRotator pads to multiple of 64
    size_t padded_dim = rotator->size();
    EXPECT_EQ(padded_dim % 64, 0);
    EXPECT_GE(padded_dim, dim);
}

TEST_F(RotatorTest, RejectsPaddedDimensionSmallerThanInput) {
    EXPECT_THROW(
        (choose_rotator<float>(dim, RotatorType::FhtKacRotator, dim / 2)),
        std::invalid_argument
    );
    EXPECT_THROW(
        (choose_rotator<float>(dim, RotatorType::MatrixRotator, dim / 2)),
        std::invalid_argument
    );
}

TEST_F(RotatorTest, RejectsZeroDimension) {
    EXPECT_THROW(
        (choose_rotator<float>(0, RotatorType::FhtKacRotator)), std::invalid_argument
    );
    EXPECT_THROW(
        (choose_rotator<float>(0, RotatorType::MatrixRotator)), std::invalid_argument
    );
}

TEST(FhtRotatorTest, LargeDimensionsPreserveNormAndSavedRotation) {
    if (!cpu::has_avx2()) {
        GTEST_SKIP() << "FHT rotation requires AVX2/FMA";
    }
    for (size_t dim : {4095U, 4096U, 16384U, 16385U, 65535U, 65536U}) {
        SCOPED_TRACE(dim);
        std::unique_ptr<Rotator<float>> rotator(choose_rotator<float>(dim));
        std::vector<float> data(dim), rotated(rotator->size()), restored(rotator->size());
        for (size_t i = 0; i < dim; ++i) {
            data[i] = std::sin(static_cast<float>(i) * 0.13F);
        }
        rotator->rotate(data.data(), rotated.data());
        double input_norm = 0, output_norm = 0;
        for (float value : data) {
            input_norm += static_cast<double>(value) * value;
        }
        for (float value : rotated) {
            ASSERT_TRUE(std::isfinite(value));
            output_norm += static_cast<double>(value) * value;
        }
        EXPECT_NEAR(output_norm, input_norm, input_norm * 2e-6);
        std::vector<char> state(rotator->dump_bytes());
        rotator->save(state.data());
        std::unique_ptr<Rotator<float>> loaded(choose_rotator<float>(dim));
        loaded->load(state.data());
        loaded->rotate(data.data(), restored.data());
        EXPECT_EQ(restored, rotated);
    }
}

TEST(FhtRotatorTest, RejectsUnaddressableDimensionsBeforeAllocation) {
    EXPECT_THROW(
        (rotator_impl::FhtKacRotator(64, std::numeric_limits<size_t>::max() - 63)),
        std::invalid_argument
    );
    EXPECT_THROW(
        (choose_rotator<float>(rotator_impl::FhtKacRotator::kMaxDim + 1)),
        std::invalid_argument
    );
}

uint8_t bitreverse8(uint8_t x) {
    x = (((x & 0x55) << 1) | ((x & 0xAA) >> 1));
    x = (((x & 0x33) << 2) | ((x & 0xCC) >> 2));
    x = (((x & 0x0F) << 4) | ((x & 0xF0) >> 4));
    return x;
}
TEST(FlipSignTest, FlipWorks) {
    const size_t dim = 128;
    float data[dim];
    uint8_t flip[dim / 8];  // 1 bit per float

    // Initialize data and flip pattern
    for (size_t i = 0; i < dim; ++i) {
        data[i] = static_cast<float>(i + 1);  // Example data
    }
    for (size_t i = 0; i < dim / 8; ++i) {
        flip[i] = static_cast<uint8_t>(i % 256);  // Example flip pattern
    }

    // Perform sign flipping
    rabitqlib::rotator_impl::flip_sign(flip, data, dim);

    // Output the results
    uint8_t signs = 0;
    for (size_t i = 0; i < dim; ++i) {
        ASSERT_EQ(abs(data[i]), static_cast<float>(i + 1));
        int sign = (data[i] < 0) ? 1 : 0;
        signs = (signs << 1) | sign;
        if (i % 8 == 7) {
            uint8_t expected = flip[i / 8];
            signs = bitreverse8(signs);
            ASSERT_EQ(static_cast<uint8_t>(signs & 0xFF), expected);
            signs = 0;
        }
    }
}

TEST(FhtDispatchTest, BackendsMatchScalarButterfliesAndPadding) {
    std::vector<decltype(&simd::fht_rotate)> backends;
    if (cpu::has_avx2()) {
        backends.push_back(simd::fht_rotate_avx2);
    }
    if (cpu::has_avx512_core()) {
        backends.push_back(simd::fht_rotate_avx512);
    }
    if (backends.empty()) {
        GTEST_SKIP() << "FHT rotation requires AVX2/FMA or AVX512";
    }
    backends.push_back(simd::fht_rotate);
    for (size_t dim :
         {64U,   65U,   127U,  128U,  129U,  192U,   256U,   512U,   1024U,  2048U, 2049U,
          4095U, 4096U, 4097U, 8192U, 8193U, 16384U, 16385U, 32768U, 65535U, 65536U}) {
        SCOPED_TRACE(dim);
        const size_t padded = (dim + 63) / 64 * 64;
        size_t trunc = 1;
        while (trunc * 2 <= dim) {
            trunc *= 2;
        }
        const float fac = 1.0F / std::sqrt(static_cast<float>(trunc));
        std::vector<uint8_t> flip(4 * padded / 8);
        for (size_t i = 0; i < flip.size(); ++i) {
            flip[i] = static_cast<uint8_t>(i * 73 + 13);
        }
        std::vector<float> data(dim + 1), expected(padded, 0);
        for (size_t i = 0; i < dim; ++i) {
            data[i + 1] = std::sin(static_cast<float>(i) * 0.13F);
            expected[i] = data[i + 1];
        }
        for (size_t pass = 0; pass < 4; ++pass) {
            for (size_t i = 0; i < padded; ++i) {
                if ((flip[pass * padded / 8 + i / 8] >> (i % 8)) & 1) {
                    expected[i] = -expected[i];
                }
            }
            const size_t start = pass % 2 == 0 ? 0 : padded - trunc;
            for (size_t width = 1; width < trunc; width *= 2) {
                for (size_t block = 0; block < trunc; block += width * 2) {
                    for (size_t i = 0; i < width; ++i) {
                        const size_t x = start + block + i, y = x + width;
                        const float a = expected[x], b = expected[y];
                        expected[x] = a + b;
                        expected[y] = a - b;
                    }
                }
            }
            for (size_t i = start; i < start + trunc; ++i) {
                expected[i] *= fac;
            }
            if (padded != trunc) {
                for (size_t i = 0; i < padded / 2; ++i) {
                    const float a = expected[i], b = expected[i + padded / 2];
                    expected[i] = a + b;
                    expected[i + padded / 2] = a - b;
                }
            }
        }
        if (padded != trunc) {
            for (float& value : expected) {
                value *= 0.25F;
            }
        }
        for (auto backend : backends) {
            std::vector<float> actual(padded + 2, 12345);
            backend(
                data.data() + 1, actual.data() + 1, dim, padded, trunc, fac, flip.data()
            );
            for (size_t i = 0; i < padded; ++i) {
                EXPECT_NEAR(actual[i + 1], expected[i], 2e-5F);
            }
            EXPECT_EQ(actual.front(), 12345);
            EXPECT_EQ(actual.back(), 12345);
        }
    }
}

TEST(FhtDispatchTest, PreservesZeroNormAndInnerProduct) {
    std::vector<decltype(&simd::fht_rotate)> backends;
    if (cpu::has_avx2()) {
        backends.push_back(simd::fht_rotate_avx2);
    }
    if (cpu::has_avx512_core()) {
        backends.push_back(simd::fht_rotate_avx512);
    }
    if (backends.empty()) {
        GTEST_SKIP() << "FHT rotation requires AVX2/FMA or AVX512";
    }
    for (size_t dim : {64U, 65U, 128U, 192U, 256U, 512U, 4097U, 65536U}) {
        SCOPED_TRACE(dim);
        const size_t padded = (dim + 63) / 64 * 64;
        size_t trunc = 1;
        while (trunc * 2 <= dim) {
            trunc *= 2;
        }
        const float fac = 1.0F / std::sqrt(static_cast<float>(trunc));
        std::vector<uint8_t> flip(padded / 2, 0xA5);
        std::vector<float> a(dim, 0), b(dim), rotated_a(padded), rotated_b(padded);
        a[dim - 1] = 1;  // Exercise the padded tail with an impulse.
        for (size_t i = 0; i < dim; ++i) {
            b[i] = i % 2 == 0 ? 1.0F : -1.0F;
        }
        for (auto backend : backends) {
            backend(a.data(), rotated_a.data(), dim, padded, trunc, fac, flip.data());
            backend(b.data(), rotated_b.data(), dim, padded, trunc, fac, flip.data());
            double norm_a = 0, norm_b = 0, dot = 0;
            for (size_t i = 0; i < padded; ++i) {
                norm_a += static_cast<double>(rotated_a[i]) * rotated_a[i];
                norm_b += static_cast<double>(rotated_b[i]) * rotated_b[i];
                dot += static_cast<double>(rotated_a[i]) * rotated_b[i];
            }
            EXPECT_NEAR(norm_a, 1.0, 2e-6);
            EXPECT_NEAR(norm_b, static_cast<double>(dim), 2e-6 * dim);
            EXPECT_NEAR(dot, b.back(), 2e-6);
            std::vector<float> zero(dim, 0);
            backend(zero.data(), rotated_a.data(), dim, padded, trunc, fac, flip.data());
            for (float value : rotated_a) {
                EXPECT_EQ(value, 0.0F);
            }
        }
    }
}

TEST(MatrixRotatorTest, PreservesOverlappingInputAndOutput) {
    rotator_impl::MatrixRotator<float> rotator(3, 3);
    const float matrix[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    rotator.load(reinterpret_cast<const char*>(matrix));
    for (size_t offset : {0U, 1U}) {
        float data[] = {1, 2, 3, 0};
        rotator.rotate(data, data + offset);
        EXPECT_FLOAT_EQ(data[offset], 30);
        EXPECT_FLOAT_EQ(data[offset + 1], 36);
        EXPECT_FLOAT_EQ(data[offset + 2], 42);
    }
}
