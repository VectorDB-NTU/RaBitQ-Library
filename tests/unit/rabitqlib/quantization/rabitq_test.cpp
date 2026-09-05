#include "rabitqlib/quantization/rabitq.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace rabitqlib::quant {
namespace {

TEST(RabitqQuantizedLevelTest, MatchesThresholdsAroundRoundingBoundaries) {
    const std::array<double, 7> magnitudes = {
        0.1,
        0.3,
        1.0 / 3.0,
        std::nextafter(1.0, 0.0),
        1.0,
        static_cast<double>(std::numeric_limits<float>::denorm_min()),
        std::numeric_limits<double>::max()};
    for (int bits = 0; bits <= 8; ++bits) {
        const int max_code = (1 << bits) - 1;
        EXPECT_EQ(rabitq_impl::ex_bits::quantized_level_at_scale(0, 1, max_code), 0);
        for (double magnitude : magnitudes) {
            for (int level = 1; level <= max_code + 1; ++level) {
                const double threshold = static_cast<double>(level) / magnitude;
                for (double t :
                     {std::nextafter(threshold, 0.0),
                      threshold,
                      std::nextafter(threshold, std::numeric_limits<double>::infinity())}) {
                    // Enumerate crossed thresholds independently of t * magnitude.
                    int expected = 0;
                    for (int code = 1; code <= max_code; ++code) {
                        if (static_cast<double>(code) / magnitude <= t) {
                            expected = code;
                        }
                    }
                    EXPECT_EQ(
                        rabitq_impl::ex_bits::quantized_level_at_scale(
                            magnitude, t, max_code
                        ),
                        expected
                    ) << "bits="
                      << bits << " magnitude=" << magnitude << " t=" << t;
                }
            }
        }
    }
}

TEST(RabitqRescaleSearchTest, SaturatedAndZeroCoordinatesHaveNoFurtherEvents) {
    const std::array<double, 2> magnitudes = {1.0, 0.0};
    std::array<int, 2> code{};
    const double start = 17.0 * rabitq_impl::ex_bits::kTightStart[3];

    EXPECT_DOUBLE_EQ(
        rabitq_impl::ex_bits::best_rescale_factor(magnitudes.data(), magnitudes.size(), 3),
        start
    );
    const double factor = rabitq_impl::ex_bits::quantize_ex(
        magnitudes.data(), code.data(), magnitudes.size(), 3
    );
    EXPECT_EQ(code, (std::array<int, 2>{7, 0}));
    EXPECT_DOUBLE_EQ(factor, 1.0 / 7.5);
}

TEST(RabitqRescaleSearchTest, KeepsInitialStateWhenLaterLegalStateIsWorse) {
    const std::array<double, 2> magnitudes = {0.8, 0.6};
    std::array<int, 2> code{};
    const double start = (17.0 / 0.8) * rabitq_impl::ex_bits::kTightStart[3];

    // The later event at 7 / 0.6 yields [7, 7], which has lower cosine similarity.
    EXPECT_DOUBLE_EQ(
        rabitq_impl::ex_bits::best_rescale_factor(magnitudes.data(), magnitudes.size(), 3),
        start
    );
    const double factor = rabitq_impl::ex_bits::quantize_ex(
        magnitudes.data(), code.data(), magnitudes.size(), 3
    );
    EXPECT_EQ(code, (std::array<int, 2>{7, 6}));
    EXPECT_DOUBLE_EQ(factor, 1.0 / (7.5 * 0.8 + 6.5 * 0.6));
}

TEST(RabitqRescaleSearchTest, EmitsAllCoordinatesCrossingTheSelectedThreshold) {
    for (size_t small_coordinate = 0; small_coordinate < 3; ++small_coordinate) {
        std::array<double, 3> magnitudes = {2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0};
        magnitudes[small_coordinate] = 1.0 / 3.0;
        std::array<int, 3> expected = {3, 3, 3};
        expected[small_coordinate] = 1;
        std::array<int, 3> code{};

        // Both larger coordinates cross into level 3 at exactly t = 4.5.
        EXPECT_DOUBLE_EQ(
            rabitq_impl::ex_bits::best_rescale_factor(
                magnitudes.data(), magnitudes.size(), 2
            ),
            4.5
        );
        const double factor = rabitq_impl::ex_bits::quantize_ex(
            magnitudes.data(), code.data(), magnitudes.size(), 2
        );
        EXPECT_EQ(code, expected);
        EXPECT_DOUBLE_EQ(factor, 6.0 / 31.0);
    }
}

TEST(RabitqDegenerateInputTest, OneBitFactorsAreFiniteForZeroResidual) {
    constexpr size_t kDim = 64;
    std::array<float, kDim> data{};
    std::array<float, kDim> centroid{};
    std::array<int, kDim> code{};

    for (MetricType metric : {METRIC_L2, METRIC_IP}) {
        float f_add = 0;
        float f_rescale = 0;
        float f_error = 0;
        rabitq_impl::one_bit::one_bit_code_with_factor(
            data.data(),
            centroid.data(),
            kDim,
            code.data(),
            f_add,
            f_rescale,
            f_error,
            metric
        );

        EXPECT_TRUE(std::isfinite(f_add));
        EXPECT_TRUE(std::isfinite(f_rescale));
        EXPECT_FLOAT_EQ(f_error, 0.0F);
    }
}

TEST(RabitqDegenerateInputTest, ExtraBitFactorsAreFiniteForZeroResidual) {
    constexpr size_t kDim = 64;
    std::array<float, kDim> data{};
    std::array<float, kDim> centroid{};
    std::array<uint8_t, kDim> code{};
    float f_add = 0;
    float f_rescale = 0;
    float f_error = 0;

    rabitq_impl::ex_bits::ex_bits_code_with_factor(
        data.data(), centroid.data(), kDim, 3, code.data(), f_add, f_rescale, f_error
    );

    EXPECT_TRUE(std::isfinite(f_add));
    EXPECT_TRUE(std::isfinite(f_rescale));
    EXPECT_FLOAT_EQ(f_error, 0.0F);
}

TEST(RabitqDegenerateInputTest, ScalarQuantizationReconstructsZeroVector) {
    constexpr size_t kDim = 64;
    std::array<float, kDim> data{};
    std::array<uint8_t, kDim> code{};
    std::array<float, kDim> reconstructed{};
    float delta = 1;
    float vl = 1;

    quantize_scalar(data.data(), kDim, 4, code.data(), delta, vl);
    reconstruct_vec(code.data(), delta, vl, kDim, reconstructed.data());

    EXPECT_TRUE(std::isfinite(delta));
    EXPECT_TRUE(std::isfinite(vl));
    EXPECT_EQ(reconstructed, data);
}

TEST(RabitqOneBitTest, FullQuantizationInitializesCodesAndFactors) {
    constexpr size_t kDim = 64;
    std::array<float, kDim> data{};
    std::array<uint8_t, kDim> code;
    code.fill(0xFF);
    data[0] = 1.0F;
    float f_add = std::numeric_limits<float>::quiet_NaN();
    float f_rescale = std::numeric_limits<float>::quiet_NaN();
    float f_error = std::numeric_limits<float>::quiet_NaN();

    quantize_full_single(data.data(), kDim, 1, code.data(), f_add, f_rescale, f_error);

    EXPECT_EQ(code[0], 1);
    for (size_t i = 1; i < kDim; ++i) {
        EXPECT_EQ(code[i], 0);
    }
    EXPECT_TRUE(std::isfinite(f_add));
    EXPECT_TRUE(std::isfinite(f_rescale));
    EXPECT_TRUE(std::isfinite(f_error));
}

TEST(RabitqFullQuantizationTest, ReconstructsFromCodeAndEstimatorFactors) {
    constexpr size_t kDim = 64;
    constexpr size_t kBits = 4;
    std::array<float, kDim> data{};
    std::array<float, kDim> centroid{};
    std::array<uint8_t, kDim> code{};
    std::array<float, kDim> reconstructed{};
    for (size_t i = 0; i < kDim; ++i) {
        data[i] = static_cast<float>(i % 11) - 5.0F;
        centroid[i] = static_cast<float>(i % 3) * 0.25F;
    }

    float f_add = 0;
    float f_rescale = 0;
    float f_error = 0;
    quantize_full_single(
        data.data(), centroid.data(), kDim, kBits, code.data(), f_add, f_rescale, f_error
    );
    reconstruct_full_vec(
        code.data(), centroid.data(), kDim, kBits, f_rescale, reconstructed.data()
    );

    const float code_center = -static_cast<float>((1 << kBits) - 1) / 2;
    const float scale = -f_rescale / 2;
    for (size_t i = 0; i < kDim; ++i) {
        EXPECT_FLOAT_EQ(
            reconstructed[i],
            centroid[i] + scale * (static_cast<float>(code[i]) + code_center)
        );
    }
}

// An exactly-zero residual coordinate must be encoded consistently by both halves of
// the split code. one_bit_code() uses (residual > 0) and so calls zero negative; if
// ex_bits_code() calls it positive, the assembled code lands 2^ex_bits away from where
// the factors assume it is. The <c*xu_cb, r> == ||r||^2 invariant cannot see this,
// because a zero coordinate contributes nothing to that inner product.
TEST(RabitqSignConventionTest, ZeroResidualCoordinateGetsSmallestMagnitudeCode) {
    constexpr size_t kDim = 64;
    constexpr size_t kTotalBits = 8;
    constexpr size_t kExBits = kTotalBits - 1;

    std::array<float, kDim> data{};
    for (size_t i = 0; i < kDim; ++i) {
        data[i] = static_cast<float>(i % 7) - 3.0F;  // zero at i % 7 == 3
    }

    std::array<uint8_t, kDim> code{};
    float f_add = 0;
    float f_rescale = 0;
    float f_error = 0;
    quantize_full_single(
        data.data(), kDim, kTotalBits, code.data(), f_add, f_rescale, f_error
    );

    // centered code is code + cb, cb = -(2^total_bits - 1)/2
    const float cb = -((1 << kExBits) - 0.5F);
    size_t n_zero = 0;
    for (size_t i = 0; i < kDim; ++i) {
        if (data[i] != 0.0F) {
            continue;
        }
        ++n_zero;
        EXPECT_FLOAT_EQ(static_cast<float>(code[i]) + cb, -0.5F)
            << "zero residual at dim " << i << " encoded as " << +code[i];
    }
    ASSERT_GT(n_zero, 0U) << "test vector must contain exact zeros";
}

TEST(RabitqSignConventionTest, ZeroResidualDoesNotInflateReconstructionNorm) {
    constexpr size_t kDim = 64;
    constexpr size_t kTotalBits = 8;
    constexpr size_t kExBits = kTotalBits - 1;

    std::array<float, kDim> data{};
    for (size_t i = 0; i < kDim; ++i) {
        data[i] = static_cast<float>((i * 37) % 23) - 11.0F;  // zeros at rem 11
    }

    std::array<uint8_t, kDim> code{};
    float f_add = 0;
    float f_rescale = 0;
    float f_error = 0;
    quantize_full_single(
        data.data(), kDim, kTotalBits, code.data(), f_add, f_rescale, f_error
    );

    const double cb = -((1 << kExBits) - 0.5);
    const double c = f_rescale / -2.0;
    double nsq = 0;
    double qq = 0;
    size_t n_zero = 0;
    for (size_t i = 0; i < kDim; ++i) {
        const double q = static_cast<double>(code[i]) + cb;
        nsq += static_cast<double>(data[i]) * data[i];
        qq += q * q;
        n_zero += (data[i] == 0.0F);
    }
    ASSERT_GT(n_zero, 0U) << "test vector must contain exact zeros";

    // ||c*xu_cb||^2 / ||r||^2 - 1 == tan^2 of the angle between the residual and its
    // reconstruction. A mismatched sign convention inflates this by orders of magnitude.
    const double tan_sq = ((c * c * qq) / nsq) - 1.0;
    EXPECT_LT(tan_sq, 1e-3) << "reconstruction norm inflated, tan^2 = "
                            << std::to_string(tan_sq);
}

}  // namespace
}  // namespace rabitqlib::quant
