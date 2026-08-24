#include "rabitqlib/quantization/rabitq.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rabitqlib::quant {
namespace {

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

}  // namespace
}  // namespace rabitqlib::quant
