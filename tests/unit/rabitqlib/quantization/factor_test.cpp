#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rabitqlib/core/detail/rabitq_impl.hpp"
#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/metric.hpp"

namespace rabitqlib::quant {
namespace {

template <typename T>
void check_collinear_factors() {
    constexpr size_t kDim = 64;
    constexpr T kCenter = T{0.25};
    std::vector<T> data(kDim), centroid(kDim, kCenter), residual(kDim);
    std::vector<int> codes(kDim);
    for (const auto metric : {METRIC_L2, METRIC_IP}) {
        for (const T value : {T{0}, T{0.5}, T{-0.5}}) {
            std::fill(data.begin(), data.end(), kCenter + value);
            std::fill(residual.begin(), residual.end(), value);
            for (const size_t bits : {1U, 2U, 4U, 9U}) {
                for (const double scale : {-1.0, 7.0}) {
                    RabitqConfig config;
                    config.t_const = scale;
                    T add, rescale, error;
                    quantize_full_single(
                        data.data(),
                        centroid.data(),
                        kDim,
                        bits,
                        codes.data(),
                        add,
                        rescale,
                        error,
                        metric,
                        config
                    );
                    ASSERT_TRUE(std::all_of(codes.begin(), codes.end(), [&](int code) {
                        return code == codes.front();
                    }));
                    const T centered_code = static_cast<T>(codes.front()) -
                                            static_cast<T>((1U << bits) - 1) / 2;
                    // Constant residual/code vectors are collinear: the error is zero.
                    // L2 add = d*r^2 + 2*d*r*c; IP's centroid terms cancel to 1.
                    const T expected_add =
                        metric == METRIC_L2
                            ? kDim * value * value + 2 * kDim * value * kCenter
                            : T{1};
                    const T expected_rescale = value == 0 ? T{0}
                                               : metric == METRIC_L2
                                                   ? -2 * value / centered_code
                                                   : -value / centered_code;
                    const auto check = [&] {
                        EXPECT_EQ(add, expected_add);
                        EXPECT_NEAR(
                            rescale,
                            expected_rescale,
                            8 * std::numeric_limits<T>::epsilon() *
                                std::max(T{1}, std::abs(expected_rescale))
                        );
                        EXPECT_EQ(error, T{0});
                    };
                    check();
                    rabitq_impl::xy_bits::code_factors(
                        residual.data(),
                        centroid.data(),
                        kDim,
                        codes.data(),
                        -static_cast<float>((1U << bits) - 1) / 2,
                        add,
                        rescale,
                        error,
                        metric
                    );
                    check();
                }
            }
        }
    }
}

TEST(RabitqFactorTest, CollinearMetricFactorsFloat) { check_collinear_factors<float>(); }

TEST(RabitqFactorTest, CollinearMetricFactorsDouble) { check_collinear_factors<double>(); }

template <typename T>
void check_rejected_foreign_code() {
    constexpr size_t kDim = 64;
    std::vector<T> residual(kDim), centroid(kDim, T{0});
    std::vector<int> codes(kDim);
    T add, rescale, error;
    for (const auto metric : {METRIC_L2, METRIC_IP}) {
        // A code of the opposite signs, then one orthogonal to the residual: the
        // factor derivation divides by <residual, centered code>, which must be > 0.
        for (const bool orthogonal : {false, true}) {
            for (size_t i = 0; i < kDim; ++i) {
                residual[i] = orthogonal && i % 2 == 1 ? T{-1} : T{1};
                codes[i] = orthogonal ? 1 : 0;
            }
            try {
                rabitq_impl::xy_bits::code_factors(
                    residual.data(),
                    centroid.data(),
                    kDim,
                    codes.data(),
                    -0.5F,
                    add,
                    rescale,
                    error,
                    metric
                );
                FAIL() << "Expected rejection of a code that does not quantize residual";
            } catch (const std::invalid_argument& exception) {
                EXPECT_STREQ(
                    exception.what(),
                    "code does not quantize residual: <residual, centered code> must be "
                    "positive"
                );
            }
        }
    }
}

TEST(RabitqFactorTest, NanInputIsNotCheckedByTheEncoders) {
    // The per-vector encoders do not check that inputs are finite (quantizer.md); the
    // rejection of a nonpositive divisor must not turn a NaN into an exception.
    constexpr size_t kDim = 64;
    std::vector<float> data(kDim, 0.5F), centroid(kDim, 0), residual(kDim), centered(kDim);
    data[7] = std::numeric_limits<float>::quiet_NaN();
    std::vector<int> codes(kDim);
    float add, rescale, error;
    rabitq_impl::one_bit::one_bit_code_with_factor_into(
        data.data(),
        centroid.data(),
        kDim,
        codes.data(),
        add,
        rescale,
        error,
        METRIC_L2,
        residual.data(),
        centered.data()
    );
    EXPECT_TRUE(std::isnan(add));
    EXPECT_TRUE(std::isnan(rescale));
}

TEST(RabitqFactorTest, ForeignCodeIsRejectedFloat) { check_rejected_foreign_code<float>(); }

TEST(RabitqFactorTest, ForeignCodeIsRejectedDouble) {
    check_rejected_foreign_code<double>();
}

TEST(RabitqFactorTest, InvalidMetricMessagesArePreserved) {
    constexpr size_t kDim = 64;
    const auto invalid_metric = METRIC_COSINE;  // only the ANN indexes accept it
    std::vector<float> data(kDim), centroid(kDim, 0), residual(kDim), centered(kDim);
    std::vector<int> codes(kDim);
    float add, rescale, error;
    const auto check = [](auto&& operation, const char* message) {
        try {
            operation();
            FAIL() << "Expected invalid metric rejection";
        } catch (const std::invalid_argument& exception) {
            EXPECT_STREQ(exception.what(), message);
        }
    };
    for (const float value : {0.0F, 0.5F}) {
        std::fill(data.begin(), data.end(), value);
        std::fill(residual.begin(), residual.end(), value);
        check(
            [&] {
                rabitq_impl::one_bit::one_bit_code_with_factor_into(
                    data.data(),
                    centroid.data(),
                    kDim,
                    codes.data(),
                    add,
                    rescale,
                    error,
                    invalid_metric,
                    residual.data(),
                    centered.data()
                );
            },
            "Unsupported metric type in quantization"
        );
        check(
            [&] {
                rabitq_impl::ex_bits::ex_bits_code_with_factor(
                    data.data(),
                    centroid.data(),
                    kDim,
                    3,
                    codes.data(),
                    add,
                    rescale,
                    error,
                    invalid_metric
                );
            },
            "Unsupported metric for ex_bits_code()"
        );
        std::fill(codes.begin(), codes.end(), 1);
        check(
            [&] {
                rabitq_impl::xy_bits::code_factors(
                    residual.data(),
                    centroid.data(),
                    kDim,
                    codes.data(),
                    -0.5F,
                    add,
                    rescale,
                    error,
                    invalid_metric
                );
            },
            "Unsupported metric type in code_factors()"
        );
    }
}

}  // namespace
}  // namespace rabitqlib::quant
