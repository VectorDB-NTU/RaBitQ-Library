#include "rabitqlib/utils/input_checks.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace rabitqlib {
namespace {

TEST(InputChecksTest, DetectsEveryNonFiniteValueForFloatAndDouble) {
    const std::vector<float> finite{
        0.0F,
        -0.0F,
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        1.5F,
    };
    EXPECT_TRUE(all_finite(finite.data(), finite.size()));
    EXPECT_TRUE(all_finite(finite.data(), 0));
    const std::vector<double> finite_double(finite.begin(), finite.end());
    EXPECT_TRUE(all_finite(finite_double.data(), finite_double.size()));

    const float poison[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::signaling_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    for (float value : poison) {
        for (size_t position : {size_t{0}, finite.size() / 2, finite.size() - 1}) {
            auto values = finite;
            values[position] = value;
            EXPECT_FALSE(all_finite(values.data(), values.size()));
            const std::vector<double> doubles(values.begin(), values.end());
            EXPECT_FALSE(all_finite(doubles.data(), doubles.size()));
        }
    }
}

TEST(InputChecksTest, CoordinateBoundMatchesClusteringAndAcceptsItself) {
    constexpr size_t kDim = 128;
    const double expected = std::sqrt(
        std::sqrt(static_cast<double>(std::numeric_limits<float>::max())) /
        (64.0 * static_cast<double>(kDim))
    );
    const float bound = max_coordinate_magnitude(kDim);
    EXPECT_NEAR(static_cast<double>(bound), expected, expected * 1e-6);
    EXPECT_LT(max_coordinate_magnitude(2 * kDim), bound);

    std::vector<float> values(kDim, 0.25F);
    values[0] = bound;
    values[1] = -bound;
    EXPECT_NO_THROW(check_coordinates(values.data(), values.size(), kDim, "data"));
    EXPECT_NO_THROW(check_coordinates(values.data(), 0, kDim, "data"));

    const auto expect_message = [](const auto& call, const std::string& message) {
        try {
            call();
            ADD_FAILURE() << "expected: " << message;
        } catch (const std::invalid_argument& error) { EXPECT_EQ(error.what(), message); }
    };
    values[kDim - 1] = std::nextafter(bound, std::numeric_limits<float>::infinity());
    expect_message(
        [&] { check_coordinates(values.data(), values.size(), kDim, "data"); },
        "data exceeds the safe float32 coordinate range"
    );
    values[kDim - 1] = -std::nextafter(bound, std::numeric_limits<float>::infinity());
    expect_message(
        [&] { check_coordinates(values.data(), values.size(), kDim, "queries"); },
        "queries exceeds the safe float32 coordinate range"
    );
    // Non-finite values are reported as such even when the range is also exceeded.
    values[kDim - 1] = std::numeric_limits<float>::infinity();
    expect_message(
        [&] { check_coordinates(values.data(), values.size(), kDim, "data"); },
        "data must contain only finite values"
    );
    values[kDim - 1] = std::numeric_limits<float>::quiet_NaN();
    expect_message(
        [&] { check_coordinates(values.data(), values.size(), kDim, "data"); },
        "data must contain only finite values"
    );
}

}  // namespace
}  // namespace rabitqlib
