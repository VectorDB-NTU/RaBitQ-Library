#ifndef RABITQ_TEST_HELPERS_HPP
#define RABITQ_TEST_HELPERS_HPP

#include <gtest/gtest.h>

#include <cmath>

namespace rabitq_test {

// Floating point comparison with tolerance
inline bool FloatNearlyEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

#define EXPECT_FLOAT_NEARLY_EQUAL(a, b, epsilon)                    \
    EXPECT_TRUE(rabitq_test::FloatNearlyEqual((a), (b), (epsilon))) \
        << "Expected: " << (a) << " to be nearly equal to " << (b)  \
        << " (epsilon: " << (epsilon) << "), but difference was " << std::abs((a) - (b))

}  // namespace rabitq_test

#endif  // RABITQ_TEST_HELPERS_HPP
