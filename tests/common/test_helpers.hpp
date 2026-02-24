#ifndef RABITQ_TEST_HELPERS_HPP
#define RABITQ_TEST_HELPERS_HPP

#include <cmath>
#include <vector>
#include <random>
#include <gtest/gtest.h>

namespace rabitq_test {

// Floating point comparison with tolerance
inline bool FloatNearlyEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

inline bool DoubleNearlyEqual(double a, double b, double epsilon = 1e-10) {
    return std::abs(a - b) < epsilon;
}

// Vector comparison
inline bool VectorsNearlyEqual(const float* a, const float* b, size_t size, float epsilon = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (!FloatNearlyEqual(a[i], b[i], epsilon)) {
            return false;
        }
    }
    return true;
}

// Calculate relative error
inline float RelativeError(float actual, float expected) {
    if (std::abs(expected) < 1e-10f) {
        return std::abs(actual - expected);
    }
    return std::abs((actual - expected) / expected);
}

// Calculate mean squared error
inline float MeanSquaredError(const float* a, const float* b, size_t size) {
    float mse = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = a[i] - b[i];
        mse += diff * diff;
    }
    return mse / size;
}

// Calculate dot product
inline float DotProduct(const float* a, const float* b, size_t size) {
    float result = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Calculate L2 distance
inline float L2Distance(const float* a, const float* b, size_t size) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Custom assertion macros
#define ASSERT_FLOAT_NEARLY_EQUAL(a, b, epsilon) \
    ASSERT_TRUE(rabitq_test::FloatNearlyEqual(a, b, epsilon)) \
        << "Expected: " << a << " to be nearly equal to " << b \
        << " (epsilon: " << epsilon << "), but difference was " << std::abs(a - b)

#define EXPECT_FLOAT_NEARLY_EQUAL(a, b, epsilon) \
    EXPECT_TRUE(rabitq_test::FloatNearlyEqual(a, b, epsilon)) \
        << "Expected: " << a << " to be nearly equal to " << b \
        << " (epsilon: " << epsilon << "), but difference was " << std::abs(a - b)

#define ASSERT_VECTORS_NEARLY_EQUAL(a, b, size, epsilon) \
    ASSERT_TRUE(rabitq_test::VectorsNearlyEqual(a, b, size, epsilon)) \
        << "Vectors are not nearly equal (epsilon: " << epsilon << ")"

} // namespace rabitq_test

#endif // RABITQ_TEST_HELPERS_HPP
