#include <gtest/gtest.h>
#include "rabitqlib/utils/rotator.hpp"
#include "test_helpers.hpp"
#include "test_data.hpp"
#include <vector>
#include <cmath>
#include <fstream>
#include <cstring>

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
    Rotator<float>* rotator = choose_rotator<float>(dim);
    ASSERT_NE(rotator, nullptr);

    // FhtKacRotator pads to multiple of 64
    size_t padded_dim = rotator->size();
    EXPECT_EQ(padded_dim % 64, 0);
    EXPECT_GE(padded_dim, dim);

    delete rotator;
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
        if(i%8 == 7) {
            uint8_t expected = flip[i / 8];
            signs = bitreverse8(signs);
            ASSERT_EQ(static_cast<uint8_t>(signs & 0xFF), expected);
            signs = 0;
        }
    }
}
