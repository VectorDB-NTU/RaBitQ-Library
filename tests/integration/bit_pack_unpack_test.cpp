#include <gtest/gtest.h>
#include <rabitqlib/quantization/pack_excode.hpp>
#include <rabitqlib/utils/space.hpp>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace rabitqlib;

class BitPackUnpackTest : public ::testing::Test {
protected:
    // 1. Shared Constants & Data Structures
    const size_t dim = 768;
    std::vector<float> query;
    std::vector<uint8_t> code;
    std::vector<uint8_t> compact_code;

    // 2. Common Initialization (Runs before EVERY test)
    void SetUp() override {
        srand(42);
        
        // Initialize Query (Same for all tests)
        query.resize(dim);
        for(size_t i = 0; i < dim; ++i) {
            query[i] = static_cast<float>((rand() * 100.0) / RAND_MAX);
        }
        
        // Pre-allocate code vector
        code.resize(dim);
    }

    // 3. Helper to handle bit-specific setup
    void PrepareData(size_t bits) {
        // Resize compact code based on bits
        compact_code.resize(dim * bits / 8 + 1);

        // Generate random codes based on bit depth
        for (size_t i = 0; i < dim; ++i) {
            code[i] = rand() % (1 << bits); 
        }

        // Pack the code
        rabitqlib::quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
            code.data(), compact_code.data(), dim, bits
        );
    }

    // 4. Helper for Ground Truth Calculation
    float CalculateExpected() const {
        float expected_result = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            expected_result += query[i] * code[i];
        }
        return expected_result;
    }
};

// --- Test Cases ---

TEST_F(BitPackUnpackTest, ExCode1Bit) {
    PrepareData(1); // Set up for 1-bit

    // Run the AVX function
    float result = rabitqlib::excode_ipimpl::ip16_fxu1_avx(
        query.data(), compact_code.data(), dim
    );

    ASSERT_NEAR(CalculateExpected(), result, 0.1);
}

TEST_F(BitPackUnpackTest, ExCode2Bit) {
    PrepareData(2); // Set up for 2-bit

    // Run the AVX function
    float result = rabitqlib::excode_ipimpl::ip64_fxu2_avx(
        query.data(), compact_code.data(), dim
    );

    ASSERT_NEAR(CalculateExpected(), result, 0.1);
}

TEST_F(BitPackUnpackTest, ExCode3Bit) {
    PrepareData(3); // Set up for 3-bit

    // Run the AVX function
    float result = rabitqlib::excode_ipimpl::ip64_fxu3_avx(
        query.data(), compact_code.data(), dim
    );

    ASSERT_NEAR(CalculateExpected(), result, 0.1);
}

TEST_F(BitPackUnpackTest, ExCode4Bit) {
    PrepareData(4); // Set up for 4-bit

    // Run the AVX function
    float result = rabitqlib::excode_ipimpl::ip16_fxu4_avx(
        query.data(), compact_code.data(), dim
    );

    ASSERT_NEAR(CalculateExpected(), result, 0.1);
}

TEST_F(BitPackUnpackTest, ExCode5Bit) {
    PrepareData(5); // Set up for 5-bit

    // Run the AVX function
    float result = rabitqlib::excode_ipimpl::ip64_fxu5_avx(
        query.data(), compact_code.data(), dim
    );

    ASSERT_NEAR(CalculateExpected(), result, 0.1);
}

TEST_F(BitPackUnpackTest, ExCode6Bit) {
    PrepareData(6); // Set up for 6-bit

    // Run the AVX function
    float result = rabitqlib::excode_ipimpl::ip64_fxu6_avx(
        query.data(), compact_code.data(), dim
    );

    ASSERT_NEAR(CalculateExpected(), result, 0.1);
}

TEST_F(BitPackUnpackTest, ExCode7Bit) {
    PrepareData(7); // Set up for 7-bit

    // Run the AVX function
    float result = rabitqlib::excode_ipimpl::ip64_fxu7_avx(
        query.data(), compact_code.data(), dim
    );

    ASSERT_NEAR(CalculateExpected(), result, 0.1);
}


