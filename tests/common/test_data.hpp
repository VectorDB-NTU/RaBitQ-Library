#ifndef RABITQ_TEST_DATA_HPP
#define RABITQ_TEST_DATA_HPP

#include <vector>
#include <random>
#include <cstddef>

namespace rabitq_test {

class TestDataGenerator {
public:
    // Generate random float vector with values in [min, max]
    static std::vector<float> GenerateRandomVector(
        size_t dim,
        float min = -1.0f,
        float max = 1.0f,
        unsigned int seed = 42
    );

    // Generate random normalized vector (unit length)
    static std::vector<float> GenerateNormalizedVector(
        size_t dim,
        unsigned int seed = 42
    );

    // Generate multiple random vectors
    static std::vector<std::vector<float>> GenerateRandomVectors(
        size_t num_vectors,
        size_t dim,
        float min = -1.0f,
        float max = 1.0f,
        unsigned int seed = 42
    );

    // Generate Gaussian distributed vector
    static std::vector<float> GenerateGaussianVector(
        size_t dim,
        float mean = 0.0f,
        float stddev = 1.0f,
        unsigned int seed = 42
    );

    // Generate a simple test vector with known values
    static std::vector<float> GenerateSimpleVector(size_t dim);

    // Generate zero vector
    static std::vector<float> GenerateZeroVector(size_t dim);

    // Generate vector with all ones
    static std::vector<float> GenerateOnesVector(size_t dim);

    // Generate vector with incremental values [0, 1, 2, 3, ...]
    static std::vector<float> GenerateIncrementalVector(size_t dim);
};

} // namespace rabitq_test

#endif // RABITQ_TEST_DATA_HPP
