#include "test_data.hpp"
#include <cmath>
#include <algorithm>

namespace rabitq_test {

std::vector<float> TestDataGenerator::GenerateRandomVector(
    size_t dim,
    float min,
    float max,
    unsigned int seed
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min, max);

    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

std::vector<float> TestDataGenerator::GenerateNormalizedVector(
    size_t dim,
    unsigned int seed
) {
    auto vec = GenerateRandomVector(dim, -1.0f, 1.0f, seed);

    // Calculate L2 norm
    float norm = 0.0f;
    for (float val : vec) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    // Normalize
    if (norm > 1e-10f) {
        for (float& val : vec) {
            val /= norm;
        }
    }

    return vec;
}

std::vector<std::vector<float>> TestDataGenerator::GenerateRandomVectors(
    size_t num_vectors,
    size_t dim,
    float min,
    float max,
    unsigned int seed
) {
    std::vector<std::vector<float>> vectors;
    vectors.reserve(num_vectors);

    for (size_t i = 0; i < num_vectors; ++i) {
        vectors.push_back(GenerateRandomVector(dim, min, max, seed + i));
    }

    return vectors;
}

std::vector<float> TestDataGenerator::GenerateGaussianVector(
    size_t dim,
    float mean,
    float stddev,
    unsigned int seed
) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);

    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

std::vector<float> TestDataGenerator::GenerateSimpleVector(size_t dim) {
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = static_cast<float>(i % 10) / 10.0f;
    }
    return vec;
}

std::vector<float> TestDataGenerator::GenerateZeroVector(size_t dim) {
    return std::vector<float>(dim, 0.0f);
}

std::vector<float> TestDataGenerator::GenerateOnesVector(size_t dim) {
    return std::vector<float>(dim, 1.0f);
}

std::vector<float> TestDataGenerator::GenerateIncrementalVector(size_t dim) {
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = static_cast<float>(i);
    }
    return vec;
}

} // namespace rabitq_test
