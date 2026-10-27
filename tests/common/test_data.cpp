#include "test_data.hpp"

#include <cstddef>
#include <random>
#include <vector>

namespace rabitq_test {

std::vector<float> TestDataGenerator::GenerateRandomVector(
    size_t dim, float min, float max, unsigned int seed
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min, max);

    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

}  // namespace rabitq_test
