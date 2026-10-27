#ifndef RABITQ_TEST_DATA_HPP
#define RABITQ_TEST_DATA_HPP

#include <cstddef>
#include <vector>

namespace rabitq_test {

class TestDataGenerator {
   public:
    // Generate random float vector with values in [min, max]
    static std::vector<float> GenerateRandomVector(
        size_t dim, float min = -1.0f, float max = 1.0f, unsigned int seed = 42
    );
};

}  // namespace rabitq_test

#endif  // RABITQ_TEST_DATA_HPP
