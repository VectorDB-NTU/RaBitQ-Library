#include <cstddef>
#include <random>
#include <vector>

#include "rabitqlib/index/ivf_index.hpp"

// The supported IVF API compiles and works without the existing IVF class, k-means or
// any implementation header.
int main() {
    constexpr size_t kDim = 64;
    constexpr size_t kNumPoints = 40;
    std::mt19937 rng(1);
    std::normal_distribution<float> normal;
    std::vector<float> data(kNumPoints * kDim);
    for (auto& value : data) {
        value = normal(rng);
    }
    rabitqlib::IvfOptions options;
    options.seed = 3;
    rabitqlib::IvfIndex index(kDim, 2, options);
    const rabitqlib::Id first = index.add(data.data(), kNumPoints);
    std::vector<rabitqlib::Id> ids(kNumPoints + 1);
    index.search(data.data(), 1, ids.size(), ids.data(), nullptr);
    return first == 0 && index.num_points() == kNumPoints && ids.back() == rabitqlib::kNoId
               ? 0
               : 1;
}
