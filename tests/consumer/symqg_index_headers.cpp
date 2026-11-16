#include <cstddef>
#include <random>
#include <vector>

#include "rabitqlib/index/symqg_index.hpp"

// The supported SymphonyQG API compiles and works without the existing graph classes or
// any implementation header.
int symqg_index_header_smoke() {
    constexpr size_t kDim = 64;
    constexpr size_t kNumPoints = 40;
    std::mt19937 rng(1);
    std::normal_distribution<float> normal;
    std::vector<float> data(kNumPoints * kDim);
    for (auto& value : data) {
        value = normal(rng);
    }
    rabitqlib::SymqgOptions options;
    options.init = rabitqlib::symqg::QGInitialization::Random;
    options.ef_construction = 40;
    options.seed = 3;
    rabitqlib::SymqgIndex index(kDim, options);
    const rabitqlib::Id first = index.add(data.data(), kNumPoints, 1);
    std::vector<rabitqlib::Id> ids(kNumPoints + 1);
    index.search(data.data(), 1, ids.size(), ids.data(), nullptr);
    return first == 0 && index.num_points() == kNumPoints && ids.back() == rabitqlib::kNoId
               ? 0
               : 1;
}
