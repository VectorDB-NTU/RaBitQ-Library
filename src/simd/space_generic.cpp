#include <array>
#include <cstddef>

#include "rabitqlib/simd/space_dispatch.hpp"

namespace rabitqlib::simd {
namespace {
// Independent float sums limit reduction depth without widening the arithmetic.
template <bool SquaredDistance>
float product(const float* a, const float* b, size_t dim) {
    std::array<float, 16> sums{};
    auto value = [&](size_t at) {
        if constexpr (SquaredDistance) {
            const float diff = a[at] - b[at];
            return diff * diff;
        } else {
            return a[at] * b[at];
        }
    };
    size_t i = 0;
    for (; dim - i >= sums.size(); i += sums.size()) {
        for (size_t lane = 0; lane < sums.size(); ++lane)
            sums[lane] += value(i + lane);
    }
    for (size_t stride = sums.size() / 2; stride != 0; stride /= 2) {
        for (size_t lane = 0; lane < stride; ++lane)
            sums[lane] += sums[lane + stride];
    }
    float result = sums[0];
    for (; i < dim; ++i)
        result += value(i);
    return result;
}
}  // namespace

float euclidean_sqr_generic(const float* a, const float* b, size_t dim) {
    return product<true>(a, b, dim);
}

float dot_product_generic(const float* a, const float* b, size_t dim) {
    return product<false>(a, b, dim);
}

float dot_product_dis_generic(const float* a, const float* b, size_t dim) {
    return 1.0F - dot_product_generic(a, b, dim);
}

float l2norm_sqr_generic(const float* a, size_t dim) {
    return dot_product_generic(a, a, dim);
}

}  // namespace rabitqlib::simd
