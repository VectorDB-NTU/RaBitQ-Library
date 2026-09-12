#include "rabitqlib/simd/space_dispatch.hpp"

namespace rabitqlib::simd {

// Use wider scalar accumulation to limit rounding error on long inputs.
float euclidean_sqr_generic(const float* a, const float* b, size_t dim) {
    double sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        const double delta = static_cast<double>(a[i]) - b[i];
        sum += delta * delta;
    }
    return static_cast<float>(sum);
}

float dot_product_generic(const float* a, const float* b, size_t dim) {
    double sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(a[i]) * b[i];
    }
    return static_cast<float>(sum);
}

float dot_product_dis_generic(const float* a, const float* b, size_t dim) {
    return 1.0F - dot_product_generic(a, b, dim);
}

float l2norm_sqr_generic(const float* a, size_t dim) {
    double sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(a[i]) * a[i];
    }
    return static_cast<float>(sum);
}

}  // namespace rabitqlib::simd
