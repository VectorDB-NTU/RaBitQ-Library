#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/detail/parallel_for.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::index_detail {

// The cosine metric stores inner-product codes of unit vectors, so 1 - dot is 1 - cos.
inline MetricType stored_metric(MetricType metric_type) {
    return metric_type == METRIC_COSINE ? METRIC_IP : metric_type;
}

// Scale a vector to unit L2 norm. Callers check coordinates first, so the squared norm
// cannot overflow. A zero vector stays zero: its cosine distance to every vector is 1,
// as in hnswlib and Faiss.
inline void normalize_vector(const float* input, float* output, size_t dim) {
    double squared_norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        squared_norm += static_cast<double>(input[i]) * input[i];
    }
    const double scale = squared_norm > 0 ? 1.0 / std::sqrt(squared_norm) : 0.0;
    for (size_t i = 0; i < dim; ++i) {
        output[i] = static_cast<float>(input[i] * scale);
    }
}

inline std::vector<float> normalized_rows(
    const float* data, size_t rows, size_t dim, size_t num_threads
) {
    std::vector<float> normalized(rows * dim);
    constexpr size_t kRowsPerTask = 256;
    parallel_for(
        0,
        div_round_up(rows, kRowsPerTask),
        num_threads,
        [&](size_t task, size_t /*thread*/) {
            const size_t end = std::min(rows, (task + 1) * kRowsPerTask);
            for (size_t row = task * kRowsPerTask; row < end; ++row) {
                normalize_vector(data + (row * dim), normalized.data() + (row * dim), dim);
            }
        }
    );
    return normalized;
}

}  // namespace rabitqlib::index_detail
