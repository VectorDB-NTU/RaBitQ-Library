#pragma once

#include <cstddef>

namespace rabitqlib::simd {

// Search normalized, nonnegative float magnitudes with max_code in [0, 255].
// A negative result requests the event sweep when the bounded search cannot
// certify a winner within its work limit.
double best_rescale_factor(
    const float* magnitudes, size_t dim, int max_code, double start, double end
);
double best_rescale_factor_avx2(
    const float* magnitudes, size_t dim, int max_code, double start, double end
);
double best_rescale_factor_avx512(
    const float* magnitudes, size_t dim, int max_code, double start, double end
);

}  // namespace rabitqlib::simd
