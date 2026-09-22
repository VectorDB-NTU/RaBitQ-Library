#include <cstddef>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/simd/fastscan_dispatch.hpp"

namespace rabitqlib::fastscan::simd {
void pack_lut_generic(size_t dim, const float* query, float* lut) {
    // Keep the initial +0 addition observable; MSVC otherwise turns +0 + -0 into -0.
    volatile float zero = 0;
    for (size_t group = 0; group < dim / 4; ++group) {
        lut[0] = zero;
        for (size_t j = 1; j < 16; ++j) {
            lut[j] = lut[j - LOWBIT(j)] + query[kPos[j]];
        }
        query += 4;
        lut += 16;
    }
}
}  // namespace rabitqlib::fastscan::simd
