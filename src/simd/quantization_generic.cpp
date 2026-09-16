#include <cstddef>

#include "rabitqlib/simd/quantization_dispatch.hpp"
#include "rescale_search.hpp"

namespace rabitqlib::simd {
// Request the existing scalar event sweep; it certifies ambiguous SIMD searches.
double best_rescale_factor_generic(const float*, size_t, int, double, double) { return -1; }
namespace detail {

RescaleScratch& get_thread_local_rescale_scratch(size_t dim) {
    // Search evaluation is synchronous and does not re-enter the quantizer.
    // Retain the largest buffers on each worker; smaller vectors overwrite only
    // their active prefix without shrinking or zero-initializing the storage.
    thread_local RescaleScratch scratch;
    if (scratch.magnitudes.size() < dim)
        scratch.magnitudes.resize(dim);
    if (scratch.reciprocals.size() < dim)
        scratch.reciprocals.resize(dim);
    return scratch;
}

}  // namespace detail
}  // namespace rabitqlib::simd
