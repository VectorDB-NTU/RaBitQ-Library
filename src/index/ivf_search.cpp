#include <cstddef>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/utils/buffer.hpp"

namespace rabitqlib::ivf::detail {
// Keep the scalar top-k loop out of the large inlined IVF search body. This avoids
// register spills in wheel builds without changing candidate order or tie handling.
void insert_candidates(
    buffer::SearchBuffer<float>& knns, const PID* ids, const float* distances, size_t count
) {
    for (size_t i = 0; i < count; ++i) {
        knns.insert(ids[i], distances[i]);
    }
}
}  // namespace rabitqlib::ivf::detail
