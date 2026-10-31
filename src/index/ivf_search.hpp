#pragma once

#include <cstddef>

#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib::ivf::detail {
void insert_candidates(
    buffer::SearchBuffer<float>& result_pool,
    const PID* ids,
    const float* distances,
    size_t count
);
}  // namespace rabitqlib::ivf::detail
