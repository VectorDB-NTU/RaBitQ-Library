#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib::symqg::detail {

// A temporary graph. No vectors, distances, or clustering scratch survive export.
struct InitialGraph {
    std::vector<size_t> offsets;
    std::vector<PID> neighbors;
};

InitialGraph build_initial_graph(
    const float* data,
    size_t count,
    size_t dim,
    size_t degree,
    MetricType metric = METRIC_L2,
    size_t num_threads = std::numeric_limits<size_t>::max()
);

}  // namespace rabitqlib::symqg::detail
