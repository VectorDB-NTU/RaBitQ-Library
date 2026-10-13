#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/types.hpp"

// Private declarations shared only by the implementation and reference tests.
namespace rabitqlib::symqg::detail::pipnn_impl {
constexpr size_t kLeafSize = 1024;
using Bucket = std::vector<PID>;

// Non-owning slices of a build-scoped allocation. Large matrices are released
// after seeding rather than retained in worker heaps across graph resets.
struct Workspace {
    float* points;
    float* distances;
    float* norms;
    std::vector<std::pair<float, PID>> candidates;
};

struct ScratchPool {
    size_t point_size;
    size_t distance_size;
    size_t stride;
    std::unique_ptr<float[]> storage;

    ScratchPool(size_t count, size_t dim, size_t threads);
    Workspace worker(size_t id);
};

void gather(const float*, size_t, const Bucket&, Workspace&);
void pairwise(Workspace&, size_t, size_t, MetricType);
std::vector<Bucket> cluster(const float*, size_t, size_t, MetricType, size_t, ScratchPool&);

}  // namespace rabitqlib::symqg::detail::pipnn_impl
