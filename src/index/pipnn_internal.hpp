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
    // Buffer sizes in float elements.
    size_t points_buffer_size;
    size_t distances_buffer_size;
    size_t worker_buffer_size;
    std::unique_ptr<float[]> storage;

    ScratchPool(size_t count, size_t dim, size_t num_workers);
    Workspace worker(size_t id);
};

void gather(const float*, size_t, const Bucket&, Workspace&);
void pairwise(Workspace&, size_t, size_t, MetricType);
// After pairwise(), write two sorted (distance, ID) candidates per local row.
// Excludes self; a missing second neighbor retains kPidMax.
void select_local_neighbors(const Bucket&, Workspace&);
std::vector<Bucket> cluster(const float*, size_t, size_t, MetricType, size_t, ScratchPool&);

}  // namespace rabitqlib::symqg::detail::pipnn_impl
