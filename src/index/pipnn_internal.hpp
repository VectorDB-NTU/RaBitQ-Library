#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/types.hpp"

// Private declarations shared only by the implementation and reference tests.
namespace rabitqlib::symqg::detail {
inline constexpr size_t kLeafSize = 1024;
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

    ScratchPool(size_t num_points, size_t dim, size_t num_workers);
    Workspace worker(size_t worker_id);
};

void gather(const float* data, size_t dim, const Bucket& ids, Workspace& work);
void pairwise(Workspace& work, size_t num_points, size_t dim, MetricType metric_type);
// After pairwise(), write two sorted (distance, ID) candidates per local row.
// Excludes self; a missing second neighbor retains kPidMax.
void select_local_neighbors(const Bucket& ids, Workspace& work);
std::vector<Bucket> cluster(
    const float* data,
    size_t num_points,
    size_t dim,
    MetricType metric_type,
    size_t num_workers,
    ScratchPool& scratch
);

}  // namespace rabitqlib::symqg::detail
