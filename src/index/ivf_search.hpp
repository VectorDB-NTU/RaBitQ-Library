#pragma once

#include <cstddef>
#include <vector>

#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib::ivf::detail {
void insert_candidates(
    buffer::SearchBuffer<float>& result_pool,
    const PID* ids,
    const float* distances,
    size_t count
);

struct TopNAssignmentScratch {
    std::vector<float> products;
    std::vector<double> point_norms;
};

// Inputs are validated, contiguous rows in the same padded/rotated domain.
// centroid_norms contains k double-precision squared norms, computed once per
// batch. The caller bounds n and reuses scratch across blocks on each worker.
// Run inside an OpenMP task with omp_set_num_threads(1), even for one worker,
// so Eigen cannot start a nested team or change the caller's thread settings.
// Unlike exact_assign's clustering objective, return the existing float metric
// (sqrt for L2) and break ties by ID, exactly as FlatInitializer does.
void exact_assign_topn(
    const float* x,
    const float* centroids,
    size_t n,
    size_t d,
    size_t k,
    size_t nprobe,
    bool spherical,
    const double* centroid_norms,
    AnnCandidate<float>* candidates,
    TopNAssignmentScratch& scratch
);
}  // namespace rabitqlib::ivf::detail
