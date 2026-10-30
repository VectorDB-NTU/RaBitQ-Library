#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/openmp.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib {
template <typename T>
inline PID exact_nn(
    const T* data,
    const T* query,
    size_t num_points,
    size_t dim,
    size_t num_threads,
    T (*dist_func)(const T*, const T*, size_t)
) {
#if RABITQ_USE_OPENMP
    const int num_workers = detail::openmp_team_size(num_threads, num_points);
#else
    (void)num_threads;
    constexpr int num_workers = 1;
#endif
    std::vector<AnnCandidate<T, PID>> best_entries(num_workers);

#if RABITQ_USE_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(num_workers)
#endif
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_points);
         ++index) {
        const size_t i = static_cast<size_t>(index);
#if RABITQ_USE_OPENMP
        const int tid = omp_get_thread_num();
#else
        constexpr int tid = 0;
#endif
        AnnCandidate<T, PID>& cur_entry = best_entries[tid];
        const T* cur_data = data + (dim * i);

        T distance = dist_func(cur_data, query, dim);
        if (distance < cur_entry.distance) {
            cur_entry.id = static_cast<PID>(i);
            cur_entry.distance = distance;
        }
    }

    PID nearest_neighbor = 0;
    T min_dist = std::numeric_limits<T>::max();
    for (auto& candi : best_entries) {
        if (candi.distance < min_dist) {
            nearest_neighbor = candi.id;
            min_dist = candi.distance;
        }
    }
    return nearest_neighbor;
}

}  // namespace rabitqlib
