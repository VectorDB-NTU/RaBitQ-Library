#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <thread>
#include <vector>

#include "rabitqlib/utils/tools.hpp"

#if defined(__linux__)
#include <sched.h>
#endif

namespace rabitqlib {

size_t total_threads() {
#if defined(_OPENMP) && _OPENMP >= 201511
    if (omp_get_proc_bind() != omp_proc_bind_false && omp_get_num_places() > 0) {
        // The runtime may pin the caller to one place. Count its whole partition,
        // including overlapping places only once, rather than that single mask.
        std::vector<int> places(static_cast<size_t>(omp_get_partition_num_places()));
        omp_get_partition_place_nums(places.data());
        std::vector<int> cpus;
        for (int place : places) {
            const size_t begin = cpus.size();
            cpus.resize(begin + static_cast<size_t>(omp_get_place_num_procs(place)));
            omp_get_place_proc_ids(place, cpus.data() + begin);
        }
        std::sort(cpus.begin(), cpus.end());
        cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
        if (!cpus.empty()) {
            return cpus.size();
        }
    }
#elif defined(_OPENMP) && _OPENMP >= 201307
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        // OpenMP 4.0 has binding but no place-partition query routines.
        const int count = omp_get_num_procs();
        if (count > 0) {
            return static_cast<size_t>(count);
        }
    }
#endif
#if defined(__linux__)
    // The kernel's CPU mask can exceed CPU_SETSIZE on large systems.
    std::vector<cpu_set_t> mask(1);
    while (true) {
        const size_t bytes = mask.size() * sizeof(cpu_set_t);
        if (sched_getaffinity(0, bytes, mask.data()) == 0) {
            const int count = CPU_COUNT_S(bytes, mask.data());
            if (count > 0) {
                return static_cast<size_t>(count);
            }
            break;
        }
        if (errno != EINVAL) {
            break;
        }
        mask.resize(mask.size() * 2);
    }
#endif
    const auto threads = std::thread::hardware_concurrency();
    return threads == 0 ? 1 : threads;
}

}  // namespace rabitqlib
