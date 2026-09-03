#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rabitqlib::parallel {
inline int thread_index() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

inline void set_thread_count(int count) {
#ifdef _OPENMP
    omp_set_num_threads(count);
#else
    (void)count;
#endif
}
}  // namespace rabitqlib::parallel
