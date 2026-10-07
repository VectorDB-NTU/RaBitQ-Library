#include <gtest/gtest.h>
#include <omp.h>

#if defined(__linux__)
#include <sched.h>
#endif

#include <atomic>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/utils/tools.hpp"

TEST(ThreadCount, AutomaticAndOversizedRequestsUseAvailableCount) {
    const size_t available = rabitqlib::total_threads();
    ASSERT_GE(available, 1U);
    EXPECT_EQ(rabitqlib::resolve_num_threads(0), available);
    EXPECT_EQ(rabitqlib::resolve_num_threads(1), 1U);
    EXPECT_EQ(rabitqlib::resolve_num_threads(available), available);
    EXPECT_EQ(rabitqlib::resolve_num_threads(available + 1), available);
    EXPECT_EQ(
        rabitqlib::resolve_num_threads(std::numeric_limits<size_t>::max()), available
    );
    if (available > 1) {
        EXPECT_EQ(rabitqlib::resolve_num_threads(available - 1), available - 1);
    }
}

#if defined(__linux__)
TEST(ThreadCount, AutomaticAndExplicitRequestsRespectAffinity) {
#if defined(_OPENMP) && _OPENMP >= 201307
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        GTEST_SKIP() << "OpenMP owns placement when binding is enabled";
    }
#endif
    cpu_set_t original;
    if (sched_getaffinity(0, sizeof(original), &original) != 0) {
        GTEST_SKIP() << "Cannot read affinity";
    }
    struct RestoreAffinity {
        cpu_set_t mask;
        ~RestoreAffinity() { EXPECT_EQ(sched_setaffinity(0, sizeof(mask), &mask), 0); }
    } restore{original};
    cpu_set_t restricted;
    CPU_ZERO(&restricted);
    size_t available = 0;
    for (size_t cpu = 0; cpu < CPU_SETSIZE && available < 2; ++cpu) {
        if (CPU_ISSET(cpu, &original)) {
            CPU_SET(cpu, &restricted);
            ++available;
        }
    }
    ASSERT_GT(available, 0U);
    ASSERT_EQ(sched_setaffinity(0, sizeof(restricted), &restricted), 0);
    EXPECT_EQ(rabitqlib::total_threads(), available);
    for (size_t requested : {size_t{0}, size_t{1}, std::numeric_limits<size_t>::max()}) {
        const size_t expected = requested == 1 ? 1 : available;
        EXPECT_EQ(rabitqlib::resolve_num_threads(requested), expected);
        rabitqlib::ivf::parallel_for(0, 19, requested, [&](size_t, size_t worker) {
            EXPECT_LT(worker, expected);
        });
    }
}
#endif

TEST(ThreadCount, BoundParallelForUsesRuntimePlacementAndPropagatesErrors) {
#if defined(_OPENMP) && _OPENMP >= 201511
    if (omp_get_proc_bind() == omp_proc_bind_false) {
        GTEST_SKIP() << "Run with OpenMP binding enabled";
    }
    const size_t available = rabitqlib::total_threads();
    ASSERT_GT(available, 0U);
    const size_t threads = std::min<size_t>(available, 4);
    rabitqlib::ivf::parallel_for(0, 257, threads, [&](size_t, size_t worker) {
        EXPECT_LT(worker, threads);
        EXPECT_GE(omp_get_place_num(), 0);
        if (threads > 1) {
            EXPECT_GT(omp_get_level(), 0);
        }
    });
    EXPECT_THROW(
        rabitqlib::ivf::parallel_for(
            0, 257, threads, [](size_t, size_t) { throw std::runtime_error("failed"); }
        ),
        std::runtime_error
    );
#else
    GTEST_SKIP() << "OpenMP places require OpenMP 4.5";
#endif
}

TEST(ThreadCount, ParallelForVisitsEachItemOnceWithBoundedWorkerIds) {
    const size_t available = rabitqlib::total_threads();
    const size_t count = 2 * available + 3;
    for (size_t requested : {size_t{0}, size_t{1}, std::numeric_limits<size_t>::max()}) {
        SCOPED_TRACE(requested);
        std::vector<std::atomic<size_t>> visits(count);
        for (auto& visited : visits) {
            visited.store(0);
        }
        rabitqlib::ivf::parallel_for(0, count, requested, [&](size_t i, size_t worker) {
            EXPECT_LT(worker, requested == 1 ? 1U : available);
            visits[i].fetch_add(1);
        });
        for (const auto& visited : visits) {
            EXPECT_EQ(visited.load(), 1U);
        }
        size_t calls = 0;
        rabitqlib::ivf::parallel_for(7, 8, requested, [&](size_t i, size_t worker) {
            EXPECT_EQ(i, 7U);
            EXPECT_EQ(worker, 0U);
            ++calls;
        });
        EXPECT_EQ(calls, 1U);
        rabitqlib::ivf::parallel_for(0, 0, requested, [](size_t, size_t) {
            ADD_FAILURE() << "An empty range must not invoke the callback";
        });
    }
}
