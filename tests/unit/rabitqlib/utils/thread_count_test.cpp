#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>

#if defined(__linux__)
#include <sched.h>
#endif

#include <atomic>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rabitqlib/index/detail/parallel_for.hpp"
#include "rabitqlib/utils/tools.hpp"

TEST(ThreadCount, AutomaticAndOversizedRequestsUseAvailableCount) {
    const size_t available = rabitqlib::total_threads();
    ASSERT_GE(available, 1U);
    EXPECT_EQ(
        rabitqlib::resolve_num_threads(0),
        std::min(available, static_cast<size_t>(omp_get_max_threads()))
    );
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

// Applications such as txtai set OMP_NUM_THREADS=1 on macOS so that coexisting
// OpenMP runtimes stay single-threaded; automatic teams must respect that limit.
TEST(ThreadCount, AutomaticRequestsRespectOpenMpThreadLimit) {
    const int original = omp_get_max_threads();
    struct RestoreLimit {
        int limit;
        ~RestoreLimit() { omp_set_num_threads(limit); }
    } restore{original};
    const size_t available = rabitqlib::total_threads();

    omp_set_num_threads(1);
    EXPECT_EQ(rabitqlib::default_threads(), 1U);
    EXPECT_EQ(rabitqlib::resolve_num_threads(0), 1U);
    EXPECT_EQ(rabitqlib::detail::openmp_team_size(0, 100), 1);
    rabitqlib::index_detail::parallel_for(0, 19, 0, [](size_t, size_t worker) {
        EXPECT_EQ(worker, 0U);
    });
    // Explicit requests are capped only by the available CPUs.
    EXPECT_EQ(rabitqlib::resolve_num_threads(2), std::min<size_t>(2, available));
    EXPECT_EQ(
        rabitqlib::resolve_num_threads(std::numeric_limits<size_t>::max()), available
    );

    if (available >= 2) {
        omp_set_num_threads(2);
        EXPECT_EQ(rabitqlib::resolve_num_threads(0), 2U);
    }
    omp_set_num_threads(static_cast<int>(available) + 1);
    EXPECT_EQ(rabitqlib::resolve_num_threads(0), available);
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
    const size_t automatic =
        std::min(available, static_cast<size_t>(omp_get_max_threads()));
    for (size_t requested : {size_t{0}, size_t{1}, std::numeric_limits<size_t>::max()}) {
        const size_t expected =
            requested == 1 ? 1 : (requested == 0 ? automatic : available);
        EXPECT_EQ(rabitqlib::resolve_num_threads(requested), expected);
        rabitqlib::index_detail::parallel_for(0, 19, requested, [&](size_t, size_t worker) {
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
    // A binding policy can be set even when the runtime exposes no CPU places
    // (for example, LLVM libomp on macOS).
    const int places = omp_get_num_places();
    const size_t threads = std::min<size_t>(available, 4);
    rabitqlib::index_detail::parallel_for(0, 257, threads, [&](size_t, size_t worker) {
        EXPECT_LT(worker, threads);
        if (places > 0) {
            EXPECT_GE(omp_get_place_num(), 0);
            EXPECT_LT(omp_get_place_num(), places);
        } else {
            EXPECT_EQ(omp_get_place_num(), -1);
        }
        if (threads > 1) {
            EXPECT_GT(omp_get_level(), 0);
        }
    });
    EXPECT_THROW(
        rabitqlib::index_detail::parallel_for(
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
        rabitqlib::index_detail::parallel_for(
            0, count, requested, [&](size_t i, size_t worker) {
                EXPECT_LT(worker, requested == 1 ? 1U : available);
                visits[i].fetch_add(1);
            }
        );
        for (const auto& visited : visits) {
            EXPECT_EQ(visited.load(), 1U);
        }
        size_t calls = 0;
        rabitqlib::index_detail::parallel_for(
            7, 8, requested, [&](size_t i, size_t worker) {
                EXPECT_EQ(i, 7U);
                EXPECT_EQ(worker, 0U);
                ++calls;
            }
        );
        EXPECT_EQ(calls, 1U);
        for (size_t start : {0U, 5U}) {
            rabitqlib::index_detail::parallel_for(
                start, start, requested, [](size_t, size_t) {
                    ADD_FAILURE() << "An empty range must not invoke the callback";
                }
            );
        }
    }
}

TEST(ThreadCount, TopLevelParallelForReusesWorkersOnlyWhenRequested) {
#if defined(_OPENMP) && _OPENMP >= 201307
    if (rabitqlib::resolve_num_threads(2) < 2) {
        GTEST_SKIP() << "Requires two available workers";
    }
    const int max_threads = omp_get_max_threads();
    const int max_levels = omp_get_max_active_levels();
    std::vector<int> levels(17, -1);
    rabitqlib::index_detail::parallel_for(0, levels.size(), 2, [&](size_t i, size_t) {
        levels[i] = omp_get_level();
    });
    const int build_level = omp_get_proc_bind() == omp_proc_bind_false ? 0 : 1;
    for (int level : levels) {
        EXPECT_EQ(level, build_level);
    }
    for (size_t repeat = 0; repeat < 3; ++repeat) {
        rabitqlib::index_detail::parallel_for(
            0,
            levels.size(),
            2,
            [&](size_t i, size_t) { levels[i] = omp_get_level(); },
            true
        );
        for (int level : levels) {
            EXPECT_EQ(level, 1);
        }
    }
    rabitqlib::index_detail::parallel_for(
        0,
        levels.size(),
        1,
        [&](size_t i, size_t worker) {
            EXPECT_EQ(worker, 0U);
            levels[i] = omp_get_level();
        },
        true
    );
    for (int level : levels) {
        EXPECT_EQ(level, 0);
    }
    EXPECT_THROW(
        rabitqlib::index_detail::parallel_for(
            0,
            levels.size(),
            2,
            [](size_t, size_t) { throw std::runtime_error("failed"); },
            true
        ),
        std::runtime_error
    );
    EXPECT_EQ(omp_get_max_threads(), max_threads);
    EXPECT_EQ(omp_get_max_active_levels(), max_levels);
#else
    GTEST_SKIP() << "Runtime worker reuse requires OpenMP 4.0";
#endif
}

TEST(ThreadCount, NestedParallelForPreservesBindingAndWorkerIds) {
#if defined(_OPENMP) && _OPENMP >= 201307
    if (rabitqlib::resolve_num_threads(2) < 2) {
        GTEST_SKIP() << "Requires two available workers";
    }
    const bool bound = omp_get_proc_bind() != omp_proc_bind_false;
    std::vector<int> levels(17, -1);
    std::vector<size_t> workers(levels.size(), 2);
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        rabitqlib::index_detail::parallel_for(
            0,
            levels.size(),
            2,
            [&](size_t i, size_t worker) {
                levels[i] = omp_get_level();
                workers[i] = worker;
            },
            true
        );
    }
    for (size_t i = 0; i < levels.size(); ++i) {
        EXPECT_EQ(levels[i], bound ? 2 : 0);
        EXPECT_LT(workers[i], 2U);
    }
#else
    GTEST_SKIP() << "OpenMP binding requires OpenMP 4.0";
#endif
}

TEST(ThreadCount, OmpForDynamicVisitsEachItemOnceAndPropagatesErrors) {
    constexpr size_t kCount = 257;
    for (size_t workers : {1U, 4U}) {
        SCOPED_TRACE(workers);
        std::vector<std::atomic<size_t>> visits(kCount);
        for (auto& visited : visits) {
            visited.store(0);
        }
        rabitqlib::index_detail::omp_for_dynamic(
            kCount, workers, [&](size_t i, size_t worker) {
                EXPECT_LT(worker, workers);
                visits[i].fetch_add(1);
            }
        );
        for (const auto& visited : visits) {
            EXPECT_EQ(visited.load(), 1U);
        }

        std::atomic<size_t> calls{0};
        EXPECT_THROW(
            rabitqlib::index_detail::omp_for_dynamic(
                kCount,
                workers,
                [&](size_t i, size_t) {
                    calls.fetch_add(1);
                    if (i == 3) {
                        throw std::runtime_error("failed");
                    }
                }
            ),
            std::runtime_error
        );
        // Items after a failure are skipped; the serial order stops at the first one.
        if (workers == 1) {
            EXPECT_EQ(calls.load(), 4U);
        }
    }
}

TEST(ThreadCount, OpenMpTeamSizeIsPositiveAndBounded) {
    const size_t available = rabitqlib::total_threads();
    EXPECT_EQ(rabitqlib::detail::openmp_team_size(0, 0), 1);
    EXPECT_EQ(rabitqlib::detail::openmp_team_size(1, 100), 1);
    EXPECT_EQ(rabitqlib::detail::openmp_team_size(0, 1), 1);
    EXPECT_EQ(
        rabitqlib::detail::openmp_team_size(
            std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()
        ),
        static_cast<int>(std::min<size_t>(available, std::numeric_limits<int>::max()))
    );
    EXPECT_EQ(
        rabitqlib::detail::openmp_team_size(0, std::numeric_limits<size_t>::max()),
        static_cast<int>(rabitqlib::default_threads())
    );
}
