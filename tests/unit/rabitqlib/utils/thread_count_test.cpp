#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <limits>
#include <vector>

#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/utils/tools.hpp"

TEST(ThreadCount, AutomaticAndOversizedRequestsUseHardwareCount) {
    const size_t hardware = rabitqlib::total_threads();
    ASSERT_GE(hardware, 1U);
    EXPECT_EQ(rabitqlib::resolve_num_threads(0), hardware);
    EXPECT_EQ(rabitqlib::resolve_num_threads(1), 1U);
    EXPECT_EQ(rabitqlib::resolve_num_threads(hardware), hardware);
    EXPECT_EQ(rabitqlib::resolve_num_threads(hardware + 1), hardware);
    EXPECT_EQ(rabitqlib::resolve_num_threads(std::numeric_limits<size_t>::max()), hardware);
    if (hardware > 1) {
        EXPECT_EQ(rabitqlib::resolve_num_threads(hardware - 1), hardware - 1);
    }
}

TEST(ThreadCount, ParallelForVisitsEachItemOnceWithBoundedWorkerIds) {
    const size_t hardware = rabitqlib::total_threads();
    const size_t count = 2 * hardware + 3;
    for (size_t requested : {size_t{0}, size_t{1}, std::numeric_limits<size_t>::max()}) {
        SCOPED_TRACE(requested);
        std::vector<std::atomic<size_t>> visits(count);
        for (auto& visited : visits) {
            visited.store(0);
        }
        rabitqlib::ivf::parallel_for(0, count, requested, [&](size_t i, size_t worker) {
            EXPECT_LT(worker, requested == 1 ? 1U : hardware);
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
