#include "rabitqlib/index/ivf/initializer.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <stdexcept>

namespace rabitqlib::ivf {
namespace {

TEST(ParallelForTest, AutomaticThreadCountProcessesEveryItem) {
    std::atomic<size_t> calls{0};
    parallel_for(0, 100, 0, [&](size_t, size_t) { ++calls; });
    EXPECT_EQ(calls, 100U);
}

TEST(ParallelForTest, EmptyRangeDoesNotInvokeFunction) {
    size_t calls = 0;
    parallel_for(5, 5, 0, [&](size_t, size_t) { ++calls; });
    EXPECT_EQ(calls, 0U);
}

TEST(ParallelForTest, JoinsWorkersBeforeRethrowingFunctionException) {
    std::atomic<size_t> active{0};
    EXPECT_THROW(
        parallel_for(
            0,
            100,
            4,
            [&](size_t id, size_t) {
                ++active;
                --active;
                if (id == 0) {
                    throw std::runtime_error("parallel failure");
                }
            }
        ),
        std::runtime_error
    );
    EXPECT_EQ(active, 0U);
}

}  // namespace
}  // namespace rabitqlib::ivf
