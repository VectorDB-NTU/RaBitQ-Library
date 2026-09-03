#include "rabitqlib/utils/memory.hpp"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

TEST(AlignedAllocator, SupportsVectorMoveAssignment) {
    using Allocator = rabitqlib::memory::AlignedAllocator<int>;
    std::vector<int, Allocator> source{1, 2, 3}, destination;
    destination = std::move(source);
    ASSERT_EQ(destination.size(), 3U);
    EXPECT_EQ(destination[0], 1);
    EXPECT_EQ(destination[2], 3);
}
