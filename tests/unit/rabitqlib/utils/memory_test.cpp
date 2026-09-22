#include "rabitqlib/utils/memory.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "rabitqlib/utils/array.hpp"

namespace rabitqlib::memory {
namespace {

TEST(AlignedAllocatorTest, RejectsSizeThatOverflowsAlignmentRounding) {
    AlignedAllocator<char, 64> allocator;
    EXPECT_THROW(
        (void)allocator.allocate(std::numeric_limits<size_t>::max()),
        std::bad_array_new_length
    );
}

TEST(AlignedAllocationTest, RejectsSizeThatOverflowsAlignmentRounding) {
    EXPECT_THROW(
        (align_allocate<64, char>(std::numeric_limits<size_t>::max())),
        std::bad_array_new_length
    );
}

TEST(AlignedAllocationTest, UsesMatchingDeallocator) {
    auto* ptr = align_allocate<64, char>(65);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0U);
    aligned_deallocate(ptr);

    AlignedAllocator<int, 64> allocator;
    auto* values = allocator.allocate(3);
    ASSERT_NE(values, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(values) % 64, 0U);
    allocator.deallocate(values, 3);
}

TEST(ArrayTest, RejectsDimensionProductOverflow) {
    EXPECT_THROW(
        (Array<char>(std::vector<size_t>{std::numeric_limits<size_t>::max(), 2})),
        std::bad_array_new_length
    );
}

}  // namespace
}  // namespace rabitqlib::memory
