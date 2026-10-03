#include "rabitqlib/utils/memory.hpp"

#include <gtest/gtest.h>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
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

TEST(AlignedAllocatorTest, ValueInitializesScalarsByDefault) {
    const std::vector<float, AlignedAllocator<float>> values(17);
    for (float value : values) {
        EXPECT_EQ(value, 0.0F);
    }
}

TEST(AlignedAllocatorTest, DefaultInitializationPreservesObjectConstructionAndCopies) {
    using Storage = std::vector<std::string, DefaultInitAlignedAllocator<std::string>>;
    Storage values(3);
    for (const auto& value : values) {
        EXPECT_TRUE(value.empty());
    }
    values[0] = "graph";
    values.resize(17, "row");
    const Storage copy(values);
    EXPECT_EQ(copy, values);
    EXPECT_EQ(copy[0], "graph");
    EXPECT_EQ(copy.back(), "row");
    EXPECT_EQ(reinterpret_cast<uintptr_t>(copy.data()) % 64, 0U);
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

TEST(AlignedAllocationTest, HugePageStorageHandlesSizeBoundaries) {
    EXPECT_EQ(huge_page_allocate<char>(0), nullptr);
    AlignedAllocator<uint64_t, 64, true> allocator;
    EXPECT_EQ(allocator.allocate(0), nullptr);

    for (size_t bytes :
         {size_t{1}, size_t{65}, kHugePageSize - 1, kHugePageSize, kHugePageSize + 1}) {
        std::unique_ptr<char, decltype(&aligned_deallocate)> storage(
            huge_page_allocate<char>(bytes), aligned_deallocate
        );
        ASSERT_NE(storage, nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(storage.get()) % 64, 0U);
        std::fill_n(storage.get(), bytes, 'q');
        EXPECT_EQ(storage.get()[bytes - 1], 'q');

        // Exercise the element-count API as well as the byte-count API.
        const size_t count = (bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
        std::vector<uint64_t, AlignedAllocator<uint64_t, 64, true>> values(count, 42);
        EXPECT_EQ(values.front(), 42U);
        EXPECT_EQ(values.back(), 42U);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(values.data()) % 64, 0U);
#if defined(__linux__)
        if (bytes >= kHugePageSize) {
            EXPECT_EQ(reinterpret_cast<uintptr_t>(storage.get()) % kHugePageSize, 0U);
        }
        if (count * sizeof(uint64_t) >= kHugePageSize) {
            EXPECT_EQ(reinterpret_cast<uintptr_t>(values.data()) % kHugePageSize, 0U);
        }
#endif
    }
}

TEST(AlignedAllocationTest, SmallHugePageStorageAvoidsMegabytePadding) {
#if defined(__GLIBC__)
    // Inspect the allocated capacity, not pointer alignment, which can happen
    // to exceed the request. This catches rounding small buffers to 2/4 MiB.
    std::unique_ptr<char, decltype(&aligned_deallocate)> storage(
        huge_page_allocate<char>(65), aligned_deallocate
    );
    EXPECT_LT(malloc_usable_size(storage.get()), kHugePageSize);
    std::vector<float, DefaultInitAlignedAllocator<float, 64, true>> rows(17);
    EXPECT_LT(malloc_usable_size(rows.data()), kHugePageSize);
    std::vector<uint64_t, AlignedAllocator<uint64_t, 64, true>> ids(17);
    EXPECT_LT(malloc_usable_size(ids.data()), kHugePageSize);
#else
    GTEST_SKIP() << "Allocated-capacity inspection requires glibc";
#endif
}

TEST(AlignedAllocationTest, HugePageStoragePreservesExplicitAlignmentAndOverflowChecks) {
    struct alignas(128) OverAligned {
        char value;
    };
    std::unique_ptr<OverAligned, decltype(&aligned_deallocate)> storage(
        huge_page_allocate<OverAligned>(sizeof(OverAligned)), aligned_deallocate
    );
    EXPECT_EQ(reinterpret_cast<uintptr_t>(storage.get()) % alignof(OverAligned), 0U);
    EXPECT_THROW(
        (huge_page_allocate<char>(std::numeric_limits<size_t>::max())),
        std::bad_array_new_length
    );
    AlignedAllocator<uint64_t, 64, true> allocator;
    EXPECT_THROW(
        (void)allocator.allocate(std::numeric_limits<size_t>::max() / sizeof(uint64_t) + 1),
        std::bad_array_new_length
    );
#if defined(__linux__)
    // Fits 64-byte rounding but overflows when huge-page alignment is selected.
    EXPECT_THROW(
        (huge_page_allocate<char>(std::numeric_limits<size_t>::max() - 63)),
        std::bad_array_new_length
    );
#endif
}

TEST(ArrayTest, RejectsDimensionProductOverflow) {
    EXPECT_THROW(
        (Array<char>(std::vector<size_t>{std::numeric_limits<size_t>::max(), 2})),
        std::bad_array_new_length
    );
}

}  // namespace
}  // namespace rabitqlib::memory
