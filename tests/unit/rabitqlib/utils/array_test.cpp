#include "rabitqlib/utils/array.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace {
struct Allocations {
    std::map<void*, int> owners;
    bool mismatch = false;
    bool fail = false;
};

template <bool Propagate>
struct TrackedAllocator {
    using value_type = int;
    using propagate_on_container_move_assignment = std::bool_constant<Propagate>;
    using is_always_equal = std::false_type;
    Allocations* allocations;
    int id;

    int* allocate(size_t count) {
        if (allocations->fail) {
            throw std::bad_alloc();
        }
        auto* ptr = std::allocator<int>{}.allocate(count);
        allocations->owners[ptr] = id;
        return ptr;
    }
    void deallocate(int* ptr, size_t count) {
        allocations->mismatch |= allocations->owners.at(ptr) != id;
        allocations->owners.erase(ptr);
        std::allocator<int>{}.deallocate(ptr, count);
    }
    bool operator!=(const TrackedAllocator& other) const { return id != other.id; }
};

template <bool Propagate>
void check_move(int destination_id) {
    Allocations state;
    using Array = rabitqlib::Array<int, std::array<size_t, 1>, TrackedAllocator<Propagate>>;
    static_assert(std::is_nothrow_move_assignable_v<Array> == Propagate);
    {
        Array source({3}, {&state, 1});
        Array target({2}, {&state, destination_id});
        source[0] = 17;
        source[1] = 23;
        source[2] = 42;
        auto* previous = source.data();
        target = std::move(source);
        EXPECT_EQ(target[0], 17);
        EXPECT_EQ(target[1], 23);
        EXPECT_EQ(target[2], 42);
        // A successful move explicitly leaves the source pointer null.
        // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        EXPECT_EQ(source.data(), nullptr);
        EXPECT_EQ(target.data() == previous, Propagate || destination_id == 1);
        auto& alias = target;
        target = std::move(alias);
        EXPECT_EQ(target[2], 42);
    }
    EXPECT_TRUE(state.owners.empty());
    EXPECT_FALSE(state.mismatch);
}

TEST(ArrayOwnership, MoveAssignmentHonorsAllocator) {
    check_move<false>(1);
    check_move<false>(2);
    check_move<true>(1);
    check_move<true>(2);
}

TEST(ArrayOwnership, FailedUnequalMovePreservesBothArrays) {
    Allocations state;
    using Array = rabitqlib::Array<int, std::array<size_t, 1>, TrackedAllocator<false>>;
    {
        Array source({3}, {&state, 1}), target({2}, {&state, 2});
        source[0] = 17;
        target[0] = 23;
        const auto* source_ptr = source.data();
        const auto* target_ptr = target.data();
        state.fail = true;
        EXPECT_THROW(target = std::move(source), std::bad_alloc);
        state.fail = false;
        // The failed allocation must leave the source untouched.
        // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        EXPECT_EQ(source.data(), source_ptr);
        EXPECT_EQ(target.data(), target_ptr);
        EXPECT_EQ(source[0], 17);
        EXPECT_EQ(target[0], 23);
    }
    EXPECT_TRUE(state.owners.empty());
    EXPECT_FALSE(state.mismatch);
}
}  // namespace
