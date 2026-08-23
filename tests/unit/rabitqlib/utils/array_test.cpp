#include "rabitqlib/utils/array.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using rabitqlib::Array;
using rabitqlib::memory::AlignedAllocator;
using rabitqlib::memory::Allocator;

static_assert(!std::is_copy_constructible_v<Array<int>>);
static_assert(!std::is_copy_assignable_v<Array<int>>);

template <typename T>
class StatefulAllocator {
   public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::false_type;
    using is_always_equal = std::false_type;

    template <typename U>
    struct rebind {
        using other = StatefulAllocator<U>;
    };

    StatefulAllocator() = default;
    explicit StatefulAllocator(int id) : id_(id) {}

    template <typename U>
    explicit StatefulAllocator(const StatefulAllocator<U>& other) : id_(other.id()) {}

    [[nodiscard]] T* allocate(size_t count) {
        T* pointer = std::allocator<T>{}.allocate(count);
        owners()[pointer] = id_;
        ++allocation_calls_storage();
        return pointer;
    }

    void deallocate(T* pointer, size_t count) {
        const auto owner = owners().find(pointer);
        if (owner == owners().end() || owner->second != id_) {
            ++deallocation_errors();
        } else {
            owners().erase(owner);
        }
        std::allocator<T>{}.deallocate(pointer, count);
    }

    [[nodiscard]] int id() const { return id_; }

    static void clear_tracking() {
        owners().clear();
        deallocation_errors() = 0;
        allocation_calls_storage() = 0;
    }

    [[nodiscard]] static size_t outstanding_allocations() { return owners().size(); }
    [[nodiscard]] static int errors() { return deallocation_errors(); }
    [[nodiscard]] static size_t allocation_calls() { return allocation_calls_storage(); }

    friend bool operator==(
        const StatefulAllocator& first, const StatefulAllocator& second
    ) {
        return first.id_ == second.id_;
    }

    friend bool operator!=(
        const StatefulAllocator& first, const StatefulAllocator& second
    ) {
        return !(first == second);
    }

   private:
    static std::unordered_map<void*, int>& owners() {
        static std::unordered_map<void*, int> value;
        return value;
    }

    static int& deallocation_errors() {
        static int value = 0;
        return value;
    }

    static size_t& allocation_calls_storage() {
        static size_t value = 0;
        return value;
    }

    int id_ = 0;
};

template <typename T>
class PropagatingAllocator {
   public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::false_type;

    template <typename U>
    struct rebind {
        using other = PropagatingAllocator<U>;
    };

    PropagatingAllocator() = default;
    explicit PropagatingAllocator(int id) noexcept : id_(id) {}
    PropagatingAllocator(const PropagatingAllocator&) noexcept = default;
    PropagatingAllocator(PropagatingAllocator&&) noexcept = default;
    PropagatingAllocator& operator=(const PropagatingAllocator&) noexcept = default;
    PropagatingAllocator& operator=(PropagatingAllocator&&) noexcept = default;

    template <typename U>
    explicit PropagatingAllocator(const PropagatingAllocator<U>& other) noexcept
        : id_(other.id()) {}

    [[nodiscard]] T* allocate(size_t count) {
        T* result = std::allocator<T>{}.allocate(count);
        owners()[result] = id_;
        return result;
    }

    void deallocate(T* pointer, size_t count) noexcept {
        const auto owner = owners().find(pointer);
        if (owner == owners().end() || owner->second != id_) {
            ++deallocation_errors();
        } else {
            owners().erase(owner);
        }
        std::allocator<T>{}.deallocate(pointer, count);
    }

    [[nodiscard]] int id() const noexcept { return id_; }

    static void clear_tracking() {
        owners().clear();
        deallocation_errors() = 0;
    }

    [[nodiscard]] static size_t outstanding_allocations() { return owners().size(); }
    [[nodiscard]] static int errors() { return deallocation_errors(); }

    friend bool operator==(
        const PropagatingAllocator& first, const PropagatingAllocator& second
    ) noexcept {
        return first.id_ == second.id_;
    }

    friend bool operator!=(
        const PropagatingAllocator& first, const PropagatingAllocator& second
    ) noexcept {
        return !(first == second);
    }

   private:
    static std::unordered_map<void*, int>& owners() {
        static std::unordered_map<void*, int> value;
        return value;
    }

    static int& deallocation_errors() {
        static int value = 0;
        return value;
    }

    int id_ = 0;
};

static_assert(rabitqlib::array_impl::dimensions_allocator_swap_safe_v<std::vector<size_t>>);
static_assert(rabitqlib::array_impl::dimensions_allocator_swap_safe_v<
              std::array<size_t, 2>>);
static_assert(!rabitqlib::array_impl::dimensions_allocator_swap_safe_v<
              std::vector<size_t, StatefulAllocator<size_t>>>);

class ThrowingMoveDimensions {
   public:
    using value_type = size_t;
    using const_iterator = std::array<size_t, 1>::const_iterator;

    ThrowingMoveDimensions() noexcept = default;
    explicit ThrowingMoveDimensions(size_t size) noexcept : values_{size} {}
    ThrowingMoveDimensions(const ThrowingMoveDimensions&) = default;
    ThrowingMoveDimensions& operator=(const ThrowingMoveDimensions&) = default;

    // Intentionally throwing to verify Array's conditional noexcept contract.
    // NOLINTNEXTLINE(performance-noexcept-move-constructor)
    ThrowingMoveDimensions(ThrowingMoveDimensions&& other) {
        if (throw_on_move_) {
            throw std::runtime_error("dimension move failed");
        }
        values_ = other.values_;
        other.values_ = {};
    }

    ThrowingMoveDimensions& operator=(ThrowingMoveDimensions&& other) noexcept {
        values_ = other.values_;
        other.values_ = {};
        return *this;
    }

    [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return values_.end(); }
    [[nodiscard]] size_t operator[](size_t index) const noexcept { return values_[index]; }

    static void set_throw_on_move(bool value) noexcept { throw_on_move_ = value; }

    friend void swap(
        ThrowingMoveDimensions& first, ThrowingMoveDimensions& second
    ) noexcept {
        first.values_.swap(second.values_);
    }

   private:
    std::array<size_t, 1> values_{};
    static inline bool throw_on_move_ = false;
};

using ThrowingDimensionsArray = Array<int, ThrowingMoveDimensions>;
static_assert(std::is_nothrow_move_assignable_v<ThrowingMoveDimensions>);
static_assert(!std::is_nothrow_move_constructible_v<ThrowingMoveDimensions>);
static_assert(!std::is_nothrow_move_assignable_v<ThrowingDimensionsArray>);

struct AllocationControl {
    bool fail_allocation = false;
    size_t outstanding_allocations = 0;
};

template <typename T>
class ControlledAllocator {
   public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::false_type;
    using is_always_equal = std::false_type;

    template <typename U>
    struct rebind {
        using other = ControlledAllocator<U>;
    };

    ControlledAllocator() : control_(std::make_shared<AllocationControl>()) {}
    ControlledAllocator(std::shared_ptr<AllocationControl> control, int id)
        : control_(std::move(control)), id_(id) {}

    template <typename U>
    ControlledAllocator(const ControlledAllocator<U>& other)
        : control_(other.control()), id_(other.id()) {}

    [[nodiscard]] T* allocate(size_t count) {
        if (control_->fail_allocation) {
            throw std::bad_alloc();
        }
        T* result = std::allocator<T>{}.allocate(count);
        ++control_->outstanding_allocations;
        return result;
    }

    void deallocate(T* pointer, size_t count) noexcept {
        --control_->outstanding_allocations;
        std::allocator<T>{}.deallocate(pointer, count);
    }

    [[nodiscard]] const std::shared_ptr<AllocationControl>& control() const noexcept {
        return control_;
    }
    [[nodiscard]] int id() const noexcept { return id_; }

    template <typename U>
    [[nodiscard]] bool operator==(const ControlledAllocator<U>& other) const noexcept {
        return control_.get() == other.control().get() && id_ == other.id();
    }

    template <typename U>
    [[nodiscard]] bool operator!=(const ControlledAllocator<U>& other) const noexcept {
        return !(*this == other);
    }

   private:
    std::shared_ptr<AllocationControl> control_;
    int id_ = 0;
};

template <typename T>
class PatternAllocator {
   public:
    using value_type = T;
    using is_always_equal = std::true_type;

    template <typename U>
    struct rebind {
        using other = PatternAllocator<U>;
    };

    [[nodiscard]] T* allocate(size_t count) {
        T* result = std::allocator<T>{}.allocate(count);
        std::memset(result, kPattern, count * sizeof(T));
        return result;
    }

    void deallocate(T* pointer, size_t count) noexcept {
        std::allocator<T>{}.deallocate(pointer, count);
    }

    template <typename U>
    [[nodiscard]] constexpr bool operator==(const PatternAllocator<U>&) const noexcept {
        return true;
    }

    template <typename U>
    [[nodiscard]] constexpr bool operator!=(const PatternAllocator<U>&) const noexcept {
        return false;
    }

    static constexpr unsigned char kPattern = 0xA5;
};

struct alignas(128) OverAlignedTrivial {
    uint64_t value;
};

static_assert(std::is_trivial_v<OverAlignedTrivial>);

TEST(Array, DefaultAndMovedFromArraysAreEmpty) {
    Array<int> empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0);
    EXPECT_EQ(empty.size_bytes(), 0);
    EXPECT_EQ(empty.data(), nullptr);
    EXPECT_TRUE(empty.dimensions().empty());
    EXPECT_EQ(empty.begin(), empty.end());

    Array<int> source(std::vector<size_t>{2, 3});
    source[0] = 17;
    Array<int> destination(std::move(source));

    EXPECT_EQ(destination.size(), 6);
    EXPECT_EQ(destination[0], 17);
    // The documented moved-from state is part of Array's contract.
    // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_TRUE(source.dimensions().empty());
    // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
}

TEST(Array, StoresDimensionsAndContiguousData) {
    Array<int> array(std::vector<size_t>{2, 3});
    EXPECT_EQ(array.size(), 6);
    EXPECT_EQ(array.size_bytes(), 6 * sizeof(int));
    EXPECT_EQ(array.dimensions(), (std::vector<size_t>{2, 3}));

    for (size_t index = 0; index < array.size(); ++index) {
        array[index] = static_cast<int>(index * 3);
    }

    const Array<int>& const_array = array;
    EXPECT_EQ(const_array[4], 12);
    EXPECT_EQ(const_array.at(5), 15);
    EXPECT_EQ(const_array.end() - const_array.begin(), 6);
}

TEST(Array, SupportsFixedSizeDimensionContainers) {
    Array<int, std::array<size_t, 2>> array(std::array<size_t, 2>{2, 3});
    EXPECT_EQ(array.size(), 6);
    EXPECT_EQ(array.dimensions(), (std::array<size_t, 2>{2, 3}));
}

TEST(Array, TreatsEmptyOrZeroDimensionsAsEmpty) {
    Array<int> no_dimensions(std::vector<size_t>{});
    Array<int> zero_dimension(std::vector<size_t>{4, 0, 8});

    EXPECT_TRUE(no_dimensions.empty());
    EXPECT_EQ(no_dimensions.data(), nullptr);
    EXPECT_TRUE(zero_dimension.empty());
    EXPECT_EQ(zero_dimension.data(), nullptr);
    EXPECT_EQ(zero_dimension.dimensions(), (std::vector<size_t>{4, 0, 8}));
}

TEST(Array, AtChecksBoundsWhileSubscriptRemainsAvailable) {
    Array<int> array(std::vector<size_t>{2});
    array[0] = 4;
    array[1] = 9;

    EXPECT_EQ(array.at(1), 9);
    EXPECT_THROW(static_cast<void>(array.at(2)), std::out_of_range);

    const Array<int>& const_array = array;
    EXPECT_THROW(static_cast<void>(const_array.at(2)), std::out_of_range);
}

TEST(Array, RejectsDimensionAndByteSizeOverflow) {
    const size_t maximum = std::numeric_limits<size_t>::max();
    EXPECT_THROW((Array<char>(std::vector<size_t>{maximum, 2})), std::length_error);
    EXPECT_THROW(
        (Array<uint64_t>(std::vector<size_t>{(maximum / sizeof(uint64_t)) + 1})),
        std::length_error
    );
}

TEST(Array, ResetReleasesOrReplacesStorage) {
    Array<int> array(std::vector<size_t>{3});
    array[0] = 11;

    array.reset(std::vector<size_t>{2, 2});
    EXPECT_EQ(array.size(), 4);
    EXPECT_EQ(array.dimensions(), (std::vector<size_t>{2, 2}));

    array.reset();
    EXPECT_TRUE(array.empty());
    EXPECT_EQ(array.data(), nullptr);
    EXPECT_TRUE(array.dimensions().empty());

    array.reset();
    EXPECT_TRUE(array.empty());
}

TEST(Array, FailedDimensionMoveLeavesResetTargetUnchanged) {
    ThrowingMoveDimensions initial_dimensions(2);
    ThrowingDimensionsArray array(initial_dimensions);
    array[0] = 11;
    array[1] = 29;

    ThrowingMoveDimensions replacement_dimensions(4);
    ThrowingMoveDimensions::set_throw_on_move(true);
    EXPECT_THROW(array.reset(replacement_dimensions), std::runtime_error);
    ThrowingMoveDimensions::set_throw_on_move(false);

    EXPECT_EQ(array.size(), 2);
    EXPECT_EQ(array.dimensions()[0], 2);
    EXPECT_EQ(array[0], 11);
    EXPECT_EQ(array[1], 29);
}

TEST(Array, FailedAllocationLeavesResetAndMoveAssignmentOperandsUnchanged) {
    using FailingAllocator = ControlledAllocator<int>;
    using ControlledArray = Array<int, std::vector<size_t>, FailingAllocator>;
    const auto target_control = std::make_shared<AllocationControl>();
    const auto source_control = std::make_shared<AllocationControl>();

    {
        ControlledArray target(std::vector<size_t>{2}, FailingAllocator(target_control, 1));
        target[0] = 17;
        target[1] = 23;

        target_control->fail_allocation = true;
        EXPECT_THROW(target.reset(std::vector<size_t>{4}), std::bad_alloc);
        EXPECT_EQ(target.size(), 2);
        EXPECT_EQ(target.dimensions(), (std::vector<size_t>{2}));
        EXPECT_EQ(target[0], 17);
        EXPECT_EQ(target[1], 23);

        ControlledArray source(std::vector<size_t>{3}, FailingAllocator(source_control, 2));
        source[0] = 31;
        source[1] = 37;
        source[2] = 41;

        EXPECT_THROW(target = std::move(source), std::bad_alloc);
        EXPECT_EQ(target.size(), 2);
        EXPECT_EQ(target.dimensions(), (std::vector<size_t>{2}));
        EXPECT_EQ(target[0], 17);
        EXPECT_EQ(target[1], 23);
        // The failed move assignment must leave the source usable and unchanged.
        // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        EXPECT_EQ(source.size(), 3);
        EXPECT_EQ(source.dimensions(), (std::vector<size_t>{3}));
        EXPECT_EQ(source[0], 31);
        EXPECT_EQ(source[2], 41);
        // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    }

    EXPECT_EQ(target_control->outstanding_allocations, 0);
    EXPECT_EQ(source_control->outstanding_allocations, 0);
}

TEST(Array, StartsTrivialObjectLifetimesWithoutInitializingTheirValues) {
    using PatternArray = Array<uint32_t, std::vector<size_t>, PatternAllocator<uint32_t>>;
    PatternArray array(std::vector<size_t>{4});

    const auto* bytes = reinterpret_cast<const unsigned char*>(array.data());
    for (size_t index = 0; index < array.size_bytes(); ++index) {
        EXPECT_EQ(bytes[index], PatternAllocator<uint32_t>::kPattern);
    }

    for (size_t index = 0; index < array.size(); ++index) {
        array[index] = static_cast<uint32_t>(100 + index);
    }
    EXPECT_EQ(array[0], 100U);
    EXPECT_EQ(array[3], 103U);
}

TEST(Array, MoveAssignmentHandlesSelfMoveAndUnequalAllocators) {
    using Allocator = StatefulAllocator<int>;
    using StatefulArray = Array<int, std::vector<size_t>, Allocator>;
    Allocator::clear_tracking();

    {
        StatefulArray target(std::vector<size_t>{2}, Allocator(1));
        StatefulArray source(std::vector<size_t>{3}, Allocator(2));
        source[0] = 7;
        source[1] = 8;
        source[2] = 9;

        target = std::move(source);
        EXPECT_EQ(target.size(), 3);
        EXPECT_EQ(target[0], 7);
        EXPECT_EQ(target[2], 9);
        // The documented moved-from state is part of Array's contract.
        // NOLINTNEXTLINE(bugprone-use-after-move)
        EXPECT_TRUE(source.empty());

        target = std::move(target);
        EXPECT_EQ(target.size(), 3);
        EXPECT_EQ(target[1], 8);

        StatefulArray moved(std::move(target));
        EXPECT_EQ(moved.size(), 3);
        EXPECT_EQ(moved[2], 9);
        // The documented moved-from state is part of Array's contract.
        // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        EXPECT_TRUE(target.empty());
    }

    EXPECT_EQ(Allocator::errors(), 0);
    EXPECT_EQ(Allocator::outstanding_allocations(), 0);
}

TEST(Array, EqualAllocatorMovesStealStorageWithoutAllocating) {
    using TestAllocator = StatefulAllocator<int>;
    using StatefulArray = Array<int, std::vector<size_t>, TestAllocator>;
    TestAllocator::clear_tracking();

    {
        StatefulArray target(std::vector<size_t>{2}, TestAllocator(7));
        StatefulArray source(std::vector<size_t>{3}, TestAllocator(7));
        source[0] = 43;
        source[2] = 47;
        int* source_data = source.data();
        ASSERT_EQ(TestAllocator::allocation_calls(), 2);

        target = std::move(source);
        EXPECT_EQ(TestAllocator::allocation_calls(), 2);
        EXPECT_EQ(target.data(), source_data);
        EXPECT_EQ(target[0], 43);
        EXPECT_EQ(target[2], 47);
        // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        EXPECT_TRUE(source.empty());

        StatefulArray destination(std::move(target));
        EXPECT_EQ(TestAllocator::allocation_calls(), 2);
        EXPECT_EQ(destination.data(), source_data);
        EXPECT_EQ(destination[2], 47);
    }

    EXPECT_EQ(TestAllocator::errors(), 0);
    EXPECT_EQ(TestAllocator::outstanding_allocations(), 0);
}

TEST(Array, MoveAssignmentPropagatesAllocatorOwnership) {
    using TestAllocator = PropagatingAllocator<int>;
    using StatefulArray = Array<int, std::vector<size_t>, TestAllocator>;
    TestAllocator::clear_tracking();

    {
        StatefulArray target(std::vector<size_t>{2}, TestAllocator(1));
        StatefulArray source(std::vector<size_t>{3}, TestAllocator(2));
        source[0] = 53;
        source[2] = 59;

        target = std::move(source);
        EXPECT_EQ(target.size(), 3);
        EXPECT_EQ(target[0], 53);
        EXPECT_EQ(target[2], 59);
        // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        EXPECT_TRUE(source.empty());
    }

    EXPECT_EQ(TestAllocator::errors(), 0);
    EXPECT_EQ(TestAllocator::outstanding_allocations(), 0);
}

TEST(AlignedAllocator, HonorsAlignmentAndHandlesBoundaryCases) {
    AlignedAllocator<int, 64> allocator;
    AlignedAllocator<char, 64> rebound_allocator;
    static_assert(noexcept(allocator.deallocate(nullptr, 0)));
    EXPECT_TRUE(allocator == rebound_allocator);
    EXPECT_FALSE(allocator != rebound_allocator);
    EXPECT_EQ(allocator.allocate(0), nullptr);

    int* data = allocator.allocate(5);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(data) % 64, 0);
    allocator.deallocate(data, 5);

    Array<int, std::vector<size_t>, AlignedAllocator<int, 128>> array(std::vector<size_t>{5}
    );
    EXPECT_EQ(reinterpret_cast<uintptr_t>(array.data()) % 128, 0);

    const size_t maximum = std::numeric_limits<size_t>::max();
    EXPECT_THROW(
        static_cast<void>(allocator.allocate(maximum / sizeof(int))), std::length_error
    );
}

TEST(Allocator, ProvidesRawStorageIncludingForOverAlignedTypes) {
    Allocator<OverAlignedTrivial> allocator;
    EXPECT_EQ(allocator.allocate(0), nullptr);

    OverAlignedTrivial* data = allocator.allocate(2);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(data) % alignof(OverAlignedTrivial), 0);
    std::uninitialized_default_construct_n(data, 2);
    data[0].value = 73;
    EXPECT_EQ(data[0].value, 73U);
    allocator.deallocate(data, 2);

    const size_t maximum = std::numeric_limits<size_t>::max();
    EXPECT_THROW(
        static_cast<void>(allocator.allocate(maximum / sizeof(OverAlignedTrivial) + 1)),
        std::bad_array_new_length
    );
}

class ArraySerializationTest : public ::testing::Test {
   protected:
    void TearDown() override {
        std::remove(kRoundTripPath);
        std::remove(kTruncatedPath);
    }

    static constexpr const char* kRoundTripPath = "array_roundtrip_test.bin";
    static constexpr const char* kTruncatedPath = "array_truncated_test.bin";
};

TEST_F(ArraySerializationTest, SavesOnlyRawPayloadAndLoadsItExactly) {
    Array<uint32_t> source(std::vector<size_t>{3});
    source[0] = 0x01020304U;
    source[1] = 0x11223344U;
    source[2] = 0xFFEEDDCCU;

    {
        std::ofstream output(kRoundTripPath, std::ios::binary);
        source.save(output);
    }

    std::ifstream raw_input(kRoundTripPath, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(raw_input.good());
    EXPECT_EQ(raw_input.tellg(), static_cast<std::streamoff>(source.size_bytes()));
    raw_input.seekg(0);
    std::vector<char> bytes(source.size_bytes());
    raw_input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(raw_input.good());
    EXPECT_EQ(std::memcmp(bytes.data(), source.data(), source.size_bytes()), 0);
    raw_input.close();

    Array<uint32_t> loaded(std::vector<size_t>{3});
    std::ifstream input(kRoundTripPath, std::ios::binary);
    loaded.load(input);
    EXPECT_EQ(loaded[0], source[0]);
    EXPECT_EQ(loaded[1], source[1]);
    EXPECT_EQ(loaded[2], source[2]);
}

TEST_F(ArraySerializationTest, RejectsInvalidAndTruncatedStreams) {
    Array<uint32_t> array(std::vector<size_t>{3});

    std::ofstream invalid_output;
    EXPECT_THROW(array.save(invalid_output), std::ios_base::failure);

    {
        const uint32_t value = 42;
        std::ofstream output(kTruncatedPath, std::ios::binary);
        output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    std::ifstream truncated_input(kTruncatedPath, std::ios::binary);
    EXPECT_THROW(array.load(truncated_input), std::ios_base::failure);

    std::ifstream invalid_input;
    EXPECT_THROW(array.load(invalid_input), std::ios_base::failure);
}

}  // namespace
