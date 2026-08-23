// The implementation is largely based on the implementation of SVS.
// https://github.com/intel/ScalableVectorSearch

/*
 * Copyright 2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "rabitqlib/utils/memory.hpp"

namespace rabitqlib {
namespace array_impl {
template <typename Dims, typename = void>
struct dimensions_allocator_swap_safe : std::true_type {};

template <typename Dims>
struct dimensions_allocator_swap_safe<Dims, std::void_t<typename Dims::allocator_type>>
    : std::bool_constant<
          std::allocator_traits<
              typename Dims::allocator_type>::propagate_on_container_swap::value ||
          std::allocator_traits<typename Dims::allocator_type>::is_always_equal::value> {};

template <typename Dims>
inline constexpr bool dimensions_allocator_swap_safe_v =
    dimensions_allocator_swap_safe<Dims>::value;

template <typename Dims>
[[nodiscard]] auto size(const Dims& dims) -> size_t {
    static_assert(std::is_same_v<typename Dims::value_type, size_t>);

    if (dims.begin() == dims.end()) {
        return 0;
    }

    size_t res = 1;
    for (const size_t dim : dims) {
        if (dim == 0) {
            return 0;
        }
        if (res > std::numeric_limits<size_t>::max() / dim) {
            throw std::length_error("Array dimensions overflow");
        }
        res *= dim;
    }
    return res;
}
}  // namespace array_impl

template <
    typename T,
    typename Dims = std::vector<size_t>,
    typename Alloc = memory::Allocator<T>>
class Array {
   private:
    static_assert(std::is_trivial_v<T>);  // only handle trivial types
    static_assert(
        std::is_nothrow_default_constructible_v<Dims>,
        "Array dimensions must be nothrow default constructible"
    );
    static_assert(
        std::is_nothrow_swappable_v<Dims>, "Array dimensions must be nothrow swappable"
    );
    static_assert(
        array_impl::dimensions_allocator_swap_safe_v<Dims>,
        "allocator-aware Array dimensions must either propagate their allocator on swap "
        "or use an always-equal allocator"
    );

   public:
    using allocator_type = Alloc;
    using atraits = std::allocator_traits<allocator_type>;
    using pointer = typename atraits::pointer;
    using const_pointer = typename atraits::const_pointer;

    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;

   private:
    static_assert(
        std::is_same_v<pointer, T*>, "Array requires an allocator with raw pointers"
    );
    static_assert(
        std::is_same_v<const_pointer, const T*>,
        "Array requires an allocator with raw pointers"
    );

    [[nodiscard]] static size_t checked_size(const Dims& dims) {
        const size_t num_elements = array_impl::size(dims);
        if (num_elements > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::length_error("Array byte size overflow");
        }
        return num_elements;
    }

    static constexpr bool kNothrowMoveAssignment =
        (atraits::propagate_on_container_move_assignment::value
             ? std::is_nothrow_copy_constructible_v<Alloc> &&
                   std::is_nothrow_move_assignable_v<Alloc>
             : atraits::is_always_equal::value) &&
        std::is_nothrow_move_constructible_v<Dims>;

    void allocate() {
        if (size_ != 0) {
            pointer allocated = atraits::allocate(allocator_, size_);
            try {
                // Default initialization starts each trivial object's lifetime without
                // initializing its value. Optimized builds elide this operation for trivial
                // T.
                static_cast<void>(std::uninitialized_default_construct_n(allocated, size_));
            } catch (...) {
                atraits::deallocate(allocator_, allocated, size_);
                throw;
            }
            pointer_ = allocated;
        }
    }

    void release() noexcept {
        if (pointer_ != nullptr) {
            atraits::deallocate(allocator_, pointer_, size_);
        }
        pointer_ = nullptr;
        size_ = 0;
    }

    void take_storage(Array& other) noexcept {
        using std::swap;
        swap(dims_, other.dims_);
        size_ = std::exchange(other.size_, 0);
        pointer_ = std::exchange(other.pointer_, nullptr);
    }

    void swap_storage(Array& other) noexcept {
        using std::swap;
        swap(dims_, other.dims_);
        swap(size_, other.size_);
        swap(pointer_, other.pointer_);
    }

   public:
    Array() = default;

    explicit Array(Dims dims, const Alloc& allocator)
        : dims_(std::move(dims)), allocator_(allocator), size_(checked_size(dims_)) {
        allocate();
    }

    explicit Array(Dims dims) : Array(std::move(dims), Alloc()) {}

    ~Array() noexcept { release(); }

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;

    /// @brief move constructor
    Array(Array&& other) noexcept(std::is_nothrow_copy_constructible_v<Alloc>)
        : allocator_{other.allocator_} {
        take_storage(other);
    }

    // Unequal non-propagating allocators require allocating replacement storage.
    // NOLINTNEXTLINE(performance-noexcept-move-constructor)
    Array& operator=(Array&& other) noexcept(kNothrowMoveAssignment) {
        if (this == &other) {
            return *this;
        }

        if constexpr (atraits::propagate_on_container_move_assignment::value) {
            static_assert(
                std::is_nothrow_move_assignable_v<Alloc>,
                "a propagating Array allocator must be nothrow move assignable"
            );
            Array replacement(std::move(other));
            reset();
            allocator_ = std::move(replacement.allocator_);
            take_storage(replacement);
        } else {
            if constexpr (!atraits::is_always_equal::value) {
                if (allocator_ != other.allocator_) {
                    Array replacement(other.dims_, allocator_);
                    if (!other.empty()) {
                        std::memcpy(replacement.data(), other.data(), other.size_bytes());
                    }
                    swap_storage(replacement);
                    other.reset();
                    return *this;
                }
            }
            reset();
            take_storage(other);
        }
        return *this;
    }

    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] size_t size_bytes() const noexcept { return sizeof(T) * size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] const Dims& dimensions() const noexcept { return dims_; }

    [[nodiscard]] pointer data() { return pointer_; }
    [[nodiscard]] const_pointer data() const { return pointer_; }

    [[nodiscard]] iterator begin() noexcept { return pointer_; }
    [[nodiscard]] const_iterator begin() const noexcept { return pointer_; }
    [[nodiscard]] const_iterator cbegin() const noexcept { return pointer_; }
    [[nodiscard]] iterator end() noexcept {
        return pointer_ == nullptr ? nullptr : pointer_ + size_;
    }
    [[nodiscard]] const_iterator end() const noexcept {
        return pointer_ == nullptr ? nullptr : pointer_ + size_;
    }
    [[nodiscard]] const_iterator cend() const noexcept {
        return pointer_ == nullptr ? nullptr : pointer_ + size_;
    }

    [[nodiscard]] reference at(size_t idx) {
        if (idx >= size_) {
            throw std::out_of_range("Array index out of range");
        }
        return pointer_[idx];
    }
    [[nodiscard]] const_reference at(size_t idx) const {
        if (idx >= size_) {
            throw std::out_of_range("Array index out of range");
        }
        return pointer_[idx];
    }

    void reset() {
        Dims empty_dimensions{};
        release();
        using std::swap;
        swap(dims_, empty_dimensions);
    }

    void reset(Dims dims) {
        Array replacement(std::move(dims), allocator_);
        swap_storage(replacement);
    }

    void save(std::ofstream& output) const {
        if (!output.good()) {
            throw std::ios_base::failure("cannot write Array to an invalid stream");
        }
        if (size_bytes() >
            static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::length_error("Array is too large to serialize");
        }
        if (!empty()) {
            output.write(
                reinterpret_cast<const char*>(pointer_),
                static_cast<std::streamsize>(size_bytes())
            );
        }
        if (!output.good()) {
            throw std::ios_base::failure("failed to write Array data");
        }
    }

    void load(std::ifstream& input) {
        if (!input.good()) {
            throw std::ios_base::failure("cannot read Array from an invalid stream");
        }
        if (size_bytes() >
            static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::length_error("Array is too large to deserialize");
        }
        if (!empty()) {
            input.read(
                reinterpret_cast<char*>(pointer_),
                static_cast<std::streamsize>(size_bytes())
            );
        }
        if (!input.good()) {
            throw std::ios_base::failure("failed to read complete Array data");
        }
    }

    reference operator[](size_t idx) { return pointer_[idx]; }
    const_reference operator[](size_t idx) const { return pointer_[idx]; }

   private:
    [[no_unique_address]] Dims dims_{};
    [[no_unique_address]] Alloc allocator_;
    size_t size_ = 0;
    pointer pointer_ = nullptr;
};
}  // namespace rabitqlib
