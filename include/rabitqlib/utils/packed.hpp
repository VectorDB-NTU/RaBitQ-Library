#pragma once

#include <cstddef>
#include <cstring>
#include <iterator>
#include <type_traits>

namespace rabitqlib {
namespace detail {

// Factor blocks are part of a packed byte layout and therefore need not be aligned for T.
// Access them through memcpy so the representation remains valid C++17 storage rather than
// manufacturing T objects inside a char buffer with reinterpret_cast.
template <typename T>
class PackedValueRef {
    static_assert(
        std::is_trivially_copyable_v<T>, "packed values must be trivially copyable"
    );

   public:
    explicit PackedValueRef(char* data) : data_(data) {}
    PackedValueRef(const PackedValueRef&) = default;

    PackedValueRef& operator=(T value) {
        std::memcpy(data_, &value, sizeof(value));
        return *this;
    }

    PackedValueRef& operator=(const PackedValueRef& other) {
        return *this = static_cast<T>(other);
    }

    operator T() const {
        T value;
        std::memcpy(&value, data_, sizeof(value));
        return value;
    }

   private:
    char* data_;
};

template <typename T>
class PackedArrayView {
    static_assert(
        std::is_trivially_copyable_v<T>, "packed values must be trivially copyable"
    );

   public:
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = void;
    using reference = PackedValueRef<T>;
    using iterator_category = std::forward_iterator_tag;

    explicit PackedArrayView(char* data) : data_(data) {}

    [[nodiscard]] reference operator*() const { return reference(data_); }
    [[nodiscard]] reference operator[](size_t index) const {
        return reference(data_ + (index * sizeof(T)));
    }

    PackedArrayView& operator++() {
        data_ += sizeof(T);
        return *this;
    }
    PackedArrayView operator++(int) {
        PackedArrayView previous = *this;
        ++*this;
        return previous;
    }
    PackedArrayView& operator--() {
        data_ -= sizeof(T);
        return *this;
    }
    PackedArrayView& operator+=(difference_type count) {
        data_ += count * static_cast<difference_type>(sizeof(T));
        return *this;
    }
    PackedArrayView& operator-=(difference_type count) { return *this += -count; }

    friend PackedArrayView operator+(PackedArrayView view, difference_type count) {
        view += count;
        return view;
    }
    friend PackedArrayView operator+(difference_type count, PackedArrayView view) {
        return view + count;
    }
    friend PackedArrayView operator-(PackedArrayView view, difference_type count) {
        view -= count;
        return view;
    }
    friend difference_type operator-(PackedArrayView lhs, PackedArrayView rhs) {
        return (lhs.data_ - rhs.data_) / static_cast<difference_type>(sizeof(T));
    }
    friend bool operator==(PackedArrayView lhs, PackedArrayView rhs) {
        return lhs.data_ == rhs.data_;
    }
    friend bool operator!=(PackedArrayView lhs, PackedArrayView rhs) {
        return !(lhs == rhs);
    }

    void copy_to(T* output, size_t count) const {
        std::memcpy(output, data_, count * sizeof(T));
    }

   private:
    char* data_;
};

template <typename T>
class ConstPackedArrayView {
    static_assert(
        std::is_trivially_copyable_v<T>, "packed values must be trivially copyable"
    );

   public:
    explicit ConstPackedArrayView(const char* data) : data_(data) {}

    [[nodiscard]] T operator[](size_t index) const {
        T value;
        std::memcpy(&value, data_ + (index * sizeof(T)), sizeof(value));
        return value;
    }

    void copy_to(T* output, size_t count) const {
        std::memcpy(output, data_, count * sizeof(T));
    }

   private:
    const char* data_;
};

template <typename T>
[[nodiscard]] inline T load_packed_value(const char* data) {
    static_assert(
        std::is_trivially_copyable_v<T>, "packed values must be trivially copyable"
    );
    T value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

}  // namespace detail
}  // namespace rabitqlib
