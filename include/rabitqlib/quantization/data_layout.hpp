#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <type_traits>

#include "rabitqlib/fastscan/fastscan.hpp"

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

template <typename T>
struct BatchDataMap {
   public:
    explicit BatchDataMap(char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_(data + (padded_dim * fastscan::kBatchSize / 8))  // 1 bit code
        , f_rescale_(
              data + (padded_dim * fastscan::kBatchSize / 8) +
              (sizeof(T) * fastscan::kBatchSize)
          )
        , f_error_(
              data + (padded_dim * fastscan::kBatchSize / 8) +
              (sizeof(T) * fastscan::kBatchSize * 2)
          ) {}

    [[nodiscard]] uint8_t* bin_code() { return batch_bin_code_; }
    [[nodiscard]] detail::PackedArrayView<T> f_add() { return f_add_; }
    [[nodiscard]] detail::PackedArrayView<T> f_rescale() { return f_rescale_; }
    [[nodiscard]] detail::PackedArrayView<T> f_error() { return f_error_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim * fastscan::kBatchSize / 8) +
               (sizeof(T) * fastscan::kBatchSize * 3);
    }

   private:
    uint8_t* batch_bin_code_;
    detail::PackedArrayView<T> f_add_;
    detail::PackedArrayView<T> f_rescale_;
    detail::PackedArrayView<T> f_error_;
};

template <typename T>
struct ConstBatchDataMap {
   public:
    explicit ConstBatchDataMap(const char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_(data + (padded_dim * fastscan::kBatchSize / 8))  // 1 bit code
        , f_rescale_(
              data + (padded_dim * fastscan::kBatchSize / 8) +
              (sizeof(T) * fastscan::kBatchSize)
          )
        , f_error_(
              data + (padded_dim * fastscan::kBatchSize / 8) +
              (sizeof(T) * fastscan::kBatchSize * 2)
          ) {}

    [[nodiscard]] const uint8_t* bin_code() const { return batch_bin_code_; }
    [[nodiscard]] detail::ConstPackedArrayView<T> f_add() const { return f_add_; }
    [[nodiscard]] detail::ConstPackedArrayView<T> f_rescale() const { return f_rescale_; }
    [[nodiscard]] detail::ConstPackedArrayView<T> f_error() const { return f_error_; }

   private:
    const uint8_t* batch_bin_code_;
    detail::ConstPackedArrayView<T> f_add_;
    detail::ConstPackedArrayView<T> f_rescale_;
    detail::ConstPackedArrayView<T> f_error_;
};

template <typename T>
struct QGBatchDataMap {
   public:
    explicit QGBatchDataMap(char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_(data + (padded_dim * fastscan::kBatchSize / 8))  // 1 bit code
        , f_rescale_(
              data + (padded_dim * fastscan::kBatchSize / 8) +
              (sizeof(T) * fastscan::kBatchSize)
          ) {}

    [[nodiscard]] uint8_t* bin_code() { return batch_bin_code_; }
    [[nodiscard]] detail::PackedArrayView<T> f_add() { return f_add_; }
    [[nodiscard]] detail::PackedArrayView<T> f_rescale() { return f_rescale_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim * fastscan::kBatchSize / 8) +
               (sizeof(T) * fastscan::kBatchSize * 2);
    }

   private:
    uint8_t* batch_bin_code_;
    detail::PackedArrayView<T> f_add_;
    detail::PackedArrayView<T> f_rescale_;
};

template <typename T>
struct ConstQGBatchDataMap {
   public:
    explicit ConstQGBatchDataMap(const char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_(data + (padded_dim * fastscan::kBatchSize / 8))  // 1 bit code
        , f_rescale_(
              data + (padded_dim * fastscan::kBatchSize / 8) +
              (sizeof(T) * fastscan::kBatchSize)
          ) {}

    [[nodiscard]] const uint8_t* bin_code() const { return batch_bin_code_; }
    [[nodiscard]] detail::ConstPackedArrayView<T> f_add() const { return f_add_; }
    [[nodiscard]] detail::ConstPackedArrayView<T> f_rescale() const { return f_rescale_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim * fastscan::kBatchSize / 8) +
               (sizeof(T) * fastscan::kBatchSize * 2);
    }

   private:
    const uint8_t* batch_bin_code_;
    detail::ConstPackedArrayView<T> f_add_;
    detail::ConstPackedArrayView<T> f_rescale_;
};

template <typename T>
struct ExDataMap {
   public:
    explicit ExDataMap(char* data, size_t padded_dim, size_t ex_bits)
        : ex_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_ex_(data + (padded_dim * ex_bits / 8))
        , f_recale_ex_(data + (padded_dim * ex_bits / 8) + sizeof(T)) {}

    static size_t data_bytes(size_t padded_dim, size_t ex_bits) {
        return ex_bits > 0 ? (padded_dim * ex_bits / 8) + (sizeof(T) * 2) : 0;
    }

    [[nodiscard]] uint8_t* ex_code() { return ex_code_; }
    [[nodiscard]] detail::PackedValueRef<T> f_add_ex() { return f_add_ex_; }
    [[nodiscard]] detail::PackedValueRef<T> f_rescale_ex() { return f_recale_ex_; }

   private:
    uint8_t* ex_code_;
    detail::PackedValueRef<T> f_add_ex_;
    detail::PackedValueRef<T> f_recale_ex_;
};

template <typename T>
struct ConstExDataMap {
   public:
    explicit ConstExDataMap(const char* data, size_t padded_dim, size_t ex_bits)
        : ex_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_ex_(data + (padded_dim * ex_bits / 8))
        , f_recale_ex_(data + (padded_dim * ex_bits / 8) + sizeof(T)) {}

    [[nodiscard]] const uint8_t* ex_code() const { return ex_code_; }
    [[nodiscard]] T f_add_ex() const { return detail::load_packed_value<T>(f_add_ex_); }
    [[nodiscard]] T f_rescale_ex() const {
        return detail::load_packed_value<T>(f_recale_ex_);
    }

   private:
    const uint8_t* ex_code_;
    const char* f_add_ex_;
    const char* f_recale_ex_;
};

template <typename T>
struct BaseDataMap {
   public:
    explicit BaseDataMap(char* data, size_t padded_dim, size_t base_bits)
        : base_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_(data + (padded_dim * base_bits / 8))
        , f_rescale_(data + (padded_dim * base_bits / 8) + sizeof(T))
        , f_error_(data + (padded_dim * base_bits / 8) + (sizeof(T) * 2)) {}

    static size_t data_bytes(size_t padded_dim, size_t base_bits) {
        return (padded_dim * base_bits / 8) + (sizeof(T) * 3);
    }

    [[nodiscard]] uint8_t* base_code() { return base_code_; }
    [[nodiscard]] detail::PackedValueRef<T> f_add() { return f_add_; }
    [[nodiscard]] detail::PackedValueRef<T> f_rescale() { return f_rescale_; }
    [[nodiscard]] detail::PackedValueRef<T> f_error() { return f_error_; }

   private:
    uint8_t* base_code_;
    detail::PackedValueRef<T> f_add_;
    detail::PackedValueRef<T> f_rescale_;
    detail::PackedValueRef<T> f_error_;
};

template <typename T>
struct ConstBaseDataMap {
   public:
    explicit ConstBaseDataMap(const char* data, size_t padded_dim, size_t base_bits)
        : base_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_(data + (padded_dim * base_bits / 8))
        , f_rescale_(data + (padded_dim * base_bits / 8) + sizeof(T))
        , f_error_(data + (padded_dim * base_bits / 8) + (sizeof(T) * 2)) {}

    [[nodiscard]] const uint8_t* base_code() const { return base_code_; }
    [[nodiscard]] T f_add() const { return detail::load_packed_value<T>(f_add_); }
    [[nodiscard]] T f_rescale() const { return detail::load_packed_value<T>(f_rescale_); }
    [[nodiscard]] T f_error() const { return detail::load_packed_value<T>(f_error_); }

   private:
    const uint8_t* base_code_;
    const char* f_add_;
    const char* f_rescale_;
    const char* f_error_;
};

template <typename T>
struct BinDataMap {
   public:
    explicit BinDataMap(char* data, size_t padded_dim)
        : bin_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_(data + (padded_dim / 8))
        , f_rescale_(data + (padded_dim / 8) + sizeof(T))
        , f_error_(data + (padded_dim / 8) + (sizeof(T) * 2)) {}

    [[nodiscard]] uint8_t* bin_code() { return bin_code_; }
    [[nodiscard]] detail::PackedValueRef<T> f_add() { return f_add_; }
    [[nodiscard]] detail::PackedValueRef<T> f_rescale() { return f_rescale_; }
    [[nodiscard]] detail::PackedValueRef<T> f_error() { return f_error_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim / 8) + (sizeof(T) * 3);
    }

   private:
    uint8_t* bin_code_;
    detail::PackedValueRef<T> f_add_;
    detail::PackedValueRef<T> f_rescale_;
    detail::PackedValueRef<T> f_error_;
};

template <typename T>
struct ConstBinDataMap {
   public:
    explicit ConstBinDataMap(const char* data, size_t padded_dim)
        : bin_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_(data + (padded_dim / 8))
        , f_rescale_(data + (padded_dim / 8) + sizeof(T))
        , f_error_(data + (padded_dim / 8) + (sizeof(T) * 2)) {}

    [[nodiscard]] const uint8_t* bin_code() const { return bin_code_; }
    [[nodiscard]] T f_add() const { return detail::load_packed_value<T>(f_add_); }
    [[nodiscard]] T f_rescale() const { return detail::load_packed_value<T>(f_rescale_); }
    [[nodiscard]] T f_error() const { return detail::load_packed_value<T>(f_error_); }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim / 8) + (sizeof(T) * 3);
    }

   private:
    const uint8_t* bin_code_;
    const char* f_add_;
    const char* f_rescale_;
    const char* f_error_;
};
}  // namespace rabitqlib
