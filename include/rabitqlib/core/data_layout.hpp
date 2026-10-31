#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "rabitqlib/utils/packed.hpp"

namespace rabitqlib {
namespace detail {
// Byte offsets shared by writable and read-only single-vector views.
// The one-bit layout is the base-code layout with bits == 1.
template <typename T>
struct SingleCodeOffsets {
    static size_t add_offset(size_t padded_dim, size_t bits = 1) {
        return padded_dim * bits / 8;
    }
    static size_t rescale_offset(size_t padded_dim, size_t bits = 1) {
        return add_offset(padded_dim, bits) + sizeof(T);
    }
    static size_t error_offset(size_t padded_dim, size_t bits = 1) {
        return rescale_offset(padded_dim, bits) + sizeof(T);
    }
};
}  // namespace detail

template <typename T>
struct ExDataMap {
   public:
    explicit ExDataMap(char* data, size_t padded_dim, size_t ex_bits)
        : ex_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_ex_(data + detail::SingleCodeOffsets<T>::add_offset(padded_dim, ex_bits))
        , f_rescale_ex_(
              data + detail::SingleCodeOffsets<T>::rescale_offset(padded_dim, ex_bits)
          ) {}

    static size_t data_bytes(size_t padded_dim, size_t ex_bits) {
        if (ex_bits == 0) {
            return 0;
        }
        const size_t max_size = std::numeric_limits<size_t>::max();
        if (padded_dim > max_size / ex_bits) {
            throw std::length_error("ExDataMap storage size exceeds size_t");
        }
        const size_t code_bytes =
            detail::SingleCodeOffsets<T>::add_offset(padded_dim, ex_bits);
        const size_t factor_bytes = sizeof(T) * 2;
        if (code_bytes > max_size - factor_bytes) {
            throw std::length_error("ExDataMap storage size exceeds size_t");
        }
        return code_bytes + factor_bytes;
    }

    [[nodiscard]] uint8_t* ex_code() { return ex_code_; }
    [[nodiscard]] detail::PackedValueRef<T> f_add_ex() { return f_add_ex_; }
    [[nodiscard]] detail::PackedValueRef<T> f_rescale_ex() { return f_rescale_ex_; }

   private:
    uint8_t* ex_code_;
    detail::PackedValueRef<T> f_add_ex_;
    detail::PackedValueRef<T> f_rescale_ex_;
};

template <typename T>
struct ConstExDataMap {
   public:
    explicit ConstExDataMap(const char* data, size_t padded_dim, size_t ex_bits)
        : ex_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_ex_(data + detail::SingleCodeOffsets<T>::add_offset(padded_dim, ex_bits))
        , f_rescale_ex_(
              data + detail::SingleCodeOffsets<T>::rescale_offset(padded_dim, ex_bits)
          ) {}

    [[nodiscard]] const uint8_t* ex_code() const { return ex_code_; }
    [[nodiscard]] T f_add_ex() const { return detail::load_packed_value<T>(f_add_ex_); }
    [[nodiscard]] T f_rescale_ex() const {
        return detail::load_packed_value<T>(f_rescale_ex_);
    }

   private:
    const uint8_t* ex_code_;
    const char* f_add_ex_;
    const char* f_rescale_ex_;
};

template <typename T>
struct BaseDataMap {
   public:
    explicit BaseDataMap(char* data, size_t padded_dim, size_t base_bits)
        : base_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_(data + detail::SingleCodeOffsets<T>::add_offset(padded_dim, base_bits))
        , f_rescale_(
              data + detail::SingleCodeOffsets<T>::rescale_offset(padded_dim, base_bits)
          )
        , f_error_(
              data + detail::SingleCodeOffsets<T>::error_offset(padded_dim, base_bits)
          ) {}

    static size_t data_bytes(size_t padded_dim, size_t base_bits) {
        return detail::SingleCodeOffsets<T>::error_offset(padded_dim, base_bits) +
               sizeof(T);
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
        , f_add_(data + detail::SingleCodeOffsets<T>::add_offset(padded_dim, base_bits))
        , f_rescale_(
              data + detail::SingleCodeOffsets<T>::rescale_offset(padded_dim, base_bits)
          )
        , f_error_(
              data + detail::SingleCodeOffsets<T>::error_offset(padded_dim, base_bits)
          ) {}

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
        , f_add_(data + detail::SingleCodeOffsets<T>::add_offset(padded_dim))
        , f_rescale_(data + detail::SingleCodeOffsets<T>::rescale_offset(padded_dim))
        , f_error_(data + detail::SingleCodeOffsets<T>::error_offset(padded_dim)) {}

    [[nodiscard]] uint8_t* bin_code() { return bin_code_; }
    [[nodiscard]] detail::PackedValueRef<T> f_add() { return f_add_; }
    [[nodiscard]] detail::PackedValueRef<T> f_rescale() { return f_rescale_; }
    [[nodiscard]] detail::PackedValueRef<T> f_error() { return f_error_; }

    static size_t data_bytes(size_t padded_dim) {
        return detail::SingleCodeOffsets<T>::error_offset(padded_dim) + sizeof(T);
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
        , f_add_(data + detail::SingleCodeOffsets<T>::add_offset(padded_dim))
        , f_rescale_(data + detail::SingleCodeOffsets<T>::rescale_offset(padded_dim))
        , f_error_(data + detail::SingleCodeOffsets<T>::error_offset(padded_dim)) {}

    [[nodiscard]] const uint8_t* bin_code() const { return bin_code_; }
    [[nodiscard]] T f_add() const { return detail::load_packed_value<T>(f_add_); }
    [[nodiscard]] T f_rescale() const { return detail::load_packed_value<T>(f_rescale_); }
    [[nodiscard]] T f_error() const { return detail::load_packed_value<T>(f_error_); }

    static size_t data_bytes(size_t padded_dim) {
        return BinDataMap<T>::data_bytes(padded_dim);
    }

   private:
    const uint8_t* bin_code_;
    const char* f_add_;
    const char* f_rescale_;
    const char* f_error_;
};
}  // namespace rabitqlib
