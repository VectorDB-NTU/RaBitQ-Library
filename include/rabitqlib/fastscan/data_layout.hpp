#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/utils/packed.hpp"

namespace rabitqlib {
namespace detail {
// Standard and QG batches share the code/add/rescale prefix; only standard
// batches store the trailing error factors. Counts include all physical lanes.
template <typename T>
struct BatchCodeOffsets {
    static size_t add(size_t padded_dim) { return padded_dim * fastscan::kBatchSize / 8; }
    static size_t rescale(size_t padded_dim) {
        return add(padded_dim) + sizeof(T) * fastscan::kBatchSize;
    }
    static size_t error(size_t padded_dim) {
        return rescale(padded_dim) + sizeof(T) * fastscan::kBatchSize;
    }
};
}  // namespace detail

template <typename T>
struct BatchDataMap {
   public:
    explicit BatchDataMap(char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<uint8_t*>(data))  // 1-bit codes
        , f_add_(data + detail::BatchCodeOffsets<T>::add(padded_dim))
        , f_rescale_(data + detail::BatchCodeOffsets<T>::rescale(padded_dim))
        , f_error_(data + detail::BatchCodeOffsets<T>::error(padded_dim)) {}

    [[nodiscard]] uint8_t* bin_code() { return batch_bin_code_; }
    [[nodiscard]] detail::PackedArrayView<T> f_add() { return f_add_; }
    [[nodiscard]] detail::PackedArrayView<T> f_rescale() { return f_rescale_; }
    [[nodiscard]] detail::PackedArrayView<T> f_error() { return f_error_; }

    static size_t data_bytes(size_t padded_dim) {
        return detail::BatchCodeOffsets<T>::error(padded_dim) +
               sizeof(T) * fastscan::kBatchSize;
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
        : batch_bin_code_(reinterpret_cast<const uint8_t*>(data))  // 1-bit codes
        , f_add_(data + detail::BatchCodeOffsets<T>::add(padded_dim))
        , f_rescale_(data + detail::BatchCodeOffsets<T>::rescale(padded_dim))
        , f_error_(data + detail::BatchCodeOffsets<T>::error(padded_dim)) {}

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

// Two-factor batch layout used by FastScan consumers, including SymphonyQG.
template <typename T>
struct QGBatchDataMap {
   public:
    explicit QGBatchDataMap(char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<uint8_t*>(data))  // 1-bit codes
        , f_add_(data + detail::BatchCodeOffsets<T>::add(padded_dim))
        , f_rescale_(data + detail::BatchCodeOffsets<T>::rescale(padded_dim)) {}

    [[nodiscard]] uint8_t* bin_code() { return batch_bin_code_; }
    [[nodiscard]] detail::PackedArrayView<T> f_add() { return f_add_; }
    [[nodiscard]] detail::PackedArrayView<T> f_rescale() { return f_rescale_; }

    static size_t data_bytes(size_t padded_dim) {
        const size_t max_size = std::numeric_limits<size_t>::max();
        if (padded_dim > max_size / fastscan::kBatchSize) {
            throw std::length_error("QGBatchDataMap storage size exceeds size_t");
        }
        const size_t code_bytes = detail::BatchCodeOffsets<T>::add(padded_dim);
        const size_t factor_bytes = sizeof(T) * fastscan::kBatchSize * 2;
        if (code_bytes > max_size - factor_bytes) {
            throw std::length_error("QGBatchDataMap storage size exceeds size_t");
        }
        return code_bytes + factor_bytes;
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
        : batch_bin_code_(reinterpret_cast<const uint8_t*>(data))  // 1-bit codes
        , f_add_(data + detail::BatchCodeOffsets<T>::add(padded_dim))
        , f_rescale_(data + detail::BatchCodeOffsets<T>::rescale(padded_dim)) {}

    [[nodiscard]] const uint8_t* bin_code() const { return batch_bin_code_; }
    [[nodiscard]] detail::ConstPackedArrayView<T> f_add() const { return f_add_; }
    [[nodiscard]] detail::ConstPackedArrayView<T> f_rescale() const { return f_rescale_; }

    static size_t data_bytes(size_t padded_dim) {
        return QGBatchDataMap<T>::data_bytes(padded_dim);
    }

   private:
    const uint8_t* batch_bin_code_;
    detail::ConstPackedArrayView<T> f_add_;
    detail::ConstPackedArrayView<T> f_rescale_;
};

}  // namespace rabitqlib
