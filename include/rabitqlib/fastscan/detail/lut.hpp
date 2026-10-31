#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "rabitqlib/fastscan/detail/lut_prepare.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#include "rabitqlib/utils/input_checks.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib {

template <typename T>
class Lut {
    static constexpr size_t kNumBits = 8;
    static constexpr size_t kNumBitsHacc = 16;
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type in Lut");

   private:
    struct Shape {
        size_t table_length;
        size_t storage_length;
    };

    [[nodiscard]] static Shape checked_shape(size_t padded_dim, bool use_hacc) {
        if (padded_dim == 0 || padded_dim % 16 != 0) {
            throw std::invalid_argument(
                "FastScan dimension must be a positive multiple of 16"
            );
        }
        if (padded_dim > std::numeric_limits<size_t>::max() / 4) {
            throw std::length_error("FastScan lookup table is too large");
        }
        const size_t table_length = padded_dim * 4;
        const size_t copies = use_hacc ? 2 : 1;
        if (table_length > std::numeric_limits<size_t>::max() / copies) {
            throw std::length_error("FastScan lookup table is too large");
        }
        return {table_length, table_length * copies};
    }

    size_t table_length_ = 0;
    std::vector<uint8_t> lut_;
    T delta_ = 0;
    T sum_vl_lut_ = 0;

    // Both entry points validate the shape once. Only this method manages owned
    // storage; numerical preparation below receives valid shape and scratch.
    void reset_with_shape(
        const T* rotated_query,
        size_t padded_dim,
        T* lut_scratch,
        uint16_t* lut_u16_scratch,
        bool use_hacc,
        Shape shape
    ) {
        if (rotated_query == nullptr || lut_scratch == nullptr) {
            throw std::invalid_argument("Lut::reset requires query and LUT scratch");
        }
        if (use_hacc && lut_u16_scratch == nullptr) {
            throw std::invalid_argument(
                "Lut::reset requires uint16 scratch for high-accuracy lookup tables"
            );
        }
        // Table quantization converts query-derived values to integers.
        if (!all_finite(rotated_query, padded_dim)) {
            throw std::invalid_argument("Lut::reset requires a finite query");
        }
        lut_.resize(shape.storage_length);
        table_length_ = shape.table_length;
        build_lut(rotated_query, padded_dim, lut_scratch, lut_u16_scratch, use_hacc);
    }

    template <typename Code>
    void quantize_entries(Code* output, const T* values, T lo) const {
        if constexpr (std::is_same_v<T, float>) {
            scalar_quantize(output, values, table_length_, lo, delta_);
        } else {
            // Match the portable scalar quantizer without narrowing the query to float.
            const T reciprocal = delta_ == 0 ? T{0} : T{1} / delta_;
            for (size_t i = 0; i < table_length_; ++i) {
                const T rounded = std::round((values[i] - lo) * reciprocal);
                output[i] = static_cast<Code>(std::clamp(
                    rounded, T{0}, static_cast<T>(std::numeric_limits<Code>::max())
                ));
            }
        }
    }

    void build_lut(
        const T* rotated_query,
        size_t padded_dim,
        T* lut_scratch,
        uint16_t* lut_u16_scratch,
        bool use_hacc
    ) {
        T vl_lut;
        T vr_lut;
        bool finite_lut = false;
        if constexpr (std::is_same_v<T, float>) {
            finite_lut = fastscan::detail::pack_lut_with_range(
                padded_dim, rotated_query, lut_scratch, vl_lut, vr_lut
            );
        } else {
            fastscan::pack_lut(padded_dim, rotated_query, lut_scratch);
            data_range(lut_scratch, table_length_, vl_lut, vr_lut);
        }

        if (use_hacc) {
            delta_ = (vr_lut - vl_lut) / ((1 << kNumBitsHacc) - 1);
            quantize_entries(lut_u16_scratch, lut_scratch, vl_lut);
            fastscan::transfer_lut_hacc(lut_u16_scratch, padded_dim, lut_.data());
        } else {
            delta_ = (vr_lut - vl_lut) / ((1 << kNumBits) - 1);
            // finite_lut is only set for float, the only type the uint8 kernels take.
            if constexpr (std::is_same_v<T, float>) {
                if (finite_lut) {
                    fastscan::detail::quantize_lut_uint8(
                        lut_.data(), lut_scratch, table_length_, vl_lut, delta_
                    );
                }
            }
            if (!finite_lut) {
                quantize_entries(lut_.data(), lut_scratch, vl_lut);
            }
        }

        const size_t num_tables = table_length_ / 16;
        sum_vl_lut_ = vl_lut * static_cast<T>(num_tables);
    }

    // Construct owned storage at its known size. Growing an initially empty
    // vector through reset adds work to every fresh query's preparation.
    Lut(const T* rotated_query, size_t padded_dim, bool use_hacc, Shape shape)
        : table_length_(shape.table_length), lut_(shape.storage_length) {
        std::vector<T> lut_values(shape.table_length);
        std::vector<uint16_t> lut_u16(use_hacc ? shape.table_length : 0);
        reset_with_shape(
            rotated_query, padded_dim, lut_values.data(), lut_u16.data(), use_hacc, shape
        );
    }

   public:
    explicit Lut() = default;
    explicit Lut(const T* rotated_query, size_t padded_dim, bool use_hacc = false)
        : Lut(rotated_query, padded_dim, use_hacc, checked_shape(padded_dim, use_hacc)) {}

    Lut(const Lut&) = default;
    Lut(Lut&&) noexcept = default;
    Lut& operator=(Lut&&) noexcept = default;

    Lut& operator=(const Lut& other) {
        // Copy bytes before metadata: allocation failure must preserve the old shape.
        lut_ = other.lut_;
        table_length_ = other.table_length_;
        delta_ = other.delta_;
        sum_vl_lut_ = other.sum_vl_lut_;
        return *this;
    }

    /**
     * Reinitialize owned LUT bytes while reusing their allocation. Scratch buffers
     * are borrowed only for this call and may be overwritten/released afterwards.
     * lut_scratch must contain padded_dim * 4 values of T; high-accuracy mode
     * additionally requires padded_dim * 4 uint16_t values in lut_u16_scratch.
     */
    void reset(
        const T* rotated_query,
        size_t padded_dim,
        T* lut_scratch,
        uint16_t* lut_u16_scratch = nullptr,
        bool use_hacc = false
    ) {
        const Shape shape = checked_shape(padded_dim, use_hacc);
        reset_with_shape(
            rotated_query, padded_dim, lut_scratch, lut_u16_scratch, use_hacc, shape
        );
    }

    // Empty (including moved-from) storage is not a prepared query.
    [[nodiscard]] size_t padded_dim() const { return lut_.empty() ? 0 : table_length_ / 4; }
    [[nodiscard]] bool use_hacc() const { return lut_.size() > table_length_; }

    [[nodiscard]] const uint8_t* lut() const { return lut_.data(); }
    [[nodiscard]] T delta() const { return delta_; }
    [[nodiscard]] T sum_vl() const { return sum_vl_lut_; }
};
}  // namespace rabitqlib
