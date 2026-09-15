#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib {

template <typename T>
class Lut {
    static constexpr size_t kNumBits = 8;
    static constexpr size_t kNumBitsHacc = 16;
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type in Lut");

   private:
    [[nodiscard]] static size_t checked_table_length(size_t padded_dim) {
        if (padded_dim == 0 || padded_dim % 16 != 0) {
            throw std::invalid_argument(
                "FastScan dimension must be a positive multiple of 16"
            );
        }
        if (padded_dim > std::numeric_limits<size_t>::max() / 4) {
            throw std::length_error("FastScan lookup table is too large");
        }
        return padded_dim * 4;
    }

    [[nodiscard]] static size_t checked_storage_length(size_t padded_dim, bool use_hacc) {
        const size_t table_length = checked_table_length(padded_dim);
        const size_t copies = use_hacc ? 2 : 1;
        if (table_length > std::numeric_limits<size_t>::max() / copies) {
            throw std::length_error("FastScan lookup table is too large");
        }
        return table_length * copies;
    }

    size_t table_length_ = 0;
    std::vector<uint8_t> lut_;
    T delta_ = 0;
    T sum_vl_lut_ = 0;

   public:
    explicit Lut() = default;
    explicit Lut(const T* rotated_query, size_t padded_dim, bool use_hacc = false)
        : table_length_(checked_table_length(padded_dim))
        , lut_(checked_storage_length(padded_dim, use_hacc)) {
        // quantize float lut
        std::vector<float> lut_float(table_length_);
        fastscan::pack_lut(padded_dim, rotated_query, lut_float.data());
        T vl_lut;
        T vr_lut;
        data_range(lut_float.data(), table_length_, vl_lut, vr_lut);

        if (use_hacc) {
            delta_ = (vr_lut - vl_lut) / ((1 << kNumBitsHacc) - 1);

            // quantize float lut into uint16 then change to split table
            std::vector<uint16_t> lut_u16(table_length_);
            scalar_quantize(
                lut_u16.data(), lut_float.data(), table_length_, vl_lut, delta_
            );
            fastscan::transfer_lut_hacc(lut_u16.data(), padded_dim, lut_.data());
        } else {
            delta_ = (vr_lut - vl_lut) / ((1 << kNumBits) - 1);
            scalar_quantize(lut_.data(), lut_float.data(), table_length_, vl_lut, delta_);
        }

        size_t num_table = table_length_ / 16;
        sum_vl_lut_ = vl_lut * static_cast<float>(num_table);
    }
    [[nodiscard]] const uint8_t* lut() const { return lut_.data(); };
    [[nodiscard]] T delta() const { return delta_; };
    [[nodiscard]] T sum_vl() const { return sum_vl_lut_; };
};
}  // namespace rabitqlib
