#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rabitqlib/core/detail/rabitq_impl.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"

namespace rabitqlib::quant::rabitq_impl::one_bit {
// Encoding requires a positive padded_dim divisible by 8; index pipelines use multiples
// of 32. packed_code needs ceil(num / 32) * 32 * (padded_dim / 8) bytes, including tail
// padding.
template <
    typename T,
    bool Parallel = false,
    typename FAdd,
    typename FRescale,
    typename FError>
inline void one_bit_batch_code(
    const T* data,
    const T* centroid,
    size_t num,
    size_t padded_dim,
    uint8_t* packed_code,
    FAdd f_add,
    FRescale f_recale,
    FError f_error,
    MetricType metric_type = METRIC_L2
) {
    std::vector<uint8_t> compact_codes(num * padded_dim / 8);

    one_bit_compact_codes<T, uint8_t, Parallel>(
        data,
        centroid,
        num,
        padded_dim,
        compact_codes.data(),
        f_add,
        f_recale,
        f_error,
        metric_type
    );
    fastscan::pack_codes(padded_dim, compact_codes.data(), num, packed_code);
}

}  // namespace rabitqlib::quant::rabitq_impl::one_bit
