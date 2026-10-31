#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"

namespace rabitqlib::fastscan::detail {
// Share QG accumulation and correction while callers select the accumulator.
// The correction is a plain loop so each ISA translation unit vectorizes it with
// its own flags without instantiating shared weak helper functions.
template <typename T, auto Accumulate>
static inline void qg_batch_estdist_impl(
    const char* batch_data,
    const BatchQuery<T>& query_wrapper,
    size_t padded_dim,
    T* estimated_distances
) {
    ConstQGBatchDataMap<T> cur_batch(batch_data, padded_dim);
    std::array<int32_t, fastscan::kBatchSize> accumulated_values{};
    Accumulate(
        cur_batch.bin_code(), query_wrapper.lut(), accumulated_values.data(), padded_dim
    );

    std::array<T, fastscan::kBatchSize> f_add_values;
    std::array<T, fastscan::kBatchSize> f_rescale_values;
    cur_batch.f_add().copy_to(f_add_values.data(), f_add_values.size());
    cur_batch.f_rescale().copy_to(f_rescale_values.data(), f_rescale_values.size());

    const T delta = query_wrapper.delta();
    const T sum_vl_lut = query_wrapper.sum_vl_lut();
    const T g_add = query_wrapper.g_add();
    const T g_k1xsumq = query_wrapper.g_k1xsumq();
    for (size_t lane = 0; lane < fastscan::kBatchSize; ++lane) {
        const T ip =
            delta * static_cast<T>(accumulated_values[lane]) + sum_vl_lut + g_k1xsumq;
        estimated_distances[lane] =
            f_add_values[lane] + g_add + f_rescale_values[lane] * ip;
    }
}
}  // namespace rabitqlib::fastscan::detail
