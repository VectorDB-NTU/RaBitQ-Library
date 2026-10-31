#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"

namespace rabitqlib::fastscan::detail {
// Share QG accumulation and correction while callers select the accumulator.
// TU-local linkage keeps each ISA's Eigen expression compiled with its own flags.
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

    ConstRowMajorArrayMap<int32_t> ip_arr(
        accumulated_values.data(), 1, fastscan::kBatchSize
    );
    std::array<T, fastscan::kBatchSize> f_add_values;
    std::array<T, fastscan::kBatchSize> f_rescale_values;
    cur_batch.f_add().copy_to(f_add_values.data(), f_add_values.size());
    cur_batch.f_rescale().copy_to(f_rescale_values.data(), f_rescale_values.size());
    ConstRowMajorArrayMap<T> f_add_arr(f_add_values.data(), 1, fastscan::kBatchSize);
    ConstRowMajorArrayMap<T> f_rescale_arr(
        f_rescale_values.data(), 1, fastscan::kBatchSize
    );

    RowMajorArrayMap<T> estimated_distances_arr(
        estimated_distances, 1, fastscan::kBatchSize
    );

    estimated_distances_arr =
        f_add_arr + query_wrapper.g_add() +
        (f_rescale_arr * (query_wrapper.delta() * (ip_arr.template cast<T>()) +
                          query_wrapper.sum_vl_lut() + query_wrapper.k1xsumq()));
}
}  // namespace rabitqlib::fastscan::detail
