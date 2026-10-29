#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"

namespace rabitqlib::simd {
namespace {
// Compile the complete batch estimator at each ISA so accumulation conversion and
// factor correction use the same vector width as the scanning kernels.
template <auto Accumulate, auto AccumulateHacc>
inline void split_batch_estdist_impl(
    const char* batch_data,
    const SplitBatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances,
    float* lower_bounds,
    float* ip_x0_qr,
    bool use_hacc
) {
    ConstBatchDataMap<float> cur_batch(batch_data, padded_dim);
    std::array<int32_t, fastscan::kBatchSize> accu_values{};
    RowMajorArrayMap<int32_t> accu_arr(accu_values.data(), 1, fastscan::kBatchSize);
    if (use_hacc) {
        AccumulateHacc(
            cur_batch.bin_code(), query_wrapper.lut(), accu_values.data(), padded_dim
        );
    } else {
        Accumulate(
            cur_batch.bin_code(), query_wrapper.lut(), accu_values.data(), padded_dim
        );
    }

    std::array<float, fastscan::kBatchSize> f_add_values;
    std::array<float, fastscan::kBatchSize> f_rescale_values;
    std::array<float, fastscan::kBatchSize> f_error_values;
    cur_batch.f_add().copy_to(f_add_values.data(), f_add_values.size());
    cur_batch.f_rescale().copy_to(f_rescale_values.data(), f_rescale_values.size());
    cur_batch.f_error().copy_to(f_error_values.data(), f_error_values.size());
    ConstRowMajorArrayMap<float> f_add_arr(f_add_values.data(), 1, fastscan::kBatchSize);
    ConstRowMajorArrayMap<float> f_rescale_arr(
        f_rescale_values.data(), 1, fastscan::kBatchSize
    );
    ConstRowMajorArrayMap<float> f_error_arr(
        f_error_values.data(), 1, fastscan::kBatchSize
    );

    RowMajorArrayMap<float> est_dist_arr(estimated_distances, 1, fastscan::kBatchSize);
    RowMajorArrayMap<float> ip_x0_qr_arr(ip_x0_qr, 1, fastscan::kBatchSize);
    RowMajorArrayMap<float> low_dist_arr(lower_bounds, 1, fastscan::kBatchSize);

    ip_x0_qr_arr = query_wrapper.delta() * (accu_arr.template cast<float>()) +
                   query_wrapper.sum_vl_lut();

    est_dist_arr = f_add_arr + query_wrapper.g_add() +
                   f_rescale_arr * (ip_x0_qr_arr + query_wrapper.k1xsumq());

    low_dist_arr = est_dist_arr - f_error_arr * query_wrapper.g_error();
}

}  // namespace
}  // namespace rabitqlib::simd
