#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"

namespace rabitqlib::simd {
namespace {
// Compile the complete batch estimator at each ISA so accumulation conversion and
// factor correction use the same vector width as the scanning kernels.
inline void split_batch_estdist_impl(
    const char* batch_data,
    const SplitBatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    float* low_distance,
    float* ip_x0_qr,
    bool use_hacc
) {
    ConstBatchDataMap<float> cur_batch(batch_data, padded_dim);
    std::array<int32_t, fastscan::kBatchSize> accu_values{};
    RowMajorArrayMap<int32_t> accu_arr(accu_values.data(), 1, fastscan::kBatchSize);
    if (use_hacc) {
        fastscan::accumulate_hacc(
            cur_batch.bin_code(), q_obj.lut(), accu_values.data(), padded_dim
        );
    } else {
        fastscan::accumulate(
            cur_batch.bin_code(), q_obj.lut(), accu_values.data(), padded_dim
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

    RowMajorArrayMap<float> est_dist_arr(est_distance, 1, fastscan::kBatchSize);
    RowMajorArrayMap<float> ip_x0_qr_arr(ip_x0_qr, 1, fastscan::kBatchSize);
    RowMajorArrayMap<float> low_dist_arr(low_distance, 1, fastscan::kBatchSize);

    ip_x0_qr_arr = q_obj.delta() * (accu_arr.template cast<float>()) + q_obj.sum_vl_lut();

    est_dist_arr =
        f_add_arr + q_obj.g_add() + f_rescale_arr * (ip_x0_qr_arr + q_obj.k1xsumq());

    low_dist_arr = est_dist_arr - f_error_arr * q_obj.g_error();
}

inline void qg_batch_estdist_impl(
    const char* batch_data,
    const BatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance
) {
    using T = float;
    ConstQGBatchDataMap<T> cur_batch(batch_data, padded_dim);
    std::array<int32_t, fastscan::kBatchSize> accu_values{};
    fastscan::accumulate(cur_batch.bin_code(), q_obj.lut(), accu_values.data(), padded_dim);

    ConstRowMajorArrayMap<int32_t> ip_arr(accu_values.data(), 1, fastscan::kBatchSize);
    std::array<T, fastscan::kBatchSize> f_add_values;
    std::array<T, fastscan::kBatchSize> f_rescale_values;
    cur_batch.f_add().copy_to(f_add_values.data(), f_add_values.size());
    cur_batch.f_rescale().copy_to(f_rescale_values.data(), f_rescale_values.size());
    ConstRowMajorArrayMap<T> f_add_arr(f_add_values.data(), 1, fastscan::kBatchSize);
    ConstRowMajorArrayMap<T> f_rescale_arr(
        f_rescale_values.data(), 1, fastscan::kBatchSize
    );

    RowMajorArrayMap<T> est_dist_arr(est_distance, 1, fastscan::kBatchSize);

    est_dist_arr = f_add_arr + q_obj.g_add() +
                   (f_rescale_arr * (q_obj.delta() * (ip_arr.template cast<T>()) +
                                     q_obj.sum_vl_lut() + q_obj.k1xsumq()));
}

}  // namespace
}  // namespace rabitqlib::simd
