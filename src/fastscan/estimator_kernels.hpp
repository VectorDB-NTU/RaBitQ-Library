#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"

namespace rabitqlib::simd {
namespace {
// Compile the complete batch estimator at each ISA so accumulation conversion and
// factor correction use the same vector width as the scanning kernels. The
// correction is written as plain loops so each backend vectorizes it with its own
// flags: an Eigen expression here would instantiate identically named weak helper
// functions in every ISA translation unit, and the linker keeps one copy for all.
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
    std::array<int32_t, fastscan::kBatchSize> accumulated_values{};
    if (use_hacc) {
        AccumulateHacc(
            cur_batch.bin_code(), query_wrapper.lut(), accumulated_values.data(), padded_dim
        );
    } else {
        Accumulate(
            cur_batch.bin_code(), query_wrapper.lut(), accumulated_values.data(), padded_dim
        );
    }

    std::array<float, fastscan::kBatchSize> f_add_values;
    std::array<float, fastscan::kBatchSize> f_rescale_values;
    std::array<float, fastscan::kBatchSize> f_error_values;
    cur_batch.f_add().copy_to(f_add_values.data(), f_add_values.size());
    cur_batch.f_rescale().copy_to(f_rescale_values.data(), f_rescale_values.size());
    cur_batch.f_error().copy_to(f_error_values.data(), f_error_values.size());

    const float delta = query_wrapper.delta();
    const float sum_vl_lut = query_wrapper.sum_vl_lut();
    const float g_add = query_wrapper.g_add();
    const float k1xsumq = query_wrapper.k1xsumq();
    const float g_error = query_wrapper.g_error();
    for (size_t lane = 0; lane < fastscan::kBatchSize; ++lane) {
        const float ip = delta * static_cast<float>(accumulated_values[lane]) + sum_vl_lut;
        const float estimated =
            f_add_values[lane] + g_add + f_rescale_values[lane] * (ip + k1xsumq);
        ip_x0_qr[lane] = ip;
        estimated_distances[lane] = estimated;
        lower_bounds[lane] = estimated - f_error_values[lane] * g_error;
    }
}

}  // namespace
}  // namespace rabitqlib::simd
