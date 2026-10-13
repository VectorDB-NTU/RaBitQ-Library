#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "rabitqlib/core/data_layout.hpp"
#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/detail/encoder.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"

namespace rabitqlib::quant {
// Single-batch entry points accept 0..32 rows. Zero rows leave output unchanged.
// Allocate full BatchDataMap/QGBatchDataMap storage even for a partial batch;
// only the first num factors/results are meaningful. Inputs are borrowed for the call.
// Parallel=true enables OpenMP when built with it, otherwise encoding remains serial.
template <typename T, bool Parallel = false>
inline void quantize_one_batch(
    const T* data,
    const T* centroid,
    size_t num,
    size_t padded_dim,
    char* batch_data,
    MetricType metric_type = METRIC_L2
) {
    if (num > fastscan::kBatchSize) {
        throw std::invalid_argument(
            "FastScan single-batch encoding accepts at most 32 vectors"
        );
    }
    if (num == 0) {
        return;
    }
    BatchDataMap<T> this_batch(batch_data, padded_dim);

    rabitq_impl::one_bit::one_bit_batch_code<T, Parallel>(
        data,
        centroid,
        num,
        padded_dim,
        this_batch.bin_code(),
        this_batch.f_add(),
        this_batch.f_rescale(),
        this_batch.f_error(),
        metric_type
    );
}

template <typename T, bool Parallel = false>
inline void quantize_one_batch(
    const T* data,
    size_t num,
    size_t padded_dim,
    char* batch_data,
    MetricType metric_type = METRIC_L2
) {
    std::vector<T> centroid(padded_dim, 0);
    quantize_one_batch<T, Parallel>(
        data, centroid.data(), num, padded_dim, batch_data, metric_type
    );
}

template <typename T>
static inline void quantize_qg_batch(
    const T* data,
    const T* centroid,
    size_t num,
    size_t padded_dim,
    char* batch_data,
    MetricType metric_type = METRIC_L2
) {
    if (num > fastscan::kBatchSize) {
        throw std::invalid_argument(
            "FastScan single-batch encoding accepts at most 32 vectors"
        );
    }
    if (num == 0) {
        return;
    }
    std::array<T, fastscan::kBatchSize> f_error{};  // we dont need this factor for qg
    QGBatchDataMap<T> cur_batch(batch_data, padded_dim);

    rabitq_impl::one_bit::one_bit_batch_code<T>(
        data,
        centroid,
        num,
        padded_dim,
        cur_batch.bin_code(),
        cur_batch.f_add(),
        cur_batch.f_rescale(),
        f_error.data(),
        metric_type
    );
}

template <typename T>
static inline void quantize_qg_batch(
    const T* data,
    size_t num,
    size_t padded_dim,
    char* batch_data,
    MetricType metric_type = METRIC_L2
) {
    std::vector<T> centroid(padded_dim, 0);
    quantize_qg_batch(data, centroid.data(), num, padded_dim, batch_data, metric_type);
}

inline void quantize_split_batch(
    const float* data,
    const float* centroid,
    size_t num_points,
    size_t padded_dim,
    size_t ex_bits,
    char* batch_data,
    char* ex_data,
    MetricType metric_type = METRIC_L2,
    RabitqConfig config = RabitqConfig()
) {
    quantize_one_batch(data, centroid, num_points, padded_dim, batch_data, metric_type);

    if (ex_bits > 0) {
        for (size_t i = 0; i < num_points; ++i) {
            quantize_compact_ex_bits(
                data, centroid, padded_dim, ex_bits, ex_data, metric_type, config
            );

            ex_data += ExDataMap<float>::data_bytes(padded_dim, ex_bits);
            data += padded_dim;
        }
    }
}

}  // namespace rabitqlib::quant
