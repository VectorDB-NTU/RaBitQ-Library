#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rabitqlib/core/data_layout.hpp"
#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/core/detail/rabitq_impl.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/types.hpp"  // IWYU pragma: export

namespace rabitqlib::quant {

// Prepare a constant rescaling factor for total_bits > 1. The one-bit path
// needs no extra-code scaling and leaves the default configuration unchanged.
inline RabitqConfig faster_config(size_t dim, size_t total_bits) {
    RabitqConfig config;
    if (total_bits > 1) {
        config.t_const =
            rabitq_impl::ex_bits::get_const_scaling_factors(dim, total_bits - 1);
    }
    return config;
}

template <typename T>
inline void quantize_compact_one_bit(
    const T* data,
    const T* centroid,
    size_t padded_dim,
    char* bin_data,
    MetricType metric_type = METRIC_L2
) {
    BinDataMap<T> cur_bin_data(bin_data, padded_dim);
    T f_add;
    T f_rescale;
    T f_error;

    rabitq_impl::one_bit::one_bit_compact_code_to_bytes<T, uint64_t>(
        data,
        centroid,
        padded_dim,
        cur_bin_data.bin_code(),
        f_add,
        f_rescale,
        f_error,
        metric_type
    );
    cur_bin_data.f_add() = f_add;
    cur_bin_data.f_rescale() = f_rescale;
    cur_bin_data.f_error() = f_error;
}

template <typename T, typename TC>
inline void quantize_compact_one_bit(
    const T* data,
    const T* centroid,
    size_t padded_dim,
    TC compact_code,
    T& f_add,
    T& f_rescale,
    T& ferror,
    MetricType metric_type = METRIC_L2
) {
    rabitq_impl::one_bit::one_bit_compact_code(
        data, centroid, padded_dim, compact_code, f_add, f_rescale, ferror, metric_type
    );
}

template <typename T, typename TC>
inline void quantize_compact_one_bit(
    const T* data,
    size_t padded_dim,
    TC compact_code,
    T& f_add,
    T& f_rescale,
    T& ferror,
    MetricType metric_type = METRIC_L2
) {
    std::vector<T> centroid(padded_dim, 0);
    rabitq_impl::one_bit::one_bit_compact_code(
        data,
        centroid.data(),
        padded_dim,
        compact_code,
        f_add,
        f_rescale,
        ferror,
        metric_type
    );
}

template <typename T>
inline void quantize_compact_ex_bits(
    const T* data,
    const T* centroid,
    size_t padded_dim,
    size_t ex_bits,
    char* ex_data,
    MetricType metric_type = METRIC_L2,
    RabitqConfig config = RabitqConfig()
) {
    ExDataMap<T> cur_ex_data(ex_data, padded_dim, ex_bits);

    // we do not use this error factor here
    T f_add_ex;
    T f_rescale_ex;
    T ex_error;

    rabitq_impl::ex_bits::ex_bits_compact_code(
        data,
        centroid,
        padded_dim,
        ex_bits,
        cur_ex_data.ex_code(),
        f_add_ex,
        f_rescale_ex,
        ex_error,
        metric_type,
        config.t_const
    );
    cur_ex_data.f_add_ex() = f_add_ex;
    cur_ex_data.f_rescale_ex() = f_rescale_ex;
}

inline void quantize_split_single(
    const float* data,
    const float* centroid,
    size_t padded_dim,
    size_t ex_bits,
    char* bin_data,
    char* ex_data,
    MetricType metric_type = METRIC_L2,
    RabitqConfig config = RabitqConfig()
) {
    quantize_compact_one_bit(data, centroid, padded_dim, bin_data, metric_type);

    if (ex_bits > 0) {
        quantize_compact_ex_bits(
            data, centroid, padded_dim, ex_bits, ex_data, metric_type, config
        );
    }
}

inline void quantize_xy_single(
    const float* data,
    const float* centroid,
    size_t padded_dim,
    size_t base_bits,
    size_t ex_bits,
    char* base_data,
    char* ex_data,
    MetricType metric_type = METRIC_L2,
    RabitqConfig config = RabitqConfig()
) {
    rabitq_impl::xy_bits::split_code_with_factor<float>(
        data,
        centroid,
        padded_dim,
        base_bits,
        ex_bits,
        base_data,
        ex_data,
        metric_type,
        config.t_const
    );
}

template <typename T, typename TP>
inline void quantize_full_single(
    const T* data,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type = METRIC_L2,
    RabitqConfig config = RabitqConfig()
) {
    std::vector<T> centroid(dim, 0);
    rabitq_impl::total_bits::rabitq_full_impl<T, TP>(
        data,
        centroid.data(),
        dim,
        total_bits,
        total_code,
        f_add,
        f_rescale,
        f_error,
        metric_type,
        config.t_const
    );
}

template <typename T, typename TP>
inline void quantize_full_single(
    const T* data,
    const T* centroid,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type = METRIC_L2,
    RabitqConfig config = RabitqConfig()
) {
    rabitq_impl::total_bits::rabitq_full_impl<T, TP>(
        data,
        centroid,
        dim,
        total_bits,
        total_code,
        f_add,
        f_rescale,
        f_error,
        metric_type,
        config.t_const
    );
}

template <typename T, typename TP>
inline void quantize_scalar(
    const T* data,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& delta,
    T& vl,
    RabitqConfig config = RabitqConfig(),
    ScalarQuantizerType scalar_quantizer_type = ScalarQuantizerType::RECONSTRUCTION
) {
    std::vector<T> centroid(dim, 0);
    rabitq_impl::total_bits::rabitq_scalar_impl<T, TP>(
        data,
        centroid.data(),
        dim,
        total_bits,
        total_code,
        delta,
        vl,
        config.t_const,
        scalar_quantizer_type
    );
}

template <typename T, typename TP>
inline void quantize_scalar(
    const T* data,
    const T* centroid,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& delta,
    T& vl,
    RabitqConfig config = RabitqConfig()
) {
    rabitq_impl::total_bits::rabitq_scalar_impl<T, TP>(
        data, centroid, dim, total_bits, total_code, delta, vl, config.t_const
    );
}

template <typename T, typename TP>
inline void reconstruct_vec(
    const TP* quantized_vec, T delta, T vl, size_t dim, T* results
) {
    RowMajorArrayMap<T> result_arr(results, 1, dim);
    result_arr =
        (ConstRowMajorArrayMap<TP>(quantized_vec, 1, dim).template cast<T>() * delta) + vl;
}

template <typename T, typename TP>
inline void reconstruct_full_vec(
    const TP* quantized_vec,
    const T* centroid,
    size_t dim,
    size_t bits,
    T f_rescale,
    T* results,
    MetricType metric_type = METRIC_L2
) {
    const T code_center = -static_cast<T>((static_cast<size_t>(1) << bits) - 1) / 2;
    const T scale = metric_type == METRIC_L2 ? -f_rescale / 2 : -f_rescale;
    RowMajorArrayMap<T> result_arr(results, 1, dim);
    result_arr =
        ConstRowMajorArrayMap<T>(centroid, 1, dim) +
        scale * (ConstRowMajorArrayMap<TP>(quantized_vec, 1, dim).template cast<T>() +
                 code_center);
}

namespace detail {
template <typename TF>
inline TF full_est_dist_from_ip(
    TF ip, size_t bits, TF f_add, TF f_rescale, TF g_add, TF k1xsumq
) {
    return f_add + g_add +
           (f_rescale * (ip + k1xsumq * static_cast<float>((1 << bits) - 1)));
}
}  // namespace detail

template <typename TF, typename TI>
inline TF full_est_dist(
    const TI* quantized_vec,
    const TF* query,
    TF (*ip_func)(const TF*, const TI*, size_t),
    size_t dim,
    size_t bits,
    TF f_add,
    TF f_rescale,
    TF g_add,
    TF k1xsumq
) {
    return detail::full_est_dist_from_ip(
        ip_func(query, quantized_vec, dim), bits, f_add, f_rescale, g_add, k1xsumq
    );
}
}  // namespace rabitqlib::quant
