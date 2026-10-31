#pragma once

#include <cstddef>
#include <cstdint>

#include "rabitqlib/core/data_layout.hpp"
#include "rabitqlib/core/query.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/warmup_space.hpp"

namespace rabitqlib {
namespace detail {
struct DistanceEstimate {
    float distance;
    float lower_bound;
    float inner_product;
};
}  // namespace detail

/**
 * @brief Refine a split-code estimate using the remaining ex_bits bits.
 *
 * Query and codes must share the rotation, padded dimension, metric and bit
 * configuration. Set the query's centroid correction before scoring.
 * ex_data uses ExDataMap<float> from core/data_layout.hpp; ip_func must decode
 * its packed ex_bits layout. ip_x0_qr is the base-code inner product for this
 * same vector and query, as returned by the matching base-distance estimator.
 * Returns the refined squared-L2 or 1 - dot estimate, without an error margin.
 */
template <typename Query>
inline float split_distance_boosting(
    const char* ex_data,
    float (*ip_func)(const float*, const uint8_t*, size_t),
    const Query& query_wrapper,
    size_t padded_dim,
    size_t ex_bits,
    float ip_x0_qr
) {
    ConstExDataMap<float> cur_ex(ex_data, padded_dim, ex_bits);

    float ex_dist = cur_ex.f_add_ex() + query_wrapper.g_add() +
                    (cur_ex.f_rescale_ex() *
                     (static_cast<float>(1 << ex_bits) * ip_x0_qr +
                      ip_func(query_wrapper.rotated_query(), cur_ex.ex_code(), padded_dim) +
                      query_wrapper.kbxsumq()));

    return ex_dist;
}

template <class Kernel, typename Query>
inline void split_single_fulldist_direct(
    const char* bin_data,
    const char* ex_data,
    float (*ip_func)(const float*, const uint8_t*, size_t),
    const Query& query_wrapper,
    size_t padded_dim,
    size_t ex_bits,
    float& estimated_distance,
    float& lower_bound,
    float& ip_x0_qr,
    float g_add,
    float g_error
) {
    ConstBinDataMap<float> cur_bin(bin_data, padded_dim);
    ConstExDataMap<float> cur_ex(ex_data, padded_dim, ex_bits);

    // Direct HNSW kernels retain their legacy word-typed ABI. Generic kernels use the
    // byte-oriented overload and do not manufacture a typed view of packed storage.
    ip_x0_qr = Kernel::mask_ip_x0_q(
        query_wrapper.rotated_query(),
        reinterpret_cast<const uint64_t*>(cur_bin.bin_code()),
        padded_dim
    );

    estimated_distance =
        cur_ex.f_add_ex() + g_add +
        (cur_ex.f_rescale_ex() *
         (static_cast<float>(1 << ex_bits) * ip_x0_qr +
          ip_func(query_wrapper.rotated_query(), cur_ex.ex_code(), padded_dim) +
          query_wrapper.kbxsumq()));

    lower_bound = estimated_distance -
                  (cur_bin.f_error() * g_error / static_cast<float>(1 << ex_bits));
}

inline void split_single_fulldist(
    const char* bin_data,
    const char* ex_data,
    float (*ip_func)(const float*, const uint8_t*, size_t),
    const SplitSingleQuery<float>& query_wrapper,
    size_t padded_dim,
    size_t ex_bits,
    float& estimated_distance,
    float& lower_bound,
    float& ip_x0_qr,
    float g_add,
    float g_error
) {
    ConstBinDataMap<float> cur_bin(bin_data, padded_dim);
    ConstExDataMap<float> cur_ex(ex_data, padded_dim, ex_bits);

    ip_x0_qr = mask_ip_x0_q(query_wrapper.rotated_query(), cur_bin.bin_code(), padded_dim);

    estimated_distance =
        cur_ex.f_add_ex() + g_add +
        (cur_ex.f_rescale_ex() *
         (static_cast<float>(1 << ex_bits) * ip_x0_qr +
          ip_func(query_wrapper.rotated_query(), cur_ex.ex_code(), padded_dim) +
          query_wrapper.kbxsumq()));

    lower_bound = estimated_distance -
                  (cur_bin.f_error() * g_error / static_cast<float>(1 << ex_bits));
}

inline void split_single_estdist(
    const char* bin_data,
    const SplitSingleQuery<float>& query_wrapper,
    size_t padded_dim,
    float& ip_x0_qr,
    float& estimated_distance,
    float& lower_bound,
    float g_add = 0,
    float g_error = 0
) {
    ConstBinDataMap<float> cur_bin(bin_data, padded_dim);

    ip_x0_qr = warmup_ip_x0_q_512(
        cur_bin.bin_code(),
        query_wrapper.query_bin(),
        query_wrapper.delta(),
        query_wrapper.vl(),
        padded_dim,
        query_wrapper.num_bits()
    );

    estimated_distance = cur_bin.f_add() + g_add +
                         (cur_bin.f_rescale() * (ip_x0_qr + query_wrapper.k1xsumq()));

    lower_bound = estimated_distance - (cur_bin.f_error() * g_error);
}

template <class Kernel>
inline void split_single_estdist_direct(
    const char* bin_data,
    const SplitSingleQuery<float>& query_wrapper,
    size_t padded_dim,
    float& ip_x0_qr,
    float& estimated_distance,
    float& lower_bound,
    float g_add = 0,
    float g_error = 0
) {
    ConstBinDataMap<float> cur_bin(bin_data, padded_dim);

    // Direct HNSW kernels retain their legacy word-typed ABI; see the note above.
    ip_x0_qr = Kernel::warmup_ip_x0_q_512(
        reinterpret_cast<const uint64_t*>(cur_bin.bin_code()),
        query_wrapper.query_bin(),
        query_wrapper.delta(),
        query_wrapper.vl(),
        padded_dim,
        query_wrapper.num_bits()
    );

    estimated_distance = cur_bin.f_add() + g_add +
                         (cur_bin.f_rescale() * (ip_x0_qr + query_wrapper.k1xsumq()));

    lower_bound = estimated_distance - (cur_bin.f_error() * g_error);
}

inline void xy_single_base_dist(
    const char* base_data,
    float (*base_ip_func)(const float*, const uint8_t*, size_t),
    const SplitSingleQuery<float>& query_wrapper,
    size_t padded_dim,
    size_t base_bits,
    float& ip_base,
    float& estimated_distance,
    float& lower_bound,
    float g_add = 0,
    float g_error = 0
) {
    ConstBaseDataMap<float> cur_base(base_data, padded_dim, base_bits);

    ip_base = base_ip_func(query_wrapper.rotated_query(), cur_base.base_code(), padded_dim);

    estimated_distance = cur_base.f_add() + g_add +
                         (cur_base.f_rescale() * (ip_base + query_wrapper.kbase_sumq()));

    lower_bound = estimated_distance - (cur_base.f_error() * g_error);
}

inline void xy_single_full_dist(
    const char* base_data,
    const char* ex_data,
    float (*base_ip_func)(const float*, const uint8_t*, size_t),
    float (*ex_ip_func)(const float*, const uint8_t*, size_t),
    const SplitSingleQuery<float>& query_wrapper,
    size_t padded_dim,
    size_t base_bits,
    size_t ex_bits,
    float& estimated_distance,
    float& lower_bound,
    float& ip_base,
    float g_add = 0,
    float g_error = 0
) {
    ConstBaseDataMap<float> cur_base(base_data, padded_dim, base_bits);
    ip_base = base_ip_func(query_wrapper.rotated_query(), cur_base.base_code(), padded_dim);

    if (ex_bits == 0) {
        estimated_distance =
            cur_base.f_add() + g_add +
            (cur_base.f_rescale() * (ip_base + query_wrapper.kbase_sumq()));
        lower_bound = estimated_distance - (cur_base.f_error() * g_error);
        return;
    }

    ConstExDataMap<float> cur_ex(ex_data, padded_dim, ex_bits);

    float ip = (static_cast<float>(1 << ex_bits) * ip_base) +
               ex_ip_func(query_wrapper.rotated_query(), cur_ex.ex_code(), padded_dim);

    estimated_distance = cur_ex.f_add_ex() + g_add +
                         (cur_ex.f_rescale_ex() * (ip + query_wrapper.kbxsumq()));

    lower_bound = estimated_distance -
                  (cur_base.f_error() * g_error / static_cast<float>(1 << ex_bits));
}

// Named-result functions keep the numerical implementation and legacy output-reference
// interfaces intact. Internal callers do not need to remember output argument order.
template <class Kernel, typename Query>
inline detail::DistanceEstimate estimate_split_single_full_direct(
    const char* bin_data,
    const char* ex_data,
    float (*ip_func)(const float*, const uint8_t*, size_t),
    const Query& query_wrapper,
    size_t padded_dim,
    size_t ex_bits,
    float g_add,
    float g_error
) {
    detail::DistanceEstimate result;
    split_single_fulldist_direct<Kernel>(
        bin_data,
        ex_data,
        ip_func,
        query_wrapper,
        padded_dim,
        ex_bits,
        result.distance,
        result.lower_bound,
        result.inner_product,
        g_add,
        g_error
    );
    return result;
}

inline detail::DistanceEstimate estimate_split_single_full(
    const char* bin_data,
    const char* ex_data,
    float (*ip_func)(const float*, const uint8_t*, size_t),
    const SplitSingleQuery<float>& query_wrapper,
    size_t padded_dim,
    size_t ex_bits,
    float g_add,
    float g_error
) {
    detail::DistanceEstimate result;
    split_single_fulldist(
        bin_data,
        ex_data,
        ip_func,
        query_wrapper,
        padded_dim,
        ex_bits,
        result.distance,
        result.lower_bound,
        result.inner_product,
        g_add,
        g_error
    );
    return result;
}

inline detail::DistanceEstimate estimate_split_single(
    const char* bin_data,
    const SplitSingleQuery<float>& query_wrapper,
    size_t padded_dim,
    float g_add = 0,
    float g_error = 0
) {
    detail::DistanceEstimate result;
    split_single_estdist(
        bin_data,
        query_wrapper,
        padded_dim,
        result.inner_product,
        result.distance,
        result.lower_bound,
        g_add,
        g_error
    );
    return result;
}

template <class Kernel>
inline detail::DistanceEstimate estimate_split_single_direct(
    const char* bin_data,
    const SplitSingleQuery<float>& query_wrapper,
    size_t padded_dim,
    float g_add = 0,
    float g_error = 0
) {
    detail::DistanceEstimate result;
    split_single_estdist_direct<Kernel>(
        bin_data,
        query_wrapper,
        padded_dim,
        result.inner_product,
        result.distance,
        result.lower_bound,
        g_add,
        g_error
    );
    return result;
}

inline detail::DistanceEstimate estimate_xy_base(
    const char* base_data,
    float (*base_ip_func)(const float*, const uint8_t*, size_t),
    const SplitSingleQuery<float>& query_wrapper,
    size_t padded_dim,
    size_t base_bits,
    float g_add = 0,
    float g_error = 0
) {
    detail::DistanceEstimate result;
    xy_single_base_dist(
        base_data,
        base_ip_func,
        query_wrapper,
        padded_dim,
        base_bits,
        result.inner_product,
        result.distance,
        result.lower_bound,
        g_add,
        g_error
    );
    return result;
}

inline detail::DistanceEstimate estimate_xy_full(
    const char* base_data,
    const char* ex_data,
    float (*base_ip_func)(const float*, const uint8_t*, size_t),
    float (*ex_ip_func)(const float*, const uint8_t*, size_t),
    const SplitSingleQuery<float>& query_wrapper,
    size_t padded_dim,
    size_t base_bits,
    size_t ex_bits,
    float g_add = 0,
    float g_error = 0
) {
    detail::DistanceEstimate result;
    xy_single_full_dist(
        base_data,
        ex_data,
        base_ip_func,
        ex_ip_func,
        query_wrapper,
        padded_dim,
        base_bits,
        ex_bits,
        result.distance,
        result.lower_bound,
        result.inner_product,
        g_add,
        g_error
    );
    return result;
}

}  // namespace rabitqlib
