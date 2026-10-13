#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/detail/estimator_dispatch.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"

namespace rabitqlib {
/**
 * @brief Use FastScan to estimate batch distance
 *
 * @param batch_data codes and factors in fastscan/data_layout.hpp's BatchDataMap layout
 * @param q_obj prepared query with the matching centroid correction already set
 * @param padded_dim dim, must be multiple of 16
 * @param est_distance estimated distance
 * @param low_distance lower endpoint of the probabilistic distance interval;
 *                     not a deterministic bound for every query
 * @param ip_x0_qr  intermediate result for re-ranking
 * @param use_hacc must match the mode used to prepare q_obj
 * All three output buffers need 32 floats, including for a partial batch.
 * Throws std::invalid_argument for an unprepared or mismatched query.
 * The margin is f_error * g_error; see docs/docs/rabitq/estimator.md,
 * "Distance Estimation", for the factor definitions and probabilistic limits.
 */
inline void split_batch_estdist(
    const char* batch_data,
    const SplitBatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    float* low_distance,
    float* ip_x0_qr,
    bool use_hacc
) {
    if (padded_dim == 0 || q_obj.padded_dim() != padded_dim) {
        throw std::invalid_argument(
            "FastScan scoring requires a prepared query with matching dimension"
        );
    }
    if (q_obj.use_hacc() != use_hacc) {
        throw std::invalid_argument("FastScan scoring mode must match the prepared query");
    }
    simd::split_batch_estdist(
        batch_data, q_obj, padded_dim, est_distance, low_distance, ip_x0_qr, use_hacc
    );
}

/**
 * Estimate distances from QGBatchDataMap without producing refinement or error terms.
 * est_distance needs 32 values of T, even for a partial batch.
 * Throws std::invalid_argument for an unprepared or mismatched query.
 */
template <typename T, typename TA = int32_t>
inline void qg_batch_estdist(
    const char* batch_data, const BatchQuery<T>& q_obj, size_t padded_dim, T* est_distance
) {
    static_assert(std::is_same_v<TA, int32_t>, "FastScan accumulation requires int32_t");
    if (padded_dim == 0 || q_obj.padded_dim() != padded_dim) {
        throw std::invalid_argument(
            "FastScan scoring requires a prepared query with matching dimension"
        );
    }
    if constexpr (std::is_same_v<T, float>) {
        simd::qg_batch_estdist(batch_data, q_obj, padded_dim, est_distance);
        return;
    }

    ConstQGBatchDataMap<T> cur_batch(batch_data, padded_dim);
    std::array<TA, fastscan::kBatchSize> accu_res{};
    fastscan::accumulate(cur_batch.bin_code(), q_obj.lut(), accu_res.data(), padded_dim);
    ConstRowMajorArrayMap<TA> ip_arr(accu_res.data(), 1, fastscan::kBatchSize);
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

}  // namespace rabitqlib
