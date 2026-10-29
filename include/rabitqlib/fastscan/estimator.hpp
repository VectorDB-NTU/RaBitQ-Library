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
 * @param query_wrapper prepared query with the matching centroid correction already set
 * @param padded_dim dim, must be multiple of 16
 * @param estimated_distances estimated distance
 * @param lower_bounds lower endpoint of the probabilistic distance interval;
 *                     not a deterministic bound for every query
 * @param ip_x0_qr  intermediate result for re-ranking
 * @param use_hacc must match the mode used to prepare query_wrapper
 * All three output buffers need 32 floats, including for a partial batch.
 * Throws std::invalid_argument for an unprepared or mismatched query.
 * The margin is f_error * g_error; see docs/docs/rabitq/estimator.md,
 * "Distance Estimation", for the factor definitions and probabilistic limits.
 */
inline void split_batch_estdist(
    const char* batch_data,
    const SplitBatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances,
    float* lower_bounds,
    float* ip_x0_qr,
    bool use_hacc
) {
    if (padded_dim == 0 || query_wrapper.padded_dim() != padded_dim) {
        throw std::invalid_argument(
            "FastScan scoring requires a prepared query with matching dimension"
        );
    }
    if (query_wrapper.use_hacc() != use_hacc) {
        throw std::invalid_argument("FastScan scoring mode must match the prepared query");
    }
    simd::split_batch_estdist(
        batch_data,
        query_wrapper,
        padded_dim,
        estimated_distances,
        lower_bounds,
        ip_x0_qr,
        use_hacc
    );
}

/**
 * Estimate distances from QGBatchDataMap without producing refinement or error terms.
 * estimated_distances needs 32 values of T, even for a partial batch.
 * Throws std::invalid_argument for an unprepared or mismatched query.
 */
template <typename T, typename TA = int32_t>
inline void qg_batch_estdist(
    const char* batch_data,
    const BatchQuery<T>& query_wrapper,
    size_t padded_dim,
    T* estimated_distances
) {
    static_assert(std::is_same_v<TA, int32_t>, "FastScan accumulation requires int32_t");
    if (padded_dim == 0 || query_wrapper.padded_dim() != padded_dim) {
        throw std::invalid_argument(
            "FastScan scoring requires a prepared query with matching dimension"
        );
    }
    if constexpr (std::is_same_v<T, float>) {
        simd::qg_batch_estdist(batch_data, query_wrapper, padded_dim, estimated_distances);
        return;
    }

    ConstQGBatchDataMap<T> cur_batch(batch_data, padded_dim);
    std::array<TA, fastscan::kBatchSize> accumulated_values{};
    fastscan::accumulate(
        cur_batch.bin_code(), query_wrapper.lut(), accumulated_values.data(), padded_dim
    );
    ConstRowMajorArrayMap<TA> ip_arr(accumulated_values.data(), 1, fastscan::kBatchSize);
    std::array<T, fastscan::kBatchSize> f_add_values;
    std::array<T, fastscan::kBatchSize> f_rescale_values;
    cur_batch.f_add().copy_to(f_add_values.data(), f_add_values.size());
    cur_batch.f_rescale().copy_to(f_rescale_values.data(), f_rescale_values.size());
    ConstRowMajorArrayMap<T> f_add_arr(f_add_values.data(), 1, fastscan::kBatchSize);
    ConstRowMajorArrayMap<T> f_rescale_arr(
        f_rescale_values.data(), 1, fastscan::kBatchSize
    );

    RowMajorArrayMap<T> est_dist_arr(estimated_distances, 1, fastscan::kBatchSize);

    est_dist_arr = f_add_arr + query_wrapper.g_add() +
                   (f_rescale_arr * (query_wrapper.delta() * (ip_arr.template cast<T>()) +
                                     query_wrapper.sum_vl_lut() + query_wrapper.k1xsumq()));
}

}  // namespace rabitqlib
