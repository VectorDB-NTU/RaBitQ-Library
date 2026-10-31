#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "rabitqlib/fastscan/detail/estimator_dispatch.hpp"
#include "rabitqlib/fastscan/detail/qg_estimator.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
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
    } else {
        fastscan::detail::qg_batch_estdist_impl<T, fastscan::accumulate>(
            batch_data, query_wrapper, padded_dim, estimated_distances
        );
    }
}

}  // namespace rabitqlib
