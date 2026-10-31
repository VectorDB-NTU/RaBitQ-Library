#include <cstddef>
#include <cstdint>

#include "estimator_kernels.hpp"
#include "rabitqlib/fastscan/detail/estimator_dispatch.hpp"
#include "rabitqlib/fastscan/detail/qg_estimator.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"

namespace rabitqlib::simd {
namespace {
uint32_t candidate_mask_impl(const float* distances, float threshold) {
    uint32_t mask = 0;
    for (size_t lane = 0; lane < fastscan::kBatchSize; ++lane) {
        mask |= static_cast<uint32_t>(!(distances[lane] > threshold)) << lane;
    }
    return mask;
}
}  // namespace

void split_batch_estdist_generic(
    const char* batch_data,
    const SplitBatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances,
    float* lower_bounds,
    float* ip_x0_qr,
    bool use_hacc
) {
    // Explicit backend entry points also accept query LUTs prepared by the host
    // backend. In particular, AVX2 correction can receive AVX-512 HACC bytes.
    split_batch_estdist_impl<fastscan::accumulate, fastscan::accumulate_hacc>(
        batch_data,
        query_wrapper,
        padded_dim,
        estimated_distances,
        lower_bounds,
        ip_x0_qr,
        use_hacc
    );
}

void qg_batch_estdist_generic(
    const char* batch_data,
    const BatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances
) {
    fastscan::detail::qg_batch_estdist_impl<float, fastscan::accumulate>(
        batch_data, query_wrapper, padded_dim, estimated_distances
    );
}

uint32_t qg_candidate_mask_generic(const float* distances, float threshold) {
    return candidate_mask_impl(distances, threshold);
}

uint32_t qg_batch_estdist_mask_generic(
    const char* batch_data,
    const BatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances,
    float threshold
) {
    fastscan::detail::qg_batch_estdist_impl<float, fastscan::accumulate>(
        batch_data, query_wrapper, padded_dim, estimated_distances
    );
    return candidate_mask_impl(estimated_distances, threshold);
}

}  // namespace rabitqlib::simd
