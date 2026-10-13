#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "estimator_kernels.hpp"
#include "rabitqlib/fastscan/detail/estimator_dispatch.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"

namespace rabitqlib::simd {
namespace {
uint32_t candidate_mask_impl(const float* distances, float threshold) {
    const __m256 limit = _mm256_set1_ps(threshold);
    uint32_t mask = 0;
    for (size_t lane = 0; lane < 32; lane += 8) {
        const __m256 values = _mm256_loadu_ps(distances + lane);
        const __m256 candidates = _mm256_cmp_ps(values, limit, _CMP_NGT_UQ);
        mask |= static_cast<uint32_t>(_mm256_movemask_ps(candidates)) << lane;
    }
    return mask;
}
}  // namespace

void split_batch_estdist_avx2(
    const char* batch_data,
    const SplitBatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    float* low_distance,
    float* ip_x0_qr,
    bool use_hacc
) {
    // Explicit backend entry points also accept query LUTs prepared by the host
    // backend. In particular, AVX2 correction can receive AVX-512 HACC bytes.
    split_batch_estdist_impl<fastscan::accumulate, fastscan::accumulate_hacc>(
        batch_data, q_obj, padded_dim, est_distance, low_distance, ip_x0_qr, use_hacc
    );
}

void qg_batch_estdist_avx2(
    const char* batch_data,
    const BatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance
) {
    qg_batch_estdist_impl<fastscan::accumulate>(
        batch_data, q_obj, padded_dim, est_distance
    );
}

uint32_t qg_candidate_mask_avx2(const float* distances, float threshold) {
    return candidate_mask_impl(distances, threshold);
}

uint32_t qg_batch_estdist_mask_avx2(
    const char* batch_data,
    const BatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    float threshold
) {
    qg_batch_estdist_impl<fastscan::accumulate>(
        batch_data, q_obj, padded_dim, est_distance
    );
    return candidate_mask_impl(est_distance, threshold);
}

}  // namespace rabitqlib::simd
