#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib {
template <typename T>
class SplitBatchQuery;
template <typename T>
class BatchQuery;

namespace simd {
void split_batch_estdist(
    const char* batch_data,
    const SplitBatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances,
    float* lower_bounds,
    float* ip_x0_qr,
    bool use_hacc
);

void split_batch_estdist_generic(
    const char* batch_data,
    const SplitBatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances,
    float* lower_bounds,
    float* ip_x0_qr,
    bool use_hacc
);

void split_batch_estdist_avx2(
    const char* batch_data,
    const SplitBatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances,
    float* lower_bounds,
    float* ip_x0_qr,
    bool use_hacc
);

void split_batch_estdist_avx512(
    const char* batch_data,
    const SplitBatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances,
    float* lower_bounds,
    float* ip_x0_qr,
    bool use_hacc
);

void qg_batch_estdist(
    const char* batch_data,
    const BatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances
);
void qg_batch_estdist_generic(
    const char* batch_data,
    const BatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances
);
void qg_batch_estdist_avx2(
    const char* batch_data,
    const BatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances
);
void qg_batch_estdist_avx512(
    const char* batch_data,
    const BatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances
);

uint32_t qg_candidate_mask_generic(const float* distances, float threshold);
uint32_t qg_candidate_mask_avx2(const float* distances, float threshold);

uint32_t qg_batch_estdist_mask(
    const char* batch_data,
    const BatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances,
    float threshold
);
uint32_t qg_batch_estdist_mask_generic(
    const char* batch_data,
    const BatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances,
    float threshold
);
uint32_t qg_batch_estdist_mask_avx2(
    const char* batch_data,
    const BatchQuery<float>& query_wrapper,
    size_t padded_dim,
    float* estimated_distances,
    float threshold
);
}  // namespace simd
}  // namespace rabitqlib
