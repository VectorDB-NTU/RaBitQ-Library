#include <cstddef>

#include "estimator_kernels.hpp"
#include "rabitqlib/simd/estimator_dispatch.hpp"

namespace rabitqlib::simd {
void split_batch_estdist_avx512(
    const char* batch_data,
    const SplitBatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    float* low_distance,
    float* ip_x0_qr,
    bool use_hacc
) {
    split_batch_estdist_impl(
        batch_data, q_obj, padded_dim, est_distance, low_distance, ip_x0_qr, use_hacc
    );
}

void qg_batch_estdist_avx512(
    const char* batch_data,
    const BatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance
) {
    qg_batch_estdist_impl(batch_data, q_obj, padded_dim, est_distance);
}

}  // namespace rabitqlib::simd
