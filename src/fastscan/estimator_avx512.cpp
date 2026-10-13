#include <cstddef>
#include <stdexcept>

#include "estimator_kernels.hpp"
#include "rabitqlib/fastscan/detail/estimator_dispatch.hpp"
#include "rabitqlib/fastscan/detail/simd.hpp"

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
    if (padded_dim == 0 || padded_dim % 16 != 0) {
        throw std::invalid_argument(
            use_hacc ? "high-accuracy FastScan dimension must be a positive multiple of 16"
                     : "FastScan dimension must be a positive multiple of 16"
        );
    }
    // This backend already requires the highest FastScan ISA. Its LUT format
    // matches AVX-512 accumulation, so a second dispatch is unnecessary.
    split_batch_estdist_impl<
        fastscan::simd::accumulate_avx512,
        fastscan::simd::accumulate_hacc_avx512>(
        batch_data, q_obj, padded_dim, est_distance, low_distance, ip_x0_qr, use_hacc
    );
}

void qg_batch_estdist_avx512(
    const char* batch_data,
    const BatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance
) {
    if (padded_dim == 0 || padded_dim % 16 != 0) {
        throw std::invalid_argument("FastScan dimension must be a positive multiple of 16");
    }
    qg_batch_estdist_impl<fastscan::simd::accumulate_avx512>(
        batch_data, q_obj, padded_dim, est_distance
    );
}

}  // namespace rabitqlib::simd
