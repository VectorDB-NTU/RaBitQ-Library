#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "../simd/dispatch_resolver.hpp"
#include "fastscan_backend.hpp"
#include "rabitqlib/fastscan/detail/estimator_dispatch.hpp"
#include "rabitqlib/fastscan/detail/simd.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#if defined(__aarch64__) || defined(_M_ARM64)
#include "rabitqlib/utils/cpu_features.hpp"
#endif

namespace rabitqlib::fastscan::detail {
namespace {
constexpr Backend kGenericBackend{
    "generic",
    HaccLutLayout::Pair32,
    simd::pack_lut_generic,
    simd::transfer_lut_hacc_generic,
    simd::accumulate_unsupported,
    simd::accumulate_hacc_generic,
    rabitqlib::simd::qg_batch_estdist_generic,
    rabitqlib::simd::qg_batch_estdist_mask_generic,
    rabitqlib::simd::split_batch_estdist_generic};

#if defined(__x86_64__) || defined(_M_X64)
constexpr Backend kAvx2Backend{
    "avx2",
    HaccLutLayout::Pair32,
    simd::pack_lut_avx2,
    simd::transfer_lut_hacc_avx2,
    simd::accumulate_avx2,
    simd::accumulate_hacc_avx2,
    rabitqlib::simd::qg_batch_estdist_avx2,
    rabitqlib::simd::qg_batch_estdist_mask_avx2,
    rabitqlib::simd::split_batch_estdist_avx2};
constexpr Backend kAvx512Backend{
    "avx512",
    HaccLutLayout::Quad64,
    simd::pack_lut_avx512,
    simd::transfer_lut_hacc_avx512,
    simd::accumulate_avx512,
    simd::accumulate_hacc_avx512,
    rabitqlib::simd::qg_batch_estdist_avx512,
    rabitqlib::simd::qg_batch_estdist_mask_avx2,
    rabitqlib::simd::split_batch_estdist_avx512};
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
constexpr Backend kNeonBackend{
    "neon",
    HaccLutLayout::Pair32,
    simd::pack_lut_neon,
    simd::transfer_lut_hacc_generic,
    simd::accumulate_neon,
    simd::accumulate_hacc_neon,
    rabitqlib::simd::qg_batch_estdist_generic,
    rabitqlib::simd::qg_batch_estdist_mask_generic,
    rabitqlib::simd::split_batch_estdist_generic};
#endif
}  // namespace

RABITQ_COLD_INIT const Backend& selected_backend() {
#if defined(__aarch64__) || defined(_M_ARM64)
    static const Backend* const backend = rabitqlib::simd::resolve_optional_kernel(
        &kNeonBackend, &kGenericBackend, cpu::has_neon()
    );
#else
    static const Backend* const backend =
        RABITQ_RESOLVE(&kAvx512Backend, &kAvx2Backend, &kGenericBackend);
#endif
    return *backend;
}
}  // namespace rabitqlib::fastscan::detail

namespace rabitqlib::simd {

void qg_batch_estdist(
    const char* batch_data,
    const BatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance
) {
    static const auto kQgBatchEstdistFn = fastscan::detail::selected_backend().qg;

    kQgBatchEstdistFn(batch_data, q_obj, padded_dim, est_distance);
}

uint32_t qg_batch_estdist_mask(
    const char* batch_data,
    const BatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    float threshold
) {
    static const auto kQgBatchEstdistMaskFn = fastscan::detail::selected_backend().qg_mask;

    return kQgBatchEstdistMaskFn(batch_data, q_obj, padded_dim, est_distance, threshold);
}

void split_batch_estdist(
    const char* batch_data,
    const SplitBatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    float* low_distance,
    float* ip_x0_qr,
    bool use_hacc
) {
    static const auto kSplitBatchEstdistFn = fastscan::detail::selected_backend().split;

    kSplitBatchEstdistFn(
        batch_data, q_obj, padded_dim, est_distance, low_distance, ip_x0_qr, use_hacc
    );
}

}  // namespace rabitqlib::simd

namespace rabitqlib::fastscan {

template <>
void pack_lut<float>(size_t dim, const float* __restrict__ query, float* __restrict__ lut) {
    static const auto kPackLutFn = detail::selected_backend().pack_lut;

    kPackLutFn(dim, query, lut);
}

void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    int32_t* __restrict__ result,
    size_t dim
) {
    static const auto kAccumulateFn = detail::selected_backend().accumulate;

    if (dim == 0 || dim % 16 != 0) {
        throw std::invalid_argument("FastScan dimension must be a positive multiple of 16");
    }
    kAccumulateFn(codes, lp_table, result, dim);
}

void transfer_lut_hacc(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
    static const auto kTransferLutHaccFn = detail::selected_backend().transfer_hacc;

    if (dim == 0 || dim % 16 != 0) {
        throw std::invalid_argument(
            "high-accuracy FastScan dimension must be a positive multiple of 16"
        );
    }
    kTransferLutHaccFn(lut, dim, hc_lut);
}

void accumulate_hacc(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
) {
    static const auto kAccumulateHaccFn = detail::selected_backend().accumulate_hacc;

    if (dim == 0 || dim % 16 != 0) {
        throw std::invalid_argument(
            "high-accuracy FastScan dimension must be a positive multiple of 16"
        );
    }
    kAccumulateHaccFn(codes, hc_lut, accu_res, dim);
}

}  // namespace rabitqlib::fastscan
