#include "rabitqlib/simd/dispatch.hpp"

#include <cstddef>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#include "rabitqlib/simd/estimator_dispatch.hpp"
#include "rabitqlib/simd/fastscan_dispatch.hpp"
#include "rabitqlib/simd/hnsw_dispatch.hpp"
#include "rabitqlib/simd/matrix_dispatch.hpp"
#include "rabitqlib/simd/pack_excode_dispatch.hpp"
#include "rabitqlib/simd/quantization_dispatch.hpp"
#include "rabitqlib/simd/rotator_dispatch.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/simd/warmup_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/warmup_space.hpp"

namespace rabitqlib::simd {
namespace {
// Resolve once during static initialization; wrappers never repeat CPU checks.
// The override is used only by kernels needing a stricter AVX-512 subset.
template <typename Function>
Function resolve_kernel(
    Function avx512,
    Function avx2,
    Function fallback,
    bool avx512_supported = cpu::has_avx512_core()
) {
    if (avx512_supported) {
        return avx512;
    }
    if (cpu::has_avx2()) {
        return avx2;
    }
    return fallback;
}

template <typename Function>
Function resolve_kernel(Function avx2, Function fallback) {
    if (cpu::has_avx2()) {
        return avx2;
    }
    return fallback;
}

}  // namespace

const auto kMatrixProductFn =
    resolve_kernel(matrix_product_avx512, matrix_product_avx2, matrix_product_generic);

void matrix_product(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
) {
    kMatrixProductFn(left, right, result, rows, inner, cols);
}

const auto kMatrixProductTransposedFn = resolve_kernel(
    matrix_product_transposed_avx512,
    matrix_product_transposed_avx2,
    matrix_product_transposed_generic
);

void matrix_product_transposed(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
) {
    kMatrixProductTransposedFn(left, right, result, rows, inner, cols);
}

const auto kRowNormsFn =
    resolve_kernel(row_norms_avx512, row_norms_avx2, row_norms_generic);

void row_norms(const float* data, float* result, size_t rows, size_t dim) {
    kRowNormsFn(data, result, rows, dim);
}

const auto kPairwiseDistancesLowerFn = resolve_kernel(
    pairwise_distances_lower_avx512,
    pairwise_distances_lower_avx2,
    pairwise_distances_lower_generic
);

void pairwise_distances_lower(
    const float* data,
    float* result,
    float* norms_data,
    size_t size,
    size_t dim,
    bool inner_product
) {
    kPairwiseDistancesLowerFn(data, result, norms_data, size, dim, inner_product);
}

const auto kQgBatchEstdistFn = resolve_kernel(
    qg_batch_estdist_avx512, qg_batch_estdist_avx2, qg_batch_estdist_generic
);

void qg_batch_estdist(
    const char* batch_data,
    const BatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance
) {
    kQgBatchEstdistFn(batch_data, q_obj, padded_dim, est_distance);
}

const auto kQgBatchEstdistMaskFn =
    resolve_kernel(qg_batch_estdist_mask_avx2, qg_batch_estdist_mask_generic);

uint32_t qg_batch_estdist_mask(
    const char* batch_data,
    const BatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    float threshold
) {
    return kQgBatchEstdistMaskFn(batch_data, q_obj, padded_dim, est_distance, threshold);
}

const auto kSplitBatchEstdistFn = resolve_kernel(
    split_batch_estdist_avx512, split_batch_estdist_avx2, split_batch_estdist_generic
);

void split_batch_estdist(
    const char* batch_data,
    const SplitBatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    float* low_distance,
    float* ip_x0_qr,
    bool use_hacc
) {
    kSplitBatchEstdistFn(
        batch_data, q_obj, padded_dim, est_distance, low_distance, ip_x0_qr, use_hacc
    );
}

const auto kEuclideanSqrFn =
    resolve_kernel(euclidean_sqr_avx512, euclidean_sqr_avx2, euclidean_sqr_generic);
const auto kDotProductFn =
    resolve_kernel(dot_product_avx512, dot_product_avx2, dot_product_generic);
const auto kDotProductDisFn =
    resolve_kernel(dot_product_dis_avx512, dot_product_dis_avx2, dot_product_dis_generic);
const auto kL2normSqrFn =
    resolve_kernel(l2norm_sqr_avx512, l2norm_sqr_avx2, l2norm_sqr_generic);

float euclidean_sqr(const float* a, const float* b, size_t dim) {
    return kEuclideanSqrFn(a, b, dim);
}

float dot_product(const float* a, const float* b, size_t dim) {
    return kDotProductFn(a, b, dim);
}

float dot_product_dis(const float* a, const float* b, size_t dim) {
    return kDotProductDisFn(a, b, dim);
}

float l2norm_sqr(const float* a, size_t dim) { return kL2normSqrFn(a, dim); }

[[noreturn]] static void missing_feature(const char* feature_name) {
    throw std::runtime_error(
        std::string(feature_name) + " requires AVX2/FMA or AVX512 support"
    );
}

// With zero extra bits there is no extra code to contribute to the inner
// product, so the ex_bits == 0 slot must be a constant-zero stub rather than
// a duplicate of the 1-bit implementation.
static float ip_fxu0(
    const float* /*query*/, const uint8_t* /*compact_code*/, size_t /*dim*/
) {
    return 0.0F;
}

const auto kBestRescaleFactorFn = resolve_kernel(
    best_rescale_factor_avx512, best_rescale_factor_avx2, best_rescale_factor_generic
);

double best_rescale_factor(
    const float* magnitudes, size_t dim, int max_code, double start, double end
) {
    return kBestRescaleFactorFn(magnitudes, dim, max_code, start, end);
}

static void
missing_fht_rotate(const float*, float*, size_t, size_t, size_t, float, const uint8_t*) {
    missing_feature("sign flip");
}

const auto kFhtRotateFn =
    resolve_kernel(fht_rotate_avx512, fht_rotate_avx2, missing_fht_rotate);

void fht_rotate(
    const float* data,
    float* rotated_vec,
    size_t dim,
    size_t padded_dim,
    size_t trunc_dim,
    float fac,
    const uint8_t* flip
) {
    kFhtRotateFn(data, rotated_vec, dim, padded_dim, trunc_dim, fac, flip);
}

static float missing_excode_ip(const float*, const uint8_t*, size_t) {
    missing_feature("excode ip functions");
}

static void missing_flip_sign(const uint8_t*, float*, size_t) {
    missing_feature("sign flip");
}

static void missing_kacs_walk(float*, size_t) { missing_feature("FhtKacRotator"); }

static void missing_scalar_quantize_uint8(uint8_t*, const float*, size_t, float, float) {
    missing_feature("uint8 quantize");
}

static void missing_scalar_quantize_uint16(uint16_t*, const float*, size_t, float, float) {
    missing_feature("uint16 quantize");
}

static void missing_pack_excode(const uint8_t*, uint8_t*, size_t) {
    missing_feature("excode packing");
}

static void missing_new_transpose_bin(const uint16_t*, uint64_t*, size_t, size_t) {
    missing_feature("new transpose bin");
}

static void missing_new_transpose_bin_512(const uint8_t*, uint64_t*, size_t, size_t) {
    missing_feature("new_transpose_bin_512");
}

static float missing_mask_ip_x0_q(const float*, const uint8_t*, size_t) {
    missing_feature("mask ip x0 q");
}

static void missing_fastscan_accumulate(
    const uint8_t* __restrict__, const uint8_t* __restrict__, int32_t* __restrict__, size_t
) {
    missing_feature("fastscan accumulate");
}

static void missing_fastscan_transfer_lut_hacc(const uint16_t*, size_t, uint8_t*) {
    missing_feature("fastscan high-accuracy LUT transfer");
}

static void missing_fastscan_accumulate_hacc(
    const uint8_t* __restrict__, const uint8_t* __restrict__, int32_t*, size_t
) {
    missing_feature("fastscan high-accuracy accumulate");
}

static float missing_warmup_ip_x0_q_512(
    const uint8_t*, const uint64_t*, float, float, size_t, size_t
) {
    missing_feature("warmup_ip_x0_q_512");
}

ExcodeIpTable resolve_excode_ip_table() {
    return resolve_kernel(
        ExcodeIpTable{
            ip_fxu0,
            excode_ipimpl::ip16_fxu1_avx512,
            excode_ipimpl::ip64_fxu2_avx512,
            excode_ipimpl::ip64_fxu3_avx512,
            excode_ipimpl::ip16_fxu4_avx512,
            excode_ipimpl::ip64_fxu5_avx512,
            excode_ipimpl::ip64_fxu6_avx512,
            excode_ipimpl::ip64_fxu7_avx512,
            excode_ipimpl::ip16_fxu8_avx512,
        },
        ExcodeIpTable{
            ip_fxu0,
            excode_ipimpl::ip16_fxu1_avx2,
            excode_ipimpl::ip64_fxu2_avx2,
            excode_ipimpl::ip64_fxu3_avx2,
            excode_ipimpl::ip16_fxu4_avx2,
            excode_ipimpl::ip64_fxu5_avx2,
            excode_ipimpl::ip64_fxu6_avx2,
            excode_ipimpl::ip64_fxu7_avx2,
            excode_ipimpl::ip16_fxu8_avx2,
        },
        ExcodeIpTable{
            ip_fxu0,
            missing_excode_ip,
            missing_excode_ip,
            missing_excode_ip,
            missing_excode_ip,
            missing_excode_ip,
            missing_excode_ip,
            missing_excode_ip,
            missing_excode_ip,
        }
    );
}

const auto kFlipSignFn =
    resolve_kernel(flip_sign_avx512, flip_sign_avx2, missing_flip_sign);

const auto kKacsWalkFn =
    resolve_kernel(kacs_walk_avx512, kacs_walk_avx2, missing_kacs_walk);

const auto kScalarQuantizeUint8Fn = resolve_kernel(
    scalar_quantize_uint8_avx512, scalar_quantize_uint8_avx2, missing_scalar_quantize_uint8
);

const auto kScalarQuantizeUint16Fn = resolve_kernel(
    scalar_quantize_uint16_avx512,
    scalar_quantize_uint16_avx2,
    missing_scalar_quantize_uint16
);

const auto kPacking2BitExcodeFn = resolve_kernel(
    packing_2bit_excode_avx512, packing_2bit_excode_avx2, missing_pack_excode
);
const auto kPacking3BitExcodeFn = resolve_kernel(
    packing_3bit_excode_avx512, packing_3bit_excode_avx2, missing_pack_excode
);
const auto kPacking4BitExcodeFn = resolve_kernel(
    packing_4bit_excode_avx512, packing_4bit_excode_avx2, missing_pack_excode
);
const auto kPacking5BitExcodeFn = resolve_kernel(
    packing_5bit_excode_avx512, packing_5bit_excode_avx2, missing_pack_excode
);
const auto kPacking6BitExcodeFn = resolve_kernel(
    packing_6bit_excode_avx512, packing_6bit_excode_avx2, missing_pack_excode
);
const auto kPacking7BitExcodeFn = resolve_kernel(
    packing_7bit_excode_avx512, packing_7bit_excode_avx2, missing_pack_excode
);

void flip_sign(const uint8_t* flip, float* data, size_t dim) {
    kFlipSignFn(flip, data, dim);
}

void kacs_walk(float* data, size_t len) { kKacsWalkFn(data, len); }

void scalar_quantize_uint8(
    uint8_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    kScalarQuantizeUint8Fn(result, vec0, dim, lo, delta);
}

void scalar_quantize_uint16(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    kScalarQuantizeUint16Fn(result, vec0, dim, lo, delta);
}

void packing_2bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    kPacking2BitExcodeFn(o_raw, o_compact, dim);
}

void packing_3bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    kPacking3BitExcodeFn(o_raw, o_compact, dim);
}

void packing_4bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    kPacking4BitExcodeFn(o_raw, o_compact, dim);
}

void packing_5bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    kPacking5BitExcodeFn(o_raw, o_compact, dim);
}

void packing_6bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    kPacking6BitExcodeFn(o_raw, o_compact, dim);
}

void packing_7bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    kPacking7BitExcodeFn(o_raw, o_compact, dim);
}

}  // namespace rabitqlib::simd

namespace rabitqlib {

const simd::ExcodeIpTable kExcodeIpTable = simd::resolve_excode_ip_table();

const ex_ipfunc kIp16Fxu1AvxFn = kExcodeIpTable[1];
const ex_ipfunc kIp64Fxu2AvxFn = kExcodeIpTable[2];
const ex_ipfunc kIp64Fxu3AvxFn = kExcodeIpTable[3];
const ex_ipfunc kIp16Fxu4AvxFn = kExcodeIpTable[4];
const ex_ipfunc kIp64Fxu5AvxFn = kExcodeIpTable[5];
const ex_ipfunc kIp64Fxu6AvxFn = kExcodeIpTable[6];
const ex_ipfunc kIp64Fxu7AvxFn = kExcodeIpTable[7];

const auto kNewTransposeBinFn = rabitqlib::simd::resolve_kernel(
    simd::new_transpose_bin_avx512,
    simd::new_transpose_bin_avx2,
    simd::missing_new_transpose_bin
);

const auto kNewTransposeBin512Fn = rabitqlib::simd::resolve_kernel(
    simd::new_transpose_bin_512_avx512,
    simd::new_transpose_bin_512_avx2,
    simd::missing_new_transpose_bin_512
);

using MaskIpX0QFn = float (*)(const float*, const uint8_t*, size_t);
const MaskIpX0QFn kMaskIpX0QFn = rabitqlib::simd::resolve_kernel<MaskIpX0QFn>(
    static_cast<MaskIpX0QFn>(simd::mask_ip_x0_q_avx512),
    static_cast<MaskIpX0QFn>(simd::mask_ip_x0_q_avx2),
    simd::missing_mask_ip_x0_q
);

ex_ipfunc select_excode_ipfunc(size_t ex_bits) {
    if (ex_bits <= 8) {
        return kExcodeIpTable[ex_bits];
    }

    throw std::invalid_argument("Bad IP function for IVF");
}

float excode_ipimpl::ip16_fxu1_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp16Fxu1AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu2_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu2AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu3_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu3AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip16_fxu4_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp16Fxu4AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu5_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu5AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu6_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu6AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu7_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu7AvxFn(query, compact_code, dim);
}

void new_transpose_bin(const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query) {
    kNewTransposeBinFn(q, tq, padded_dim, b_query);
}

void new_transpose_bin_512(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    kNewTransposeBin512Fn(q, tq, padded_dim, b_query);
}

float mask_ip_x0_q(const float* query, const uint8_t* data, size_t padded_dim) {
    return kMaskIpX0QFn(query, data, padded_dim);
}

float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim) {
    return mask_ip_x0_q(query, reinterpret_cast<const uint8_t*>(data), padded_dim);
}

}  // namespace rabitqlib

namespace rabitqlib::fastscan {

const auto kPackLutFn = rabitqlib::simd::resolve_kernel(
    simd::pack_lut_avx512, simd::pack_lut_avx2, simd::pack_lut_generic
);

template <>
void pack_lut<float>(size_t dim, const float* __restrict__ query, float* __restrict__ lut) {
    kPackLutFn(dim, query, lut);
}

const auto kAccumulateFn = rabitqlib::simd::resolve_kernel(
    simd::accumulate_avx512,
    simd::accumulate_avx2,
    rabitqlib::simd::missing_fastscan_accumulate
);

const auto kTransferLutHaccFn = rabitqlib::simd::resolve_kernel(
    simd::transfer_lut_hacc_avx512,
    simd::transfer_lut_hacc_avx2,
    rabitqlib::simd::missing_fastscan_transfer_lut_hacc
);

const auto kAccumulateHaccFn = rabitqlib::simd::resolve_kernel(
    simd::accumulate_hacc_avx512,
    simd::accumulate_hacc_avx2,
    rabitqlib::simd::missing_fastscan_accumulate_hacc
);

void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    int32_t* __restrict__ result,
    size_t dim
) {
    if (dim == 0 || dim % 16 != 0) {
        throw std::invalid_argument("FastScan dimension must be a positive multiple of 16");
    }
    kAccumulateFn(codes, lp_table, result, dim);
}

void transfer_lut_hacc(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
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
    if (dim == 0 || dim % 16 != 0) {
        throw std::invalid_argument(
            "high-accuracy FastScan dimension must be a positive multiple of 16"
        );
    }
    kAccumulateHaccFn(codes, hc_lut, accu_res, dim);
}

}  // namespace rabitqlib::fastscan

namespace rabitqlib {

using WarmupIpX0Q512Fn =
    float (*)(const uint8_t*, const uint64_t*, float, float, size_t, size_t);
const WarmupIpX0Q512Fn kWarmupIpX0Q512Fn =
    rabitqlib::simd::resolve_kernel<WarmupIpX0Q512Fn>(
        static_cast<WarmupIpX0Q512Fn>(rabitqlib::simd::warmup_ip_x0_q_512_avx512),
        static_cast<WarmupIpX0Q512Fn>(rabitqlib::simd::warmup_ip_x0_q_512_avx2),
        rabitqlib::simd::missing_warmup_ip_x0_q_512,
        cpu::has_avx512_popcnt()
    );

float warmup_ip_x0_q_512(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    return kWarmupIpX0Q512Fn(data, query, delta, vl, padded_dim, b_query);
}

float warmup_ip_x0_q_512(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    return warmup_ip_x0_q_512(
        reinterpret_cast<const uint8_t*>(data), query, delta, vl, padded_dim, b_query
    );
}

}  // namespace rabitqlib

namespace rabitqlib::hnsw::detail {
namespace {
using SearchKnnFn =
    std::priority_queue<std::pair<float, PID>> (*)(HierarchicalNSW&, const float*, size_t);
std::priority_queue<std::pair<float, PID>> missing_search_knn(
    HierarchicalNSW&, const float*, size_t
) {
    throw std::runtime_error("HNSW search requires AVX2/FMA or AVX512 support");
}
// The core variant uses AVX2 warmup; the popcount variant has its own stricter tier.
const SearchKnnFn kSearchKnnFn = cpu::has_avx512_popcnt()
                                     ? search_knn_avx512_popcnt
                                     : rabitqlib::simd::resolve_kernel(
                                           search_knn_avx512_core,
                                           search_knn_avx2,
                                           missing_search_knn,
                                           cpu::has_avx512_core() && cpu::has_avx2()
                                       );
}  // namespace
std::priority_queue<std::pair<float, PID>> search_knn(
    HierarchicalNSW& index, const float* query, size_t topk
) {
    return kSearchKnnFn(index, query, topk);
}
}  // namespace rabitqlib::hnsw::detail
