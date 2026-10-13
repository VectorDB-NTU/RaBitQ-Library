#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "dispatch_resolver.hpp"
#include "excode_dispatch.hpp"
#include "rabitqlib/core/detail/simd/matrix_dispatch.hpp"
#include "rabitqlib/core/detail/simd/pack_excode_dispatch.hpp"
#include "rabitqlib/core/detail/simd/quantization_dispatch.hpp"
#include "rabitqlib/core/detail/simd/rotator_dispatch.hpp"
#include "rabitqlib/core/detail/simd/space_dispatch.hpp"
#include "rabitqlib/core/detail/simd/warmup_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/warmup_space.hpp"

namespace rabitqlib::simd {

void accumulate_cluster_sums(
    const float* x,
    const size_t* point_ids,
    size_t count,
    size_t dim,
    const uint32_t* labels,
    double* sums,
    size_t* counts
) {
    static const auto kAccumulateClusterSumsFn = RABITQ_RESOLVE(
        accumulate_cluster_sums_avx512,
        accumulate_cluster_sums_avx2,
        accumulate_cluster_sums_generic
    );

    kAccumulateClusterSumsFn(x, point_ids, count, dim, labels, sums, counts);
}

void matrix_product(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
) {
    static const auto kMatrixProductFn =
        RABITQ_RESOLVE(matrix_product_avx512, matrix_product_avx2, matrix_product_generic);

    kMatrixProductFn(left, right, result, rows, inner, cols);
}

void matrix_product_transposed(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
) {
    static const auto kMatrixProductTransposedFn = RABITQ_RESOLVE(
        matrix_product_transposed_avx512,
        matrix_product_transposed_avx2,
        matrix_product_transposed_generic
    );

    kMatrixProductTransposedFn(left, right, result, rows, inner, cols);
}

void row_norms(const float* data, float* result, size_t rows, size_t dim) {
    static const auto kRowNormsFn =
        RABITQ_RESOLVE(row_norms_avx512, row_norms_avx2, row_norms_generic);

    kRowNormsFn(data, result, rows, dim);
}

void pairwise_distances_lower(
    const float* data,
    float* result,
    float* norms_data,
    size_t size,
    size_t dim,
    bool inner_product
) {
    static const auto kPairwiseDistancesLowerFn = RABITQ_RESOLVE(
        pairwise_distances_lower_avx512,
        pairwise_distances_lower_avx2,
        pairwise_distances_lower_generic
    );

    kPairwiseDistancesLowerFn(data, result, norms_data, size, dim, inner_product);
}

float euclidean_sqr(const float* a, const float* b, size_t dim) {
#if defined(__aarch64__)
    static const auto kEuclideanSqrFn =
        resolve_optional_kernel(euclidean_sqr_neon, euclidean_sqr_generic, cpu::has_neon());
#else
    static const auto kEuclideanSqrFn =
        RABITQ_RESOLVE(euclidean_sqr_avx512, euclidean_sqr_avx2, euclidean_sqr_generic);
#endif

    return kEuclideanSqrFn(a, b, dim);
}

float dot_product(const float* a, const float* b, size_t dim) {
#if defined(__aarch64__)
    static const auto kDotProductFn =
        resolve_optional_kernel(dot_product_neon, dot_product_generic, cpu::has_neon());
#else
    static const auto kDotProductFn =
        RABITQ_RESOLVE(dot_product_avx512, dot_product_avx2, dot_product_generic);
#endif

    return kDotProductFn(a, b, dim);
}

float dot_product_dis(const float* a, const float* b, size_t dim) {
#if defined(__aarch64__)
    static const auto kDotProductDisFn = resolve_optional_kernel(
        dot_product_dis_neon, dot_product_dis_generic, cpu::has_neon()
    );
#else
    static const auto kDotProductDisFn = RABITQ_RESOLVE(
        dot_product_dis_avx512, dot_product_dis_avx2, dot_product_dis_generic
    );
#endif

    return kDotProductDisFn(a, b, dim);
}

float l2norm_sqr(const float* a, size_t dim) {
#if defined(__aarch64__)
    static const auto kL2normSqrFn =
        resolve_optional_kernel(l2norm_sqr_neon, l2norm_sqr_generic, cpu::has_neon());
#else
    static const auto kL2normSqrFn =
        RABITQ_RESOLVE(l2norm_sqr_avx512, l2norm_sqr_avx2, l2norm_sqr_generic);
#endif
    return kL2normSqrFn(a, dim);
}

// With zero extra bits there is no extra code to contribute to the inner
// product, so the ex_bits == 0 slot must be a constant-zero stub rather than
// a duplicate of the 1-bit implementation.
static float ip_fxu0(
    const float* /*query*/, const uint8_t* /*compact_code*/, size_t /*dim*/
) {
    return 0.0F;
}

double best_rescale_factor(
    const float* magnitudes, size_t dim, int max_code, double start, double end
) {
    static const auto kBestRescaleFactorFn = RABITQ_RESOLVE(
        best_rescale_factor_avx512, best_rescale_factor_avx2, best_rescale_factor_generic
    );

    return kBestRescaleFactorFn(magnitudes, dim, max_code, start, end);
}

double best_rescale_factor_with_workspace(
    const float* magnitudes,
    size_t dim,
    int max_code,
    double start,
    double end,
    double* promoted,
    double* reciprocals
) {
    static const auto kBestRescaleWorkspaceFn = RABITQ_RESOLVE(
        best_rescale_factor_with_workspace_avx512,
        best_rescale_factor_with_workspace_avx2,
        best_rescale_factor_with_workspace_generic
    );

    return kBestRescaleWorkspaceFn(
        magnitudes, dim, max_code, start, end, promoted, reciprocals
    );
}

void fht_rotate(
    const float* data,
    float* rotated_vec,
    size_t dim,
    size_t padded_dim,
    size_t trunc_dim,
    float fac,
    const uint8_t* flip
) {
#if defined(__aarch64__) || defined(_M_ARM64)
    static const auto kFhtRotateFn =
        resolve_optional_kernel(fht_rotate_neon, fht_rotate_generic, cpu::has_neon());
#else
    static const auto kFhtRotateFn =
        RABITQ_RESOLVE(fht_rotate_avx512, fht_rotate_avx2, fht_rotate_generic);
#endif

    kFhtRotateFn(data, rotated_vec, dim, padded_dim, trunc_dim, fac, flip);
}

ExcodeIpTable resolve_excode_ip_table() {
    const auto generic = ExcodeIpTable{
        ip_fxu0,
        excode_ipimpl::ip16_fxu1_generic,
        excode_ipimpl::ip64_fxu2_generic,
        excode_ipimpl::ip64_fxu3_generic,
        excode_ipimpl::ip16_fxu4_generic,
        excode_ipimpl::ip64_fxu5_generic,
        excode_ipimpl::ip64_fxu6_generic,
        excode_ipimpl::ip64_fxu7_generic,
        excode_ipimpl::ip16_fxu8_generic,
    };
#if defined(__aarch64__) || defined(_M_ARM64)
    return resolve_optional_kernel(
        ExcodeIpTable{
            ip_fxu0,
            excode_ipimpl::ip16_fxu1_neon,
            excode_ipimpl::ip64_fxu2_neon,
            excode_ipimpl::ip64_fxu3_neon,
            excode_ipimpl::ip16_fxu4_neon,
            excode_ipimpl::ip64_fxu5_neon,
            excode_ipimpl::ip64_fxu6_neon,
            excode_ipimpl::ip64_fxu7_neon,
            excode_ipimpl::ip16_fxu8_neon},
        generic,
        cpu::has_neon()
    );
#else
    return RABITQ_RESOLVE(
        (ExcodeIpTable{
            ip_fxu0,
            excode_ipimpl::ip16_fxu1_avx512,
            excode_ipimpl::ip64_fxu2_avx512,
            excode_ipimpl::ip64_fxu3_avx512,
            excode_ipimpl::ip16_fxu4_avx512,
            excode_ipimpl::ip64_fxu5_avx512,
            excode_ipimpl::ip64_fxu6_avx512,
            excode_ipimpl::ip64_fxu7_avx512,
            excode_ipimpl::ip16_fxu8_avx512,
        }),
        (ExcodeIpTable{
            ip_fxu0,
            excode_ipimpl::ip16_fxu1_avx2,
            excode_ipimpl::ip64_fxu2_avx2,
            excode_ipimpl::ip64_fxu3_avx2,
            excode_ipimpl::ip16_fxu4_avx2,
            excode_ipimpl::ip64_fxu5_avx2,
            excode_ipimpl::ip64_fxu6_avx2,
            excode_ipimpl::ip64_fxu7_avx2,
            excode_ipimpl::ip16_fxu8_avx2,
        }),
        generic
    );
#endif
}

void flip_sign(const uint8_t* flip, float* data, size_t dim) {
#if defined(__aarch64__) || defined(_M_ARM64)
    static const auto kFlipSignFn =
        resolve_optional_kernel(flip_sign_neon, flip_sign_generic, cpu::has_neon());
#else
    static const auto kFlipSignFn =
        RABITQ_RESOLVE(flip_sign_avx512, flip_sign_avx2, flip_sign_generic);
#endif

    kFlipSignFn(flip, data, dim);
}

void kacs_walk(float* data, size_t len) {
#if defined(__aarch64__) || defined(_M_ARM64)
    static const auto kKacsWalkFn =
        resolve_optional_kernel(kacs_walk_neon, kacs_walk_generic, cpu::has_neon());
#else
    static const auto kKacsWalkFn =
        RABITQ_RESOLVE(kacs_walk_avx512, kacs_walk_avx2, kacs_walk_generic);
#endif
    kKacsWalkFn(data, len);
}

void scalar_quantize_uint8(
    uint8_t* result, const float* vec0, size_t dim, float lo, float delta
) {
#if defined(__aarch64__) || defined(_M_ARM64)
    static const auto kScalarQuantizeUint8Fn = resolve_optional_kernel(
        scalar_quantize_uint8_neon, scalar_quantize_uint8_generic, cpu::has_neon()
    );
#else
    static const auto kScalarQuantizeUint8Fn = RABITQ_RESOLVE(
        scalar_quantize_uint8_avx512,
        scalar_quantize_uint8_avx2,
        scalar_quantize_uint8_generic
    );
#endif

    kScalarQuantizeUint8Fn(result, vec0, dim, lo, delta);
}

void scalar_quantize_uint16(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
) {
#if defined(__aarch64__) || defined(_M_ARM64)
    static const auto kScalarQuantizeUint16Fn = resolve_optional_kernel(
        scalar_quantize_uint16_neon, scalar_quantize_uint16_generic, cpu::has_neon()
    );
#else
    static const auto kScalarQuantizeUint16Fn = RABITQ_RESOLVE(
        scalar_quantize_uint16_avx512,
        scalar_quantize_uint16_avx2,
        scalar_quantize_uint16_generic
    );
#endif

    kScalarQuantizeUint16Fn(result, vec0, dim, lo, delta);
}

void packing_2bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    static const auto kPacking2BitExcodeFn = RABITQ_RESOLVE(
        packing_2bit_excode_avx512, packing_2bit_excode_avx2, packing_2bit_excode_generic
    );

    kPacking2BitExcodeFn(o_raw, o_compact, dim);
}

void packing_3bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    static const auto kPacking3BitExcodeFn = RABITQ_RESOLVE(
        packing_3bit_excode_avx512, packing_3bit_excode_avx2, packing_3bit_excode_generic
    );

    kPacking3BitExcodeFn(o_raw, o_compact, dim);
}

void packing_4bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    static const auto kPacking4BitExcodeFn = RABITQ_RESOLVE(
        packing_4bit_excode_avx512, packing_4bit_excode_avx2, packing_4bit_excode_generic
    );

    kPacking4BitExcodeFn(o_raw, o_compact, dim);
}

void packing_5bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    static const auto kPacking5BitExcodeFn = RABITQ_RESOLVE(
        packing_5bit_excode_avx512, packing_5bit_excode_avx2, packing_5bit_excode_generic
    );

    kPacking5BitExcodeFn(o_raw, o_compact, dim);
}

void packing_6bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    static const auto kPacking6BitExcodeFn = RABITQ_RESOLVE(
        packing_6bit_excode_avx512, packing_6bit_excode_avx2, packing_6bit_excode_generic
    );

    kPacking6BitExcodeFn(o_raw, o_compact, dim);
}

void packing_7bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    static const auto kPacking7BitExcodeFn = RABITQ_RESOLVE(
        packing_7bit_excode_avx512, packing_7bit_excode_avx2, packing_7bit_excode_generic
    );

    kPacking7BitExcodeFn(o_raw, o_compact, dim);
}

}  // namespace rabitqlib::simd

namespace rabitqlib {

static const simd::ExcodeIpTable& excode_ip_table() {
    static const auto table = simd::resolve_excode_ip_table();
    return table;
}

using MaskIpX0QFn = float (*)(const float*, const uint8_t*, size_t);

ex_ipfunc select_excode_ipfunc(size_t ex_bits) {
    if (ex_bits <= 8) {
        return excode_ip_table()[ex_bits];
    }

    throw std::invalid_argument("Bad IP function for IVF");
}

float excode_ipimpl::ip16_fxu1_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return excode_ip_table()[1](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu2_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return excode_ip_table()[2](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu3_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return excode_ip_table()[3](query, compact_code, dim);
}

float excode_ipimpl::ip16_fxu4_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return excode_ip_table()[4](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu5_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return excode_ip_table()[5](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu6_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return excode_ip_table()[6](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu7_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return excode_ip_table()[7](query, compact_code, dim);
}

void new_transpose_bin(const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query) {
#if defined(__aarch64__) || defined(_M_ARM64)
    static const auto kNewTransposeBinFn = simd::resolve_optional_kernel(
        simd::new_transpose_bin_neon, simd::new_transpose_bin_generic, cpu::has_neon()
    );
#else
    static const auto kNewTransposeBinFn = RABITQ_RESOLVE(
        simd::new_transpose_bin_avx512,
        simd::new_transpose_bin_avx2,
        simd::new_transpose_bin_generic
    );
#endif

    kNewTransposeBinFn(q, tq, padded_dim, b_query);
}

void new_transpose_bin_512(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
#if defined(__aarch64__) || defined(_M_ARM64)
    static const auto kNewTransposeBin512Fn = simd::resolve_optional_kernel(
        simd::new_transpose_bin_512_neon,
        simd::new_transpose_bin_512_generic,
        cpu::has_neon()
    );
#else
    static const auto kNewTransposeBin512Fn = RABITQ_RESOLVE(
        simd::new_transpose_bin_512_avx512,
        simd::new_transpose_bin_512_avx2,
        simd::new_transpose_bin_512_generic
    );
#endif

    kNewTransposeBin512Fn(q, tq, padded_dim, b_query);
}

float mask_ip_x0_q(const float* query, const uint8_t* data, size_t padded_dim) {
#if defined(__aarch64__) || defined(_M_ARM64)
    static const MaskIpX0QFn kMaskIpX0QFn = simd::resolve_optional_kernel(
        static_cast<MaskIpX0QFn>(simd::mask_ip_x0_q_neon),
        static_cast<MaskIpX0QFn>(simd::mask_ip_x0_q_generic),
        cpu::has_neon()
    );
#else
    static const MaskIpX0QFn kMaskIpX0QFn = RABITQ_RESOLVE(
        static_cast<MaskIpX0QFn>(simd::mask_ip_x0_q_avx512),
        static_cast<MaskIpX0QFn>(simd::mask_ip_x0_q_avx2),
        static_cast<MaskIpX0QFn>(simd::mask_ip_x0_q_generic)
    );
#endif

    return kMaskIpX0QFn(query, data, padded_dim);
}

float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim) {
    return mask_ip_x0_q(query, reinterpret_cast<const uint8_t*>(data), padded_dim);
}

}  // namespace rabitqlib

namespace rabitqlib {

using WarmupIpX0Q512Fn =
    float (*)(const uint8_t*, const uint64_t*, float, float, size_t, size_t);

float warmup_ip_x0_q_512(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
#if defined(__aarch64__) || defined(_M_ARM64)
    static const WarmupIpX0Q512Fn kWarmupIpX0Q512Fn = simd::resolve_optional_kernel(
        static_cast<WarmupIpX0Q512Fn>(simd::warmup_ip_x0_q_512_neon),
        static_cast<WarmupIpX0Q512Fn>(simd::warmup_ip_x0_q_512_generic),
        cpu::has_neon()
    );
#else
    static const WarmupIpX0Q512Fn kWarmupIpX0Q512Fn = RABITQ_RESOLVE(
        static_cast<WarmupIpX0Q512Fn>(rabitqlib::simd::warmup_ip_x0_q_512_avx512),
        static_cast<WarmupIpX0Q512Fn>(rabitqlib::simd::warmup_ip_x0_q_512_avx2),
        static_cast<WarmupIpX0Q512Fn>(rabitqlib::simd::warmup_ip_x0_q_512_generic),
        cpu::has_avx512_popcnt()
    );
#endif

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
