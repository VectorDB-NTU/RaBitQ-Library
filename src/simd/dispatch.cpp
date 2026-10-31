#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "dispatch_resolver.hpp"
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
    const float* data,
    const size_t* point_ids,
    size_t count,
    size_t dim,
    const uint32_t* assignments,
    double* sums,
    size_t* counts
) {
    static const auto kAccumulateClusterSumsFn = RABITQ_RESOLVE(
        accumulate_cluster_sums_avx512,
        accumulate_cluster_sums_avx2,
        accumulate_cluster_sums_generic
    );

    kAccumulateClusterSumsFn(data, point_ids, count, dim, assignments, sums, counts);
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
    static const auto kEuclideanSqrFn = RABITQ_RESOLVE_NEON(
        euclidean_sqr_neon, euclidean_sqr_avx512, euclidean_sqr_avx2, euclidean_sqr_generic
    );

    return kEuclideanSqrFn(a, b, dim);
}

float dot_product(const float* a, const float* b, size_t dim) {
    static const auto kDotProductFn = RABITQ_RESOLVE_NEON(
        dot_product_neon, dot_product_avx512, dot_product_avx2, dot_product_generic
    );

    return kDotProductFn(a, b, dim);
}

float dot_product_dis(const float* a, const float* b, size_t dim) {
    static const auto kDotProductDisFn = RABITQ_RESOLVE_NEON(
        dot_product_dis_neon,
        dot_product_dis_avx512,
        dot_product_dis_avx2,
        dot_product_dis_generic
    );

    return kDotProductDisFn(a, b, dim);
}

float l2norm_sqr(const float* a, size_t dim) {
    static const auto kL2normSqrFn = RABITQ_RESOLVE_NEON(
        l2norm_sqr_neon, l2norm_sqr_avx512, l2norm_sqr_avx2, l2norm_sqr_generic
    );

    return kL2normSqrFn(a, dim);
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
    static const auto kBestRescaleFactorWithWorkspaceFn = RABITQ_RESOLVE(
        best_rescale_factor_with_workspace_avx512,
        best_rescale_factor_with_workspace_avx2,
        best_rescale_factor_with_workspace_generic
    );

    return kBestRescaleFactorWithWorkspaceFn(
        magnitudes, dim, max_code, start, end, promoted, reciprocals
    );
}

void fht_rotate(
    const float* data,
    float* rotated_data,
    size_t dim,
    size_t padded_dim,
    size_t trunc_dim,
    float scale_factor,
    const uint8_t* flip
) {
    static const auto kFhtRotateFn = RABITQ_RESOLVE_NEON(
        fht_rotate_neon, fht_rotate_avx512, fht_rotate_avx2, fht_rotate_generic
    );

    kFhtRotateFn(data, rotated_data, dim, padded_dim, trunc_dim, scale_factor, flip);
}

void flip_sign(const uint8_t* flip, float* data, size_t dim) {
    static const auto kFlipSignFn = RABITQ_RESOLVE_NEON(
        flip_sign_neon, flip_sign_avx512, flip_sign_avx2, flip_sign_generic
    );

    kFlipSignFn(flip, data, dim);
}

void kacs_walk(float* data, size_t len) {
    static const auto kKacsWalkFn = RABITQ_RESOLVE_NEON(
        kacs_walk_neon, kacs_walk_avx512, kacs_walk_avx2, kacs_walk_generic
    );

    kKacsWalkFn(data, len);
}

void scalar_quantize_uint8(
    uint8_t* result, const float* data, size_t dim, float lo, float delta
) {
    static const auto kScalarQuantizeUint8Fn = RABITQ_RESOLVE_NEON(
        scalar_quantize_uint8_neon,
        scalar_quantize_uint8_avx512,
        scalar_quantize_uint8_avx2,
        scalar_quantize_uint8_generic
    );

    kScalarQuantizeUint8Fn(result, data, dim, lo, delta);
}

void scalar_quantize_uint16(
    uint16_t* result, const float* data, size_t dim, float lo, float delta
) {
    static const auto kScalarQuantizeUint16Fn = RABITQ_RESOLVE_NEON(
        scalar_quantize_uint16_neon,
        scalar_quantize_uint16_avx512,
        scalar_quantize_uint16_avx2,
        scalar_quantize_uint16_generic
    );

    kScalarQuantizeUint16Fn(result, data, dim, lo, delta);
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

namespace {
// ex_bits is in [0, kMaxExBits]; the table holds one kernel per value.
constexpr size_t kMaxExBits = 8;
using ExcodeIpTable = std::array<ex_ipfunc, kMaxExBits + 1>;

// With zero extra bits there is no extra code to contribute to the inner
// product, so the ex_bits == 0 slot must be a constant-zero stub rather than
// a duplicate of the 1-bit implementation.
float ip_fxu0(const float* /*query*/, const uint8_t* /*compact_code*/, size_t /*dim*/) {
    return 0.0F;
}

const ExcodeIpTable& excode_ip_table() {
    static const auto kExcodeIpTable = RABITQ_RESOLVE_NEON(
        (ExcodeIpTable{
            ip_fxu0,
            excode_ipimpl::ip16_fxu1_neon,
            excode_ipimpl::ip64_fxu2_neon,
            excode_ipimpl::ip64_fxu3_neon,
            excode_ipimpl::ip16_fxu4_neon,
            excode_ipimpl::ip64_fxu5_neon,
            excode_ipimpl::ip64_fxu6_neon,
            excode_ipimpl::ip64_fxu7_neon,
            excode_ipimpl::ip16_fxu8_neon,
        }),
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
        (ExcodeIpTable{
            ip_fxu0,
            excode_ipimpl::ip16_fxu1_generic,
            excode_ipimpl::ip64_fxu2_generic,
            excode_ipimpl::ip64_fxu3_generic,
            excode_ipimpl::ip16_fxu4_generic,
            excode_ipimpl::ip64_fxu5_generic,
            excode_ipimpl::ip64_fxu6_generic,
            excode_ipimpl::ip64_fxu7_generic,
            excode_ipimpl::ip16_fxu8_generic,
        })
    );

    return kExcodeIpTable;
}
}  // namespace

}  // namespace rabitqlib::simd

namespace rabitqlib {

using MaskIpX0QFn = float (*)(const float*, const uint8_t*, size_t);
using WarmupIpX0Q512Fn =
    float (*)(const uint8_t*, const uint64_t*, float, float, size_t, size_t);

ex_ipfunc select_excode_ipfunc(size_t ex_bits) {
    if (ex_bits <= simd::kMaxExBits) {
        return simd::excode_ip_table()[ex_bits];
    }

    throw std::invalid_argument("ex_bits must be in [0, 8]");
}

float excode_ipimpl::ip16_fxu1_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return simd::excode_ip_table()[1](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu2_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return simd::excode_ip_table()[2](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu3_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return simd::excode_ip_table()[3](query, compact_code, dim);
}

float excode_ipimpl::ip16_fxu4_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return simd::excode_ip_table()[4](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu5_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return simd::excode_ip_table()[5](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu6_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return simd::excode_ip_table()[6](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu7_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return simd::excode_ip_table()[7](query, compact_code, dim);
}

void new_transpose_bin(
    const uint16_t* query, uint64_t* transposed_query, size_t padded_dim, size_t b_query
) {
    static const auto kNewTransposeBinFn = RABITQ_RESOLVE_NEON(
        simd::new_transpose_bin_neon,
        simd::new_transpose_bin_avx512,
        simd::new_transpose_bin_avx2,
        simd::new_transpose_bin_generic
    );

    kNewTransposeBinFn(query, transposed_query, padded_dim, b_query);
}

void new_transpose_bin_512(
    const uint8_t* query, uint64_t* transposed_query, size_t padded_dim, size_t b_query
) {
    static const auto kNewTransposeBin512Fn = RABITQ_RESOLVE_NEON(
        simd::new_transpose_bin_512_neon,
        simd::new_transpose_bin_512_avx512,
        simd::new_transpose_bin_512_avx2,
        simd::new_transpose_bin_512_generic
    );

    kNewTransposeBin512Fn(query, transposed_query, padded_dim, b_query);
}

float mask_ip_x0_q(const float* query, const uint8_t* data, size_t padded_dim) {
    static const auto kMaskIpX0QFn = RABITQ_RESOLVE_NEON(
        static_cast<MaskIpX0QFn>(simd::mask_ip_x0_q_neon),
        static_cast<MaskIpX0QFn>(simd::mask_ip_x0_q_avx512),
        static_cast<MaskIpX0QFn>(simd::mask_ip_x0_q_avx2),
        static_cast<MaskIpX0QFn>(simd::mask_ip_x0_q_generic)
    );

    return kMaskIpX0QFn(query, data, padded_dim);
}

float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim) {
    return mask_ip_x0_q(query, reinterpret_cast<const uint8_t*>(data), padded_dim);
}

float warmup_ip_x0_q_512(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    static const auto kWarmupIpX0Q512Fn = RABITQ_RESOLVE_NEON(
        static_cast<WarmupIpX0Q512Fn>(simd::warmup_ip_x0_q_512_neon),
        static_cast<WarmupIpX0Q512Fn>(simd::warmup_ip_x0_q_512_avx512),
        static_cast<WarmupIpX0Q512Fn>(simd::warmup_ip_x0_q_512_avx2),
        static_cast<WarmupIpX0Q512Fn>(simd::warmup_ip_x0_q_512_generic),
        cpu::has_avx512_popcnt()
    );

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
