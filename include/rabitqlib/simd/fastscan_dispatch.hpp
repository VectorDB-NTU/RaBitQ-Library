#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::fastscan::simd {

// NEON uses the portable/AVX2 high-accuracy LUT byte order.
void accumulate_neon(const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim);
void accumulate_hacc_neon(
    const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim
);

void pack_lut_neon(size_t dim, const float* query, float* lut);
void pack_lut_generic(size_t dim, const float* query, float* lut);
void pack_lut_avx2(size_t dim, const float* query, float* lut);
void pack_lut_avx512(size_t dim, const float* query, float* lut);

// Scalar reference for backend correctness tests; never selected by runtime dispatch.
void accumulate_generic(
    const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim
);
void accumulate_unsupported(
    const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim
);
void accumulate_avx2(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    int32_t* __restrict__ result,
    size_t dim
);
void transfer_lut_hacc_generic(const uint16_t* lut, size_t dim, uint8_t* hc_lut);
void transfer_lut_hacc_avx2(const uint16_t* lut, size_t dim, uint8_t* hc_lut);
void accumulate_hacc_generic(
    const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim
);
void accumulate_hacc_avx2(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
);

void accumulate_avx512(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    int32_t* __restrict__ result,
    size_t dim
);
void transfer_lut_hacc_avx512(const uint16_t* lut, size_t dim, uint8_t* hc_lut);
void accumulate_hacc_avx512(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
);

}  // namespace rabitqlib::fastscan::simd
