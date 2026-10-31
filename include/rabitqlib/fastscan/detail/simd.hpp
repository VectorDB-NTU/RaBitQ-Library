#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::fastscan::simd {

// NEON uses the portable/AVX2 high-accuracy LUT byte order.
void accumulate_neon(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* __restrict__ accumulated_values,
    size_t dim
);
void accumulate_hacc_neon(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* accumulated_values,
    size_t dim
);

void pack_lut_neon(size_t dim, const float* query, float* lut);
void pack_lut_generic(size_t dim, const float* query, float* lut);
void pack_lut_avx2(size_t dim, const float* query, float* lut);
void pack_lut_avx512(size_t dim, const float* query, float* lut);

// Raw fused kernels; the wrapper preserves exceptional/zero-bound reduction behavior.
void pack_lut_range_generic(
    size_t dim, const float* query, float* lut, float& minimum, float& maximum
);
void pack_lut_range_avx2(
    size_t dim, const float* query, float* lut, float& minimum, float& maximum
);
void pack_lut_range_avx512(
    size_t dim, const float* query, float* lut, float& minimum, float& maximum
);
void pack_lut_range_neon(
    size_t dim, const float* query, float* lut, float& minimum, float& maximum
);

// Scalar reference for backend correctness tests; never selected by runtime dispatch.
void accumulate_generic(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* __restrict__ accumulated_values,
    size_t dim
);
void accumulate_unsupported(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* __restrict__ accumulated_values,
    size_t dim
);
void accumulate_avx2(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* __restrict__ accumulated_values,
    size_t dim
);
void transfer_lut_hacc_generic(const uint16_t* lut, size_t dim, uint8_t* hc_lut);
void accumulate_hacc_generic(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* accumulated_values,
    size_t dim
);
void accumulate_hacc_avx2(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* accumulated_values,
    size_t dim
);

void accumulate_avx512(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* __restrict__ accumulated_values,
    size_t dim
);
void transfer_lut_hacc_avx512(const uint16_t* lut, size_t dim, uint8_t* hc_lut);
void accumulate_hacc_avx512(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* accumulated_values,
    size_t dim
);

}  // namespace rabitqlib::fastscan::simd
