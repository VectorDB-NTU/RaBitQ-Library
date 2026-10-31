// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace rabitqlib::fastscan {

constexpr static std::array<int, 16> kPos = {
    3 /*0000*/,
    3 /*0001*/,
    2 /*0010*/,
    3 /*0011*/,
    1 /*0100*/,
    3 /*0101*/,
    2 /*0110*/,
    3 /*0111*/,
    0 /*1000*/,
    3 /*1001*/,
    2 /*1010*/,
    3 /*1011*/,
    1 /*1100*/,
    3 /*1101*/,
    2 /*1110*/,
    3 /*1111*/,
};  // kPos[j]: query coordinate of j's lowest set bit (bit 3 is q0, bit 0 is q3)

// Accumulate one batch into int32_t results. Throws if a result exceeds int32_t.
// dim must be a positive multiple of 16.
// Requires AVX2/FMA, AVX-512, or ARM NEON; otherwise throws std::runtime_error.
void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* __restrict__ result,
    size_t dim
);

// pack lookup table for fastscan, for each 4 dim, we have 16 (2^4) different results
// ! dim % 4 == 0
template <typename T>
inline void pack_lut(size_t dim, const T* __restrict__ query, T* __restrict__ lut) {
    size_t num_codebook = dim >> 2;
    // Preserve +0 when the first selected coordinate is -0.
    volatile T zero = 0;
    for (size_t i = 0; i < num_codebook; ++i) {
        lut[0] = zero;
        for (size_t j = 1; j < 16; ++j) {
            const size_t lowbit = j & (size_t{0} - j);
            lut[j] = lut[j - lowbit] + query[kPos[j]];
        }
        lut += 16;
        query += 4;
    }
}
// Float query tables use runtime SIMD dispatch; other types keep the generic path.
template <>
void pack_lut<float>(size_t dim, const float* __restrict__ query, float* __restrict__ lut);
}  // namespace rabitqlib::fastscan
