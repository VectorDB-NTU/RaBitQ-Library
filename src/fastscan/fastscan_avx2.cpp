#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "rabitqlib/fastscan/detail/simd.hpp"

namespace rabitqlib::fastscan::simd {

namespace {
// Evaluated at compile time: unoptimized builds would otherwise export these library
// functions compiled with this file's ISA flags for callers on other CPUs.
constexpr float kInfinity = std::numeric_limits<float>::infinity();
constexpr float kQuietNaN = std::numeric_limits<float>::quiet_NaN();

template <bool ComputeRange>
void pack_lut_impl(
    size_t dim, const float* query, float* lut, float& minimum, float& maximum
) {
    auto min_values = _mm256_set1_ps(kInfinity), max_values = _mm256_set1_ps(-kInfinity);
    unsigned nan_mask = 0;
    const auto update_range = [&](__m256 values) {
        min_values = _mm256_min_ps(min_values, values);
        max_values = _mm256_max_ps(max_values, values);
        nan_mask |= static_cast<unsigned>(
            _mm256_movemask_ps(_mm256_cmp_ps(values, values, _CMP_UNORD_Q))
        );
    };

    // Keep the initial +0 addition observable for signed-zero inputs.
    volatile float zero = 0;
    for (size_t group = 0; group < dim / 4; ++group) {
        __m256 lo = _mm256_set1_ps(zero);
        __m256 hi = _mm256_add_ps(lo, _mm256_set1_ps(query[0]));
        const __m256 q1 = _mm256_set1_ps(query[1]);
        const __m256 q2 = _mm256_set1_ps(query[2]);
        const __m256 q3 = _mm256_set1_ps(query[3]);
        // Select additions by subset bits, in the same order as the generic loop.
        lo = _mm256_blend_ps(lo, _mm256_add_ps(lo, q1), 0xF0);
        hi = _mm256_blend_ps(hi, _mm256_add_ps(hi, q1), 0xF0);
        lo = _mm256_blend_ps(lo, _mm256_add_ps(lo, q2), 0xCC);
        hi = _mm256_blend_ps(hi, _mm256_add_ps(hi, q2), 0xCC);
        lo = _mm256_blend_ps(lo, _mm256_add_ps(lo, q3), 0xAA);
        hi = _mm256_blend_ps(hi, _mm256_add_ps(hi, q3), 0xAA);
        if constexpr (ComputeRange) {
            update_range(lo);
            update_range(hi);
        }
        _mm256_storeu_ps(lut, lo);
        _mm256_storeu_ps(lut + 8, hi);
        query += 4;
        lut += 16;
    }
    if constexpr (ComputeRange) {
        float min_lanes[8], max_lanes[8];
        _mm256_storeu_ps(min_lanes, min_values);
        _mm256_storeu_ps(max_lanes, max_values);
        // Plain comparisons avoid std algorithms, for the reason given at kInfinity.
        minimum = min_lanes[0];
        maximum = max_lanes[0];
        for (size_t lane = 1; lane < 8; ++lane) {
            minimum = min_lanes[lane] < minimum ? min_lanes[lane] : minimum;
            maximum = max_lanes[lane] > maximum ? max_lanes[lane] : maximum;
        }
        if (nan_mask) {
            minimum = kQuietNaN;
        }
    }
}
}  // namespace

void pack_lut_avx2(size_t dim, const float* query, float* lut) {
    float unused_minimum = 0, unused_maximum = 0;
    pack_lut_impl<false>(dim, query, lut, unused_minimum, unused_maximum);
}

// Preserve the existing ordered subset additions; reduce their values in registers.
// A NaN minimum signals that the wrapper must use the original reduction.
void pack_lut_range_avx2(
    size_t dim, const float* query, float* lut, float& minimum, float& maximum
) {
    pack_lut_impl<true>(dim, query, lut, minimum, maximum);
}

namespace {

// After removing packed high-byte terms, each byte-plane sum is at most
// 255 * (1024 / 4) = 65280, below the 16-bit reduction limit.
constexpr size_t kChunkDim = 1024;
constexpr size_t kBatchSize = 32;

void accumulate_chunk_avx2(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* __restrict__ result,
    size_t dim
) {
    size_t code_length = dim << 2;
    __m256i c, lo, hi, lut, res_lo, res_hi;

    __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i accu0 = _mm256_setzero_si256();
    __m256i accu1 = _mm256_setzero_si256();
    __m256i accu2 = _mm256_setzero_si256();
    __m256i accu3 = _mm256_setzero_si256();

    for (size_t i = 0; i < code_length; i += 64) {
        c = _mm256_loadu_si256((__m256i*)&codes[i]);
        lut = _mm256_loadu_si256((__m256i*)&lookup_table[i]);
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        c = _mm256_loadu_si256((__m256i*)&codes[i + 32]);
        lut = _mm256_loadu_si256((__m256i*)&lookup_table[i + 32]);
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
    }

    accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
    __m256i dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu0, accu1, 0x21),
        _mm256_blend_epi32(accu0, accu1, 0xF0)
    );
    _mm256_storeu_si256(
        (__m256i*)result, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0))
    );
    _mm256_storeu_si256(
        (__m256i*)(result + 8), _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0, 1))
    );

    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
    __m256i dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu2, accu3, 0x21),
        _mm256_blend_epi32(accu2, accu3, 0xF0)
    );
    _mm256_storeu_si256(
        (__m256i*)(result + 16), _mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1))
    );
    _mm256_storeu_si256(
        (__m256i*)(result + 24), _mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1, 1))
    );
}

}  // namespace

void accumulate_avx2(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lookup_table,
    int32_t* __restrict__ result,
    size_t dim
) {
    if (dim <= kChunkDim) {
        accumulate_chunk_avx2(codes, lookup_table, result, dim);
        return;
    }

    std::array<int64_t, kBatchSize> totals{};
    std::array<int32_t, kBatchSize> chunk_result;
    for (size_t offset = 0; offset < dim;) {
        const size_t chunk_dim = std::min(kChunkDim, dim - offset);
        accumulate_chunk_avx2(
            codes + offset * 4, lookup_table + offset * 4, chunk_result.data(), chunk_dim
        );
        for (size_t lane = 0; lane < kBatchSize; ++lane) {
            totals[lane] += chunk_result[lane];
        }
        offset += chunk_dim;
    }
    for (size_t lane = 0; lane < kBatchSize; ++lane) {
        if (totals[lane] > std::numeric_limits<int32_t>::max()) {
            throw std::overflow_error("FastScan result exceeds int32_t");
        }
    }
    for (size_t lane = 0; lane < kBatchSize; ++lane) {
        result[lane] = static_cast<int32_t>(totals[lane]);
    }
}

namespace {

void accumulate_hacc_chunk_avx2(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accumulated_values,
    size_t dim
) {
    __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i accu[2][4];

    for (auto& a : accu) {
        for (auto& reg : a) {
            reg = _mm256_setzero_si256();
        }
    }

    size_t num_codebook = dim >> 2;

    for (size_t m = 0; m < num_codebook; m += 2) {
        __m256i c = _mm256_loadu_si256((__m256i*)codes);
        codes += 32;

        __m256i lo = _mm256_and_si256(c, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        for (int q = 0; q < 2; ++q) {
            __m256i lut = _mm256_loadu_si256((__m256i*)hc_lut);
            hc_lut += 32;

            __m256i res_lo = _mm256_shuffle_epi8(lut, lo);
            __m256i res_hi = _mm256_shuffle_epi8(lut, hi);

            accu[q][0] = _mm256_add_epi16(accu[q][0], res_lo);
            accu[q][1] = _mm256_add_epi16(accu[q][1], _mm256_srli_epi16(res_lo, 8));
            accu[q][2] = _mm256_add_epi16(accu[q][2], res_hi);
            accu[q][3] = _mm256_add_epi16(accu[q][3], _mm256_srli_epi16(res_hi, 8));
        }
    }

    __m256i res[4];
    __m256i dis0[2];
    __m256i dis1[2];

    // lamda function to horizontal sum and combine low/high bytes
    auto combine2x2 = [&](__m256i a, __m256i b) -> __m256i {
        __m256i a1b0 = _mm256_permute2f128_si256(a, b, 0x21);
        __m256i a0b1 = _mm256_blend_epi32(a, b, 0xF0);
        return _mm256_add_epi16(a1b0, a0b1);
    };

    for (size_t i = 0; i < 2; i++) {
        accu[i][0] = _mm256_sub_epi16(accu[i][0], _mm256_slli_epi16(accu[i][1], 8));
        dis0[i] = combine2x2(accu[i][0], accu[i][1]);

        accu[i][2] = _mm256_sub_epi16(accu[i][2], _mm256_slli_epi16(accu[i][3], 8));
        dis1[i] = combine2x2(accu[i][2], accu[i][3]);
    }

    auto add_shiftl8 = [&](__m256i a, __m256i b, __m256i& r0, __m256i& r1) {
        __m256i a0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(a));
        __m256i a1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(a, 1));
        __m256i b0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b));
        __m256i b1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b, 1));
        r0 = _mm256_add_epi32(a0, _mm256_slli_epi32(b0, 8));
        r1 = _mm256_add_epi32(a1, _mm256_slli_epi32(b1, 8));
    };

    // shift res of high, add res of low
    add_shiftl8(dis0[0], dis0[1], res[0], res[1]);  // res for vec 0 to 15
    add_shiftl8(dis1[0], dis1[1], res[2], res[3]);  // res for vec 16 to 31

    _mm256_storeu_si256((__m256i*)(accumulated_values), res[0]);
    _mm256_storeu_si256((__m256i*)(accumulated_values + 8), res[1]);
    _mm256_storeu_si256((__m256i*)(accumulated_values + 16), res[2]);
    _mm256_storeu_si256((__m256i*)(accumulated_values + 24), res[3]);
}

}  // namespace

void accumulate_hacc_avx2(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accumulated_values,
    size_t dim
) {
    if (dim <= kChunkDim) {
        accumulate_hacc_chunk_avx2(codes, hc_lut, accumulated_values, dim);
        return;
    }

    std::array<int64_t, kBatchSize> totals{};
    std::array<int32_t, kBatchSize> chunk_result;
    for (size_t offset = 0; offset < dim;) {
        const size_t chunk_dim = std::min(kChunkDim, dim - offset);
        accumulate_hacc_chunk_avx2(
            codes + offset * 4, hc_lut + offset * 8, chunk_result.data(), chunk_dim
        );
        for (size_t lane = 0; lane < kBatchSize; ++lane) {
            totals[lane] += chunk_result[lane];
        }
        offset += chunk_dim;
    }
    for (size_t lane = 0; lane < kBatchSize; ++lane) {
        if (totals[lane] > std::numeric_limits<int32_t>::max()) {
            throw std::overflow_error("high-accuracy FastScan result exceeds int32_t");
        }
    }
    for (size_t lane = 0; lane < kBatchSize; ++lane) {
        accumulated_values[lane] = static_cast<int32_t>(totals[lane]);
    }
}

}  // namespace rabitqlib::fastscan::simd
