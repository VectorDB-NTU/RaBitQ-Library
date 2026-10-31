#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/core/detail/simd/space_dispatch.hpp"
#include "rabitqlib/utils/bitops.hpp"
#include "rabitqlib/utils/openmp.hpp"  // IWYU pragma: keep (configuration macros)
#include "rabitqlib/utils/tools.hpp"
#include "rabitqlib/utils/warmup_space.hpp"

namespace rabitqlib {

static inline uint64_t load_unaligned_u64(const void* data) {
    uint64_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

namespace scalar_impl {
template <typename T>
void scalar_quantize_normal(
    T* __restrict__ result,
    const float* __restrict__ data,
    size_t dim,
    float lo,
    float delta
) {
    if (delta == 0.0F) {
        std::fill_n(result, dim, T{0});
        return;
    }
    float one_over_delta = 1.0F / delta;

    ConstRowMajorArrayMap<float> v0(data, 1, static_cast<long>(dim));
    RowMajorArrayMap<T> res(result, 1, dim);

    // round to nearest integer, then cast to integer
    res = ((v0 - lo) * one_over_delta).round().template cast<T>();
}

template <typename T>
void scalar_quantize_optimized(
    T* __restrict__ result,
    const float* __restrict__ data,
    size_t dim,
    float lo,
    float delta
) {
    if constexpr (std::is_same_v<T, uint8_t>) {
        simd::scalar_quantize_uint8(result, data, dim, lo, delta);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        simd::scalar_quantize_uint16(result, data, dim, lo, delta);
    } else {
        scalar_quantize_normal(result, data, dim, lo, delta);
    }
}
}  // namespace scalar_impl

template <typename T>
inline void vec_rescale(T* data, size_t dim, T val) {
    RowMajorArrayMap<T> data_arr(data, 1, dim);
    data_arr *= val;
}

template <typename T>
inline T euclidean_sqr(const T* __restrict__ vec0, const T* __restrict__ vec1, size_t dim) {
    if constexpr (std::is_same_v<T, float>) {
        return simd::euclidean_sqr(vec0, vec1, dim);
    } else {
        ConstVectorMap<T> v0(vec0, dim);
        ConstVectorMap<T> v1(vec1, dim);
        return (v0 - v1).dot(v0 - v1);
    }
}

template <typename T>
inline T dot_product_dis(
    const T* __restrict__ vec0, const T* __restrict__ vec1, size_t dim
) {
    if constexpr (std::is_same_v<T, float>) {
        return simd::dot_product_dis(vec0, vec1, dim);
    } else {
        ConstVectorMap<T> v0(vec0, dim);
        ConstVectorMap<T> v1(vec1, dim);
        return 1 - v0.dot(v1);
    }
}

template <typename T>
inline T l2norm_sqr(const T* __restrict__ vec0, size_t dim) {
    if constexpr (std::is_same_v<T, float>) {
        return simd::l2norm_sqr(vec0, dim);
    } else {
        ConstVectorMap<T> v0(vec0, dim);
        return v0.dot(v0);
    }
}

template <typename T>
inline T dot_product(const T* __restrict__ vec0, const T* __restrict__ vec1, size_t dim) {
    if constexpr (std::is_same_v<T, float>) {
        return simd::dot_product(vec0, vec1, dim);
    } else {
        ConstVectorMap<T> v0(vec0, dim);
        ConstVectorMap<T> v1(vec1, dim);
        return v0.dot(v1);
    }
}

template <typename T>
inline T normalize_vec(
    const T* __restrict__ vec, const T* __restrict__ centroid, T* res, T dist2c, size_t dim
) {
    RowMajorArrayMap<T> r(res, 1, dim);
    if (dist2c > 1e-5) {
        ConstRowMajorArrayMap<T> v(vec, 1, dim);
        ConstRowMajorArrayMap<T> c(centroid, 1, dim);
        r = (v - c) * (1 / dist2c);
        return r.sum();
    }
    T value = 1.0 / std::sqrt(static_cast<T>(dim));
    r = value;
    return static_cast<T>(dim) * value;
}

// Pack 0/1 data into words without requiring the destination byte storage to
// contain aligned, lifetime-started T objects. Length is a multiple of 8; the
// final partial word stores only its used bytes, with the first coordinate at
// its highest used bit. Full words retain their original layout.
template <typename T>
inline void pack_binary_to_bytes(
    const int* __restrict__ binary_code, uint8_t* __restrict__ compact_code, size_t length
) {
    constexpr size_t kTypeBits = sizeof(T) * 8;

    for (size_t i = 0; i < length; i += kTypeBits) {
        const size_t width = std::min(kTypeBits, length - i);
        T cur = 0;
        for (size_t j = 0; j < width; j += 8) {
            uint8_t byte = 0;
            for (size_t bit = 0; bit < 8; ++bit) {
                byte |= static_cast<uint8_t>(binary_code[i + j + bit] << (7 - bit));
            }
            cur = static_cast<T>((cur << 8) | byte);
        }
        std::memcpy(compact_code, &cur, width / 8);
        compact_code += width / 8;
    }
}

// Backward-compatible typed-output overload. The implementation remains
// byte-oriented, so it does not dereference compact_code as T.
template <typename T>
inline void pack_binary(
    const int* __restrict__ binary_code, T* __restrict__ compact_code, size_t length
) {
    pack_binary_to_bytes<T>(binary_code, reinterpret_cast<uint8_t*>(compact_code), length);
}

template <typename T>
inline void data_range(const T* __restrict__ vec0, size_t dim, T& lo, T& hi) {
    ConstRowMajorArrayMap<T> v0(vec0, 1, dim);
    lo = v0.minCoeff();
    hi = v0.maxCoeff();
}

// A constant range (delta == 0) is represented by zero codes.
template <typename T, typename TD>
void scalar_quantize(
    T* __restrict__ result, const TD* __restrict__ vec0, size_t dim, TD lo, TD delta
) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    scalar_impl::scalar_quantize_optimized(result, vec0, dim, lo, delta);
}

template <typename T>
inline std::vector<T> compute_centroid(
    const T* data, size_t num_points, size_t dim, size_t num_threads
) {
#if RABITQ_USE_OPENMP
    const int thread_count = detail::openmp_team_size(num_threads, num_points);
#else
    static_cast<void>(num_threads);
    constexpr int thread_count = 1;
#endif
    std::vector<std::vector<T>> all_results(thread_count, std::vector<T>(dim, 0));

#if RABITQ_USE_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(thread_count)
#endif
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_points);
         ++index) {
        const size_t i = static_cast<size_t>(index);
#if RABITQ_USE_OPENMP
        const int tid = omp_get_thread_num();
#else
        constexpr int tid = 0;
#endif
        std::vector<T>& cur_results = all_results[tid];
        const T* cur_data = data + (dim * i);
        for (size_t k = 0; k < dim; ++k) {
            cur_results[k] += cur_data[k];
        }
    }

    std::vector<T> centroid(dim, 0);
    for (auto& one_res : all_results) {
        for (size_t i = 0; i < dim; ++i) {
            centroid[i] += one_res[i];
        }
    }
    T inv_num_points = 1 / static_cast<T>(num_points);

    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = centroid[i] * inv_num_points;
    }

    return centroid;
}

namespace excode_ipimpl {

// Compatibility names: despite the _avx suffix, these call the runtime-selected
// kernel for their ex_bits (AVX-512, AVX2, NEON or generic), like select_excode_ipfunc.
float ip16_fxu1_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
);
float ip64_fxu2_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
);
float ip64_fxu3_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
);
float ip16_fxu4_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
);
float ip64_fxu5_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
);
float ip64_fxu6_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
);
float ip64_fxu7_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
);

// inner product between float type and int type vectors
template <typename TF, typename TI>
inline TF ip_fxi(const TF* __restrict__ vec0, const TI* __restrict__ vec1, size_t dim) {
    static_assert(std::is_floating_point_v<TF>, "TF must be a floating-point type");
    static_assert(std::is_integral_v<TI>, "TI must be an integral type");

    ConstVectorMap<TF> v0(vec0, dim);
    ConstVectorMap<TI> v1(vec1, dim);
    return v0.dot(v1.template cast<TF>());
}
}  // namespace excode_ipimpl

using ex_ipfunc = float (*)(const float*, const uint8_t*, size_t);

ex_ipfunc select_excode_ipfunc(size_t ex_bits);

// ISA-flagged sources include this header, so its scalar helpers keep internal linkage.
static inline uint32_t reverse_bits(uint32_t n) {
    n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
    n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
    n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
    n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
    return n;
}

static inline uint64_t reverse_bits_u64(uint64_t n) {
    n = ((n >> 1) & 0x5555555555555555) | ((n << 1) & 0xaaaaaaaaaaaaaaaa);
    n = ((n >> 2) & 0x3333333333333333) | ((n << 2) & 0xcccccccccccccccc);
    n = ((n >> 4) & 0x0f0f0f0f0f0f0f0f) | ((n << 4) & 0xf0f0f0f0f0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff00ff00ff) | ((n << 8) & 0xff00ff00ff00ff00);
    n = ((n >> 16) & 0x0000ffff0000ffff) | ((n << 16) & 0xffff0000ffff0000);
    n = ((n >> 32) & 0x00000000ffffffff) | ((n << 32) & 0xffffffff00000000);
    return n;
}

// padded_dim is a multiple of 32. Transposed queries need ceil(padded_dim / 64) * b_query
// uint64_t words, including a zero-extended final partial word in each bit plane.
void new_transpose_bin(
    const uint16_t* query, uint64_t* transposed_query, size_t padded_dim, size_t b_query
);

void new_transpose_bin_512(
    const uint8_t* query, uint64_t* transposed_query, size_t padded_dim, size_t b_query
);

float mask_ip_x0_q(const float* query, const uint8_t* data, size_t padded_dim);
float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim);

static inline float mask_ip_x0_q_old(
    const float* query, const uint64_t* data, size_t padded_dim
) {
    return mask_ip_x0_q(query, data, padded_dim);
}

static inline float ip_x0_q(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    return detail::ip_x0_q_impl(data, query, delta, vl, padded_dim, b_query);
}

static inline uint32_t ip_bin_bin(const uint64_t* q, const uint64_t* d, size_t padded_dim) {
    uint64_t ret = 0;
    const size_t words = padded_dim / 64;
    const auto* query = reinterpret_cast<const uint8_t*>(q);
    const auto* data = reinterpret_cast<const uint8_t*>(d);
    for (size_t i = 0; i < words; ++i) {
        ret += bitops::popcount64(
            bitops::load_word(data + i * 8, 8) & bitops::load_word(query + i * 8, 8)
        );
    }
    if (padded_dim % 64 != 0) {
        const size_t bytes = (padded_dim % 64) / 8;
        ret += bitops::popcount64(
            bitops::load_word(data + words * 8, bytes) &
            bitops::load_word(query + words * 8, bytes)
        );
    }
    return ret;
}

static inline uint32_t ip_byte_bin(
    const uint64_t* q, const uint64_t* d, size_t padded_dim, size_t b_query
) {
    uint32_t ret = 0;
    size_t offset = (padded_dim + 63) / 64;
    for (size_t i = 0; i < b_query; ++i) {
        ret += (ip_bin_bin(q, d, padded_dim) << i);
        q += offset;
    }
    return ret;
}

static inline size_t popcount(const uint64_t* __restrict__ d, size_t length) {
    size_t ret = 0;
    const size_t words = length / 64;
    const auto* data = reinterpret_cast<const uint8_t*>(d);
    for (size_t i = 0; i < words; ++i) {
        ret += bitops::popcount64(bitops::load_word(data + i * 8, 8));
    }
    if (length % 64 != 0) {
        ret += bitops::popcount64(bitops::load_word(data + words * 8, (length % 64) / 8));
    }
    return ret;
}

template <typename T>
RowMajorMatrix<T> random_gaussian_matrix(
    size_t rows, size_t cols, std::optional<unsigned> seed = std::nullopt
) {
    RowMajorMatrix<T> rand(rows, cols);
    std::mt19937 gen(seed.has_value() ? *seed : std::random_device{}());
    std::normal_distribution<T> dist(0, 1);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            rand(i, j) = dist(gen);
        }
    }

    return rand;
}
}  // namespace rabitqlib
