#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/fastscan/detail/lut.hpp"

namespace rabitqlib {
/**
 * @brief Own a standard FastScan LUT; the query input is not retained.
 */
template <typename T>
class BatchQuery {
   private:
    Lut<T> lookup_table_;  // Owns the prepared bytes; reset scratch is never retained.
    T G_add_ = 0;
    T G_k1xSumq_ = 0;  // G_k1xSumq
    MetricType metric_type_ = METRIC_L2;

    static Lut<T> make_lut(
        const T* rotated_query, size_t padded_dim, MetricType metric_type
    ) {
        validate_metric_type(metric_type);
        return Lut<T>(rotated_query, padded_dim);
    }

    static T query_correction(const T* rotated_query, size_t padded_dim) {
        constexpr float c_1 = -((1 << 1) - 1) / 2.F;
        const T sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0));
        return sumq * c_1;
    }

   public:
    explicit BatchQuery() = default;

    explicit BatchQuery(
        const T* rotated_query, size_t padded_dim, MetricType metric_type = METRIC_L2
    )
        : lookup_table_(make_lut(rotated_query, padded_dim, metric_type))
        , metric_type_(metric_type) {
        G_k1xSumq_ = query_correction(rotated_query, padded_dim);
    }

    [[nodiscard]] T delta() const { return lookup_table_.delta(); }

    [[nodiscard]] T sum_vl_lut() const { return lookup_table_.sum_vl(); }

    [[nodiscard]] T k1xsumq() const { return G_k1xSumq_; }

    [[nodiscard]] T g_add() const { return G_add_; }

    // centroid_distance is squared L2, or 1 - dot(query, centroid) for IP.
    void set_g_add(T centroid_distance) {
        // IP encoder factors already include the constant 1.
        G_add_ = metric_type_ == METRIC_IP ? centroid_distance - T{1} : centroid_distance;
    }

    /**
     * Reinitialize the query while reusing the lookup-table allocation.
     * lut_scratch needs padded_dim * 4 values of T (float for BatchQuery<float>).
     * Input and scratch are borrowed only for this call; neither is retained.
     * An optional correction must come from this exact rotated query.
     * A successful reset clears g_add; set the centroid correction before scoring.
     */
    void reset(
        const T* rotated_query,
        size_t padded_dim,
        T* lut_scratch,
        MetricType metric_type = METRIC_L2,
        const T* k1xsumq = nullptr
    ) {
        validate_metric_type(metric_type);
        // Reject invalid input before changing the correction metadata.
        lookup_table_.reset(rotated_query, padded_dim, lut_scratch);
        metric_type_ = metric_type;

        G_k1xSumq_ =
            k1xsumq != nullptr ? *k1xsumq : query_correction(rotated_query, padded_dim);
        G_add_ = 0;
    }

    [[nodiscard]] size_t padded_dim() const { return lookup_table_.padded_dim(); }

    [[nodiscard]] const uint8_t* lut() const { return lookup_table_.lut(); }
};

template <typename T>
class SplitBatchQuery {
   private:
    static void validate_parameters(size_t ex_bits, MetricType metric_type) {
        if (ex_bits > 8) {
            throw std::invalid_argument("SplitBatchQuery requires ex_bits in [0, 8]");
        }
        validate_metric_type(metric_type);
    }

    static Lut<T> make_lut(
        const T* rotated_query,
        size_t padded_dim,
        size_t ex_bits,
        MetricType metric_type,
        bool use_hacc
    ) {
        validate_parameters(ex_bits, metric_type);
        return Lut<T>(rotated_query, padded_dim, use_hacc);
    }

    // Borrowed until the next successful reset or destruction.
    const T* rotated_query_ = nullptr;
    Lut<T> lookup_table_;
    T G_add_ = 0;
    T G_error_ = 0;
    T G_k1xSumq_ = 0;
    T G_kbxSumq_ = 0;
    MetricType metric_type_ = METRIC_L2;

    void prepare_coefficients(const T* rotated_query, size_t padded_dim, size_t ex_bits) {
        const float c_1 = -static_cast<float>((1 << 1) - 1) / 2.F;
        const float c_b = -static_cast<float>((1 << (ex_bits + 1)) - 1) / 2.F;
        const T sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0));

        G_k1xSumq_ = sumq * c_1;
        G_kbxSumq_ = sumq * c_b;
    }

   public:
    explicit SplitBatchQuery() = default;

    explicit SplitBatchQuery(
        const T* rotated_query,
        size_t padded_dim,
        size_t ex_bits,
        MetricType metric_type = METRIC_L2,
        bool use_hacc = true
    )
        : rotated_query_(rotated_query)
        , lookup_table_(make_lut(rotated_query, padded_dim, ex_bits, metric_type, use_hacc))
        , metric_type_(metric_type) {
        prepare_coefficients(rotated_query, padded_dim, ex_bits);
    }

    /**
     * Reuse owned LUT storage; only first preparation or growth may allocate.
     * Scratch: padded_dim * 4 values of T, plus the same count of uint16_t for
     * HACC (nullptr otherwise). Scratch is call-only; buffers must not overlap
     * each other, the query or LUT. The query remains borrowed for refinement.
     * ex_bits: [0, 8]. Success clears g_add/g_error; call set_g_add before scoring.
     * Invalid arguments or allocation failure preserve the previous state.
     */
    void reset(
        const T* rotated_query,
        size_t padded_dim,
        size_t ex_bits,
        T* lut_scratch,
        uint16_t* lut_u16_scratch,
        MetricType metric_type = METRIC_L2,
        bool use_hacc = true
    ) {
        validate_parameters(ex_bits, metric_type);
        // Validate and grow before changing the borrowed pointer or corrections.
        lookup_table_.reset(
            rotated_query, padded_dim, lut_scratch, lut_u16_scratch, use_hacc
        );
        rotated_query_ = rotated_query;
        metric_type_ = metric_type;
        prepare_coefficients(rotated_query, padded_dim, ex_bits);
        G_add_ = 0;
        G_error_ = 0;
    }
    [[nodiscard]] const T* rotated_query() const { return rotated_query_; }

    [[nodiscard]] T delta() const { return lookup_table_.delta(); }

    [[nodiscard]] T sum_vl_lut() const { return lookup_table_.sum_vl(); }

    [[nodiscard]] T k1xsumq() const { return G_k1xSumq_; }

    [[nodiscard]] T kbxsumq() const { return G_kbxSumq_; }

    [[nodiscard]] T g_add() const { return G_add_; }

    [[nodiscard]] T g_error() const { return G_error_; }

    // residual_norm is ||query - centroid|| for both metrics (not squared distance).
    // For IP supply query_centroid_ip = dot(query, centroid); estimates use 1 - dot.
    void set_g_add(T residual_norm, T query_centroid_ip = 0) {
        if (metric_type_ == METRIC_L2) {
            G_add_ = residual_norm * residual_norm;
            G_error_ = residual_norm;
        } else if (metric_type_ == METRIC_IP) {
            G_add_ = -query_centroid_ip;
            G_error_ = residual_norm;
        }
    }

    [[nodiscard]] bool use_hacc() const { return lookup_table_.use_hacc(); }

    [[nodiscard]] size_t padded_dim() const { return lookup_table_.padded_dim(); }

    [[nodiscard]] const uint8_t* lut() const { return lookup_table_.lut(); }
};

}  // namespace rabitqlib
