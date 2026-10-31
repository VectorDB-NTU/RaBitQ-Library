#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib {
// The binary-query preparation path currently supports float inputs.
// Use SplitSingleQuery<float>; the template spelling does not imply double support.
template <typename T>
class SplitSingleQuery {
   private:
    const T* rotated_query_;  // Borrowed; keep it alive and unchanged during scoring.
    std::vector<uint64_t> query_bin_;
    T G_add_ = 0;
    T G_k1xSumq_;
    T G_kbaseSumq_;
    T G_kbxSumq_;
    T G_error_ = 0;
    T delta_;
    T vl_;
    MetricType metric_type_ = METRIC_L2;

   public:
    static constexpr size_t kNumBits = 4;

    SplitSingleQuery(const SplitSingleQuery&) = default;
    SplitSingleQuery(SplitSingleQuery&&) noexcept = default;
    SplitSingleQuery& operator=(SplitSingleQuery&&) noexcept = default;

    SplitSingleQuery& operator=(const SplitSingleQuery& other) {
        // Copy owned codes before changing their borrowed query and corrections.
        query_bin_ = other.query_bin_;
        rotated_query_ = other.rotated_query_;
        G_add_ = other.G_add_;
        G_k1xSumq_ = other.G_k1xSumq_;
        G_kbaseSumq_ = other.G_kbaseSumq_;
        G_kbxSumq_ = other.G_kbxSumq_;
        G_error_ = other.G_error_;
        delta_ = other.delta_;
        vl_ = other.vl_;
        metric_type_ = other.metric_type_;
        return *this;
    }

    explicit SplitSingleQuery(
        const T* rotated_query,
        size_t padded_dim,
        size_t ex_bits,
        quant::RabitqConfig config,
        MetricType metric_type = METRIC_L2,
        size_t base_bits = 1
    )
        : rotated_query_(rotated_query)
        , query_bin_(div_round_up(padded_dim, 64) * kNumBits, 0) {
        if (base_bits < 1 || base_bits > 8 || ex_bits > 8 || base_bits + ex_bits > 9) {
            throw std::invalid_argument(
                "SplitSingleQuery requires base_bits in [1, 8], ex_bits in [0, 8], "
                "and base_bits + ex_bits <= 9"
            );
        }
        // The query-bit transposition kernels load whole 32-coordinate blocks.
        if (padded_dim == 0 || padded_dim % 32 != 0) {
            throw std::invalid_argument(
                "SplitSingleQuery requires padded_dim to be a positive multiple of 32"
            );
        }

        validate_metric_type(metric_type);

        float c_1 = -static_cast<float>((1 << 1) - 1) / 2.0F;
        float c_base = -static_cast<float>((1U << base_bits) - 1) / 2.0F;
        float c_b = -static_cast<float>((1U << (base_bits + ex_bits)) - 1) / 2.0F;
        T sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0));

        G_k1xSumq_ = sumq * c_1;
        G_kbaseSumq_ = sumq * c_base;
        G_kbxSumq_ = sumq * c_b;

        metric_type_ = metric_type;

        std::vector<uint8_t> quant_query(padded_dim);
        quant::quantize_scalar<float, uint8_t>(
            rotated_query, padded_dim, kNumBits, quant_query.data(), delta_, vl_, config
        );
        new_transpose_bin_512(quant_query.data(), query_bin_.data(), padded_dim, kNumBits);
    }

    [[nodiscard]] size_t num_bits() const { return kNumBits; }

    [[nodiscard]] const uint64_t* query_bin() const { return query_bin_.data(); }

    [[nodiscard]] const T* rotated_query() const { return rotated_query_; }

    [[nodiscard]] T delta() const { return delta_; }

    [[nodiscard]] T vl() const { return vl_; }

    [[nodiscard]] T k1xsumq() const { return G_k1xSumq_; }

    // Equals k1xsumq() when base_bits == 1.
    [[nodiscard]] T kbase_sumq() const { return G_kbaseSumq_; }

    [[nodiscard]] T kbxsumq() const { return G_kbxSumq_; }

    [[nodiscard]] T g_add() const { return G_add_; }

    [[nodiscard]] T g_error() const { return G_error_; }

    // residual_norm is ||query - centroid||, not squared distance. For IP also
    // supply query_centroid_ip = dot(query, centroid). Set before scoring.
    void set_g_add(T residual_norm, T query_centroid_ip = 0) {
        if (metric_type_ == METRIC_L2) {
            G_add_ = residual_norm * residual_norm;
            G_error_ = residual_norm;
        } else if (metric_type_ == METRIC_IP) {
            G_add_ = -query_centroid_ip;
            G_error_ = residual_norm;
        }
    }

    void set_g_error(T residual_norm) { G_error_ = residual_norm; }
};

}  // namespace rabitqlib
