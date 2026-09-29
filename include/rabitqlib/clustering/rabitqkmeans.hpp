#pragma once

#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rabitqlib/clustering/detail/kmeans.hpp"
#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::rabitqkmeans {

using clustering::FinalAssignmentMode;
using RaBitQKMeansIterationStats = clustering::KMeansIterationStats;

struct RaBitQKMeansParameters {
    /// Number of clustering iterations.
    size_t niter = 25;
    bool verbose = false;
    /// Normalize centroids after each update.
    bool spherical = false;
    uint32_t seed = 42;
    /// Warn below this number of training vectors per centroid.
    size_t min_points_per_centroid = 39;
    /// Stop when the absolute relative objective (WCSS) change is at most this value.
    double early_stop_threshold = 0.0;
    FinalAssignmentMode final_assignment = FinalAssignmentMode::Approximate;
    uint32_t num_threads = 0;
};

namespace detail {

// Encode immutable training points once; each moving centroid supplies one LUT.
// RaBitQKMeans validates inputs and keeps the borrowed x buffer alive through training.
class RaBitQAssigner {
    size_t d_;
    size_t k_;
    const float* x_;
    size_t n_;
    size_t num_batches_;
    uint32_t num_threads_;
    MetricType metric_type_;
    size_t padded_dim_;
    rotator_impl::FhtKacRotator rotator_;
    std::vector<float> center_;
    std::vector<char> batch_data_;
    std::vector<BatchQuery<float>> centroid_queries_;
    std::vector<float> rotated_query_;
    std::vector<float> lookup_table_;

   public:
    static constexpr const char* kName = "RaBitQKMeans";

    static void validate_parameters(size_t k, const RaBitQKMeansParameters&) {
        if (k - 1 > std::numeric_limits<PID>::max()) {
            throw std::invalid_argument(
                "RaBitQKMeans k exceeds the representable assignment ID range"
            );
        }
    }

    RaBitQAssigner(
        size_t d,
        size_t k,
        const float* x,
        size_t n,
        uint32_t num_threads,
        const RaBitQKMeansParameters& cp
    )
        : d_(d)
        , k_(k)
        , x_(x)
        , n_(n)
        , num_batches_(1 + (n - 1) / fastscan::kBatchSize)
        , num_threads_(static_cast<uint32_t>(std::min<size_t>(num_threads, num_batches_)))
        , metric_type_(cp.spherical ? METRIC_IP : METRIC_L2)
        , padded_dim_(rotator_impl::padding_requirement(d, RotatorType::FhtKacRotator))
        , rotator_(d, padded_dim_, cp.seed)
        , center_(d)
        , batch_data_(num_batches_ * QGBatchDataMap<float>::data_bytes(padded_dim_), 0)
        , centroid_queries_(k)
        , rotated_query_(padded_dim_)
        , lookup_table_(padded_dim_ * 4) {
        {
            std::vector<std::vector<double>> partial_sums(
                num_threads_, std::vector<double>(d_, 0.0)
            );
#pragma omp parallel num_threads(num_threads_)
            {
                auto& sums = partial_sums[static_cast<size_t>(omp_get_thread_num())];
#pragma omp for schedule(static)
                for (std::ptrdiff_t point = 0; point < static_cast<std::ptrdiff_t>(n_);
                     ++point) {
                    const float* vector = x_ + static_cast<size_t>(point) * d_;
                    for (size_t dim = 0; dim < d_; ++dim) {
                        sums[dim] += static_cast<double>(vector[dim]);
                    }
                }
            }
            for (size_t dim = 0; dim < d_; ++dim) {
                double sum = 0.0;
                for (const auto& partial : partial_sums) {
                    sum += partial[dim];
                }
                center_[dim] = static_cast<float>(sum / static_cast<double>(n_));
            }
        }

        std::vector<float> rotated_center(padded_dim_);
        rotator_.rotate(center_.data(), rotated_center.data());
        std::vector<std::vector<float>> rotated_blocks(
            num_threads_, std::vector<float>(fastscan::kBatchSize * padded_dim_)
        );
        std::vector<std::exception_ptr> errors(num_threads_);
#pragma omp parallel for num_threads(num_threads_) schedule(static)
        for (std::ptrdiff_t batch = 0; batch < static_cast<std::ptrdiff_t>(num_batches_);
             ++batch) {
            const auto thread = static_cast<size_t>(omp_get_thread_num());
            if (errors[thread]) {
                continue;
            }
            try {
                const auto block = static_cast<size_t>(batch);
                const size_t first = block * fastscan::kBatchSize;
                const size_t count = std::min(fastscan::kBatchSize, n_ - first);
                auto& rotated = rotated_blocks[thread];
                for (size_t point = 0; point < count; ++point) {
                    rotator_.rotate(
                        x_ + (first + point) * d_, rotated.data() + point * padded_dim_
                    );
                }
                quant::quantize_qg_batch(
                    rotated.data(),
                    rotated_center.data(),
                    count,
                    padded_dim_,
                    batch_data_.data() +
                        block * QGBatchDataMap<float>::data_bytes(padded_dim_),
                    metric_type_
                );
            } catch (...) { errors[thread] = std::current_exception(); }
        }
        for (const auto& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
    }

    void assign(const float* centroids, PID* labels, float* distances, bool = false) {
        const auto distance_func =
            metric_type_ == METRIC_IP ? dot_product_dis<float> : euclidean_sqr<float>;
        for (size_t cluster = 0; cluster < k_; ++cluster) {
            const float* centroid = centroids + cluster * d_;
            rotator_.rotate(centroid, rotated_query_.data());
            auto& query = centroid_queries_[cluster];
            query.reset(
                rotated_query_.data(), padded_dim_, lookup_table_.data(), metric_type_
            );
            query.set_g_add(distance_func(centroid, center_.data(), d_));
        }

        std::vector<std::exception_ptr> errors(num_threads_);
#pragma omp parallel for num_threads(num_threads_) schedule(static)
        for (std::ptrdiff_t batch = 0; batch < static_cast<std::ptrdiff_t>(num_batches_);
             ++batch) {
            const auto thread = static_cast<size_t>(omp_get_thread_num());
            if (errors[thread]) {
                continue;
            }
            try {
                const auto block = static_cast<size_t>(batch);
                const size_t first = block * fastscan::kBatchSize;
                const size_t count = std::min(fastscan::kBatchSize, n_ - first);
                const char* codes = batch_data_.data() +
                                    block * QGBatchDataMap<float>::data_bytes(padded_dim_);
                std::array<float, fastscan::kBatchSize> estimated_distances{};
                std::array<float, fastscan::kBatchSize> best_distances;
                best_distances.fill(std::numeric_limits<float>::infinity());
                std::array<PID, fastscan::kBatchSize> best_labels{};
                for (size_t cluster = 0; cluster < k_; ++cluster) {
                    qg_batch_estdist(
                        codes,
                        centroid_queries_[cluster],
                        padded_dim_,
                        estimated_distances.data()
                    );
                    for (size_t point = 0; point < count; ++point) {
                        // Ascending centroid order retains the lowest ID for exact score
                        // ties.
                        if (estimated_distances[point] < best_distances[point]) {
                            best_distances[point] = estimated_distances[point];
                            best_labels[point] = static_cast<PID>(cluster);
                        }
                    }
                }
                for (size_t point = 0; point < count; ++point) {
                    const size_t id = first + point;
                    labels[id] = best_labels[point];
                    // Scores choose the centroid; objectives retain original-space
                    // distances.
                    distances[id] = distance_func(
                        x_ + id * d_, centroids + best_labels[point] * d_, d_
                    );
                }
            } catch (...) { errors[thread] = std::current_exception(); }
        }
        for (const auto& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
    }
};

}  // namespace detail

/** K-means with a graph-free RaBitQ scan, recommended for small cluster counts. */
class RaBitQKMeans : public clustering::detail::
                         LloydKMeans<RaBitQKMeansParameters, detail::RaBitQAssigner> {
    using Base =
        clustering::detail::LloydKMeans<RaBitQKMeansParameters, detail::RaBitQAssigner>;

   public:
    RaBitQKMeans(size_t d, size_t k) : RaBitQKMeans(d, k, RaBitQKMeansParameters{}) {}

    RaBitQKMeans(size_t d, size_t k, const RaBitQKMeansParameters& cp) : Base(d, k, cp) {}
};

}  // namespace rabitqlib::rabitqkmeans
