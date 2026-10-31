#pragma once

#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rabitqlib/clustering/detail/kmeans.hpp"  // IWYU pragma: export
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/encoder.hpp"
#include "rabitqlib/fastscan/estimator.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"
#include "rabitqlib/index/detail/parallel_exception.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::rabitqkmeans {

using clustering::FinalAssignmentMode;
using RaBitQKMeansIterationStats = clustering::KMeansIterationStats;

struct RaBitQKMeansParameters {
    // Number of clustering iterations.
    size_t niter = 25;
    bool verbose = false;
    // Normalize centroids after each update.
    bool spherical = false;
    uint32_t seed = 42;
    // Warn below this number of training vectors per centroid.
    size_t min_points_per_centroid = 39;
    // Stop when the absolute relative objective (WCSS) change is at most this value.
    double early_stop_threshold = 0.0;
    FinalAssignmentMode final_assignment = FinalAssignmentMode::Approximate;
    uint32_t num_threads = 0;
    // When positive, train on a random sample of k * max_points_per_centroid
    // vectors if there are more, then assign all vectors; 0 trains on all vectors.
    size_t max_points_per_centroid = 0;
};

namespace detail {

// Encode immutable training points once; each moving centroid supplies one LUT.
// RaBitQKMeans validates inputs and keeps the borrowed x buffer alive through training.
// Points are encoded around their mean, or around a given d-element center.
class RaBitQAssignment {
    size_t dim_;
    size_t num_clusters_;
    const float* data_;
    size_t num_points_;
    size_t num_batches_;
    uint32_t num_workers_;
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

    RaBitQAssignment(
        size_t d,
        size_t k,
        const float* x,
        size_t n,
        uint32_t num_workers,
        const RaBitQKMeansParameters& cp,
        const float* center = nullptr
    )
        : dim_(d)
        , num_clusters_(k)
        , data_(x)
        , num_points_(n)
        , num_batches_(1 + (n - 1) / fastscan::kBatchSize)
        , num_workers_(static_cast<uint32_t>(std::min<size_t>(num_workers, num_batches_)))
        , metric_type_(cp.spherical ? METRIC_IP : METRIC_L2)
        , padded_dim_(rotator_impl::padding_requirement(d, RotatorType::FhtKacRotator))
        , rotator_(d, padded_dim_, cp.seed)
        , center_(d)
        , batch_data_(num_batches_ * QGBatchDataMap<float>::data_bytes(padded_dim_), 0)
        , centroid_queries_(k)
        , rotated_query_(padded_dim_)
        , lookup_table_(padded_dim_ * 4) {
        if (center != nullptr) {
            std::copy_n(center, dim_, center_.data());
        } else {
            std::vector<std::vector<double>> partial_sums(
                num_workers_, std::vector<double>(dim_, 0.0)
            );
#pragma omp parallel num_threads(num_workers_)
            {
                auto& sums = partial_sums[static_cast<size_t>(omp_get_thread_num())];
#pragma omp for schedule(static)
                for (std::ptrdiff_t point = 0;
                     point < static_cast<std::ptrdiff_t>(num_points_);
                     ++point) {
                    const float* vector = data_ + static_cast<size_t>(point) * dim_;
                    for (size_t dim = 0; dim < dim_; ++dim) {
                        sums[dim] += static_cast<double>(vector[dim]);
                    }
                }
            }
            for (size_t dim = 0; dim < dim_; ++dim) {
                double sum = 0.0;
                for (const auto& partial : partial_sums) {
                    sum += partial[dim];
                }
                center_[dim] = static_cast<float>(sum / static_cast<double>(num_points_));
            }
        }

        std::vector<float> rotated_center(padded_dim_);
        rotator_.rotate(center_.data(), rotated_center.data());
        std::vector<std::vector<float>> rotated_blocks(
            num_workers_, std::vector<float>(fastscan::kBatchSize * padded_dim_)
        );
        index_detail::ParallelException failure;
#pragma omp parallel for num_threads(num_workers_) schedule(static)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_batches_);
             ++index) {
            const auto thread = static_cast<size_t>(omp_get_thread_num());
            if (failure.failed()) {
                continue;
            }
            try {
                const auto batch = static_cast<size_t>(index);
                const size_t first = batch * fastscan::kBatchSize;
                const size_t count = std::min(fastscan::kBatchSize, num_points_ - first);
                auto& rotated = rotated_blocks[thread];
                for (size_t point = 0; point < count; ++point) {
                    rotator_.rotate(
                        data_ + (first + point) * dim_, rotated.data() + point * padded_dim_
                    );
                }
                quant::quantize_qg_batch(
                    rotated.data(),
                    rotated_center.data(),
                    count,
                    padded_dim_,
                    batch_data_.data() +
                        batch * QGBatchDataMap<float>::data_bytes(padded_dim_),
                    metric_type_
                );
            } catch (...) { failure.capture(); }
        }
        failure.rethrow();
    }

    // Kept by the model after training: later assignments encode points around
    // the training center, independently of the other points in a call.
    struct ModelState {
        std::vector<float> center;
    };

    void store_model_state(ModelState& state) const { state.center = center_; }

    [[nodiscard]] static bool has_model_state(const ModelState& state, size_t d) {
        return state.center.size() == d;
    }

    // Requires validated inputs and centroids.
    static void assign_with_state(
        size_t d,
        size_t k,
        const float* centroids,
        const float* x,
        size_t n,
        uint32_t num_workers,
        const RaBitQKMeansParameters& cp,
        const ModelState& state,
        PID* assignments,
        float* distances
    ) {
        RaBitQAssignment assigner(d, k, x, n, num_workers, cp, state.center.data());
        assigner.assign(centroids, assignments, distances);
    }

    void assign(const float* centroids, PID* assignments, float* distances, bool = false) {
        const auto distance_func =
            metric_type_ == METRIC_IP ? dot_product_dis<float> : euclidean_sqr<float>;
        for (size_t cluster = 0; cluster < num_clusters_; ++cluster) {
            const float* centroid = centroids + cluster * dim_;
            rotator_.rotate(centroid, rotated_query_.data());
            auto& query = centroid_queries_[cluster];
            query.reset(
                rotated_query_.data(), padded_dim_, lookup_table_.data(), metric_type_
            );
            query.set_g_add(distance_func(centroid, center_.data(), dim_));
        }

        index_detail::ParallelException failure;
#pragma omp parallel for num_threads(num_workers_) schedule(static)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_batches_);
             ++index) {
            if (failure.failed()) {
                continue;
            }
            try {
                const auto batch = static_cast<size_t>(index);
                const size_t first = batch * fastscan::kBatchSize;
                const size_t count = std::min(fastscan::kBatchSize, num_points_ - first);
                const char* codes = batch_data_.data() +
                                    batch * QGBatchDataMap<float>::data_bytes(padded_dim_);
                std::array<float, fastscan::kBatchSize> estimated_distances{};
                std::array<float, fastscan::kBatchSize> best_distances;
                best_distances.fill(std::numeric_limits<float>::infinity());
                std::array<PID, fastscan::kBatchSize> best_cluster_ids{};
                for (size_t cluster = 0; cluster < num_clusters_; ++cluster) {
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
                            best_cluster_ids[point] = static_cast<PID>(cluster);
                        }
                    }
                }
                for (size_t point = 0; point < count; ++point) {
                    const size_t id = first + point;
                    assignments[id] = best_cluster_ids[point];
                    // Scores choose the centroid; objectives retain original-space
                    // distances.
                    distances[id] = distance_func(
                        data_ + id * dim_, centroids + best_cluster_ids[point] * dim_, dim_
                    );
                }
            } catch (...) { failure.capture(); }
        }
        failure.rethrow();
    }
};

}  // namespace detail

/** K-means with a graph-free RaBitQ scan, recommended for small cluster counts. */
class RaBitQKMeans : public clustering::detail::
                         LloydKMeans<RaBitQKMeansParameters, detail::RaBitQAssignment> {
    using Base =
        clustering::detail::LloydKMeans<RaBitQKMeansParameters, detail::RaBitQAssignment>;

   public:
    RaBitQKMeans(size_t d, size_t k) : RaBitQKMeans(d, k, RaBitQKMeansParameters{}) {}

    RaBitQKMeans(size_t d, size_t k, const RaBitQKMeansParameters& cp) : Base(d, k, cp) {}
};

}  // namespace rabitqlib::rabitqkmeans
