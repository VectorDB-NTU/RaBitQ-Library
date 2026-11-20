#pragma once

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rabitqlib/clustering/detail/kmeans.hpp"  // IWYU pragma: export
#include "rabitqlib/clustering/qg_assigner.hpp"    // IWYU pragma: export
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::qgkmeans {

using clustering::FinalAssignmentMode;
using QGKMeansIterationStats = clustering::KMeansIterationStats;

struct QGKMeansParameters {
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

    // SymphonyQG assignment parameters.
    uint32_t graph_degree = 32;
    uint32_t ef_build = 240;
    uint32_t ef_search = 16;
    // Graph-building passes after PiPNN initialization; the last refines the graph.
    uint32_t graph_build_iterations = 1;
    uint32_t num_threads = 0;
    // Centroid storage in QG: 0 for raw float32, or 4/8-bit RaBitQ.
    uint32_t quantization_bits = 0;
    // When positive, train on a random sample of k * max_points_per_centroid
    // vectors if there are more, then assign all vectors; 0 trains on all vectors.
    size_t max_points_per_centroid = 0;
};

namespace detail {

// Whether a training call reads query sums back after filling them: from the
// second Lloyd iteration, or in an approximate final assignment.
[[nodiscard]] inline bool reuses_query_sums(const QGKMeansParameters& parameters) {
    return parameters.niter > 1 ||
           parameters.final_assignment != FinalAssignmentMode::Exact;
}

}  // namespace detail

namespace detail {

// Graph over a model's centroids for stateless assignment, rebuilt when the
// centroids or graph settings differ from those it was built from. Copies share
// the immutable graph; each object has its own lock.
class CentroidGraphCache {
    struct Entry {
        std::vector<float> centroids;
        QGKMeansParameters parameters;
        QGAssigner assigner;

        Entry(size_t d, size_t k, const float* source, const QGKMeansParameters& cp)
            : centroids(source, source + k * d)
            , parameters(cp)
            , assigner(
                  d,
                  k,
                  cp.graph_degree,
                  cp.ef_build,
                  cp.ef_search,
                  cp.graph_build_iterations,
                  cp.num_threads,
                  cp.seed,
                  cp.spherical ? METRIC_IP : METRIC_L2,
                  cp.quantization_bits
              ) {
            // The builder borrows this copy, which lives as long as the graph.
            assigner.rebuild_centroid_graph(centroids.data());
        }

        [[nodiscard]] bool matches(
            size_t d, size_t k, const float* source, const QGKMeansParameters& cp
        ) const {
            const auto& p = parameters;
            return centroids.size() == k * d &&
                   std::equal(centroids.begin(), centroids.end(), source) &&
                   p.spherical == cp.spherical && p.seed == cp.seed &&
                   p.graph_degree == cp.graph_degree && p.ef_build == cp.ef_build &&
                   p.ef_search == cp.ef_search &&
                   p.graph_build_iterations == cp.graph_build_iterations &&
                   p.num_threads == cp.num_threads &&
                   p.quantization_bits == cp.quantization_bits;
        }
    };

    mutable std::mutex mutex_;
    mutable std::shared_ptr<const Entry> entry_;

    [[nodiscard]] std::shared_ptr<const Entry> snapshot() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return entry_;
    }

   public:
    CentroidGraphCache() = default;

    CentroidGraphCache(const CentroidGraphCache& other) : entry_(other.snapshot()) {}

    CentroidGraphCache& operator=(const CentroidGraphCache& other) {
        if (this != &other) {
            auto entry = other.snapshot();
            const std::lock_guard<std::mutex> lock(mutex_);
            entry_ = std::move(entry);
        }
        return *this;
    }

    ~CentroidGraphCache() = default;

    void clear() {
        const std::lock_guard<std::mutex> lock(mutex_);
        entry_.reset();
    }

    // Requires k * d validated centroid coordinates.
    void assign(
        size_t d,
        size_t k,
        const float* centroids,
        const QGKMeansParameters& cp,
        const float* x,
        size_t n,
        PID* assignments,
        float* distances
    ) const {
        std::shared_ptr<const Entry> entry;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (!entry_ || !entry_->matches(d, k, centroids, cp)) {
                entry_ = std::make_shared<const Entry>(d, k, centroids, cp);
            }
            entry = entry_;
        }
        entry->assigner.search_built_graph(
            entry->centroids.data(), x, n, assignments, distances
        );
    }
};

// Retain graph-specific state for one synchronous training call.
class QGAssignment {
    size_t dim_;
    const float* data_;
    size_t num_points_;
    uint32_t num_workers_;
    MetricType metric_type_;
    bool quantized_;
    QGAssigner assigner_;
    QuerySumCache query_sums_;

   public:
    static constexpr const char* kName = "QGKMeans";

    static void validate_parameters(size_t, const QGKMeansParameters& cp) {
        if (cp.quantization_bits != 0 && cp.quantization_bits != 4 &&
            cp.quantization_bits != 8) {
            throw std::invalid_argument("QGKMeans quantization_bits must be 0, 4, or 8");
        }
    }

    QGAssignment(
        size_t d,
        size_t k,
        const float* x,
        size_t n,
        uint32_t num_workers,
        const QGKMeansParameters& cp
    )
        : dim_(d)
        , data_(x)
        , num_points_(n)
        , num_workers_(num_workers)
        , metric_type_(cp.spherical ? METRIC_IP : METRIC_L2)
        , quantized_(cp.quantization_bits != 0)
        , assigner_(
              d,
              k,
              cp.graph_degree,
              cp.ef_build,
              cp.ef_search,
              cp.graph_build_iterations,
              cp.num_threads,
              cp.seed,
              metric_type_,
              cp.quantization_bits
          )
        , query_sums_(assigner_.graph_->padded_dim(), reuses_query_sums(cp)) {}

    void assign(
        const float* centroids, PID* assignments, float* distances, bool final = false
    ) {
        query_sums_.prepare(num_points_);
        std::vector<PID> previous_assignments;
        if (final && quantized_) {
            previous_assignments.assign(assignments, assignments + num_points_);
        }
        assigner_.assign_with_cache(
            centroids, data_, num_points_, assignments, distances, query_sums_
        );
        if (previous_assignments.empty()) {
            return;
        }

        // Protect the returned assignments without changing exploration during
        // Lloyd iterations: uphill approximate moves can improve final centroids.
        const auto distance_func =
            metric_type_ == METRIC_IP ? dot_product_dis<float> : euclidean_sqr<float>;
#pragma omp parallel for num_threads(num_workers_) schedule(static)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_points_);
             ++index) {
            const size_t point = static_cast<size_t>(index);
            const PID previous = previous_assignments[point];
            if (previous == assignments[point]) {
                continue;
            }
            const float distance =
                distance_func(data_ + point * dim_, centroids + previous * dim_, dim_);
            if (distance < distances[point]) {
                assignments[point] = previous;
                distances[point] = distance;
            }
        }
    }

    // Kept by the model after training for stateless assignment.
    struct ModelState {
        bool trained = false;
        CentroidGraphCache graph;
    };

    void store_model_state(ModelState& state) const {
        state.graph.clear();
        state.trained = true;
    }

    [[nodiscard]] static bool has_model_state(const ModelState& state, size_t) {
        return state.trained;
    }

    // Requires validated inputs and centroids. Reuses the cached centroid graph.
    static void assign_with_state(
        size_t d,
        size_t k,
        const float* centroids,
        const float* x,
        size_t n,
        uint32_t,
        const QGKMeansParameters& cp,
        const ModelState& state,
        PID* assignments,
        float* distances
    ) {
        state.graph.assign(d, k, centroids, cp, x, n, assignments, distances);
    }
};

}  // namespace detail

/** K-means with SymphonyQG graph assignment. */
class QGKMeans
    : public clustering::detail::LloydKMeans<QGKMeansParameters, detail::QGAssignment> {
    using Base = clustering::detail::LloydKMeans<QGKMeansParameters, detail::QGAssignment>;

   public:
    QGKMeans(size_t d, size_t k) : QGKMeans(d, k, QGKMeansParameters{}) {}

    QGKMeans(size_t d, size_t k, const QGKMeansParameters& cp) : Base(d, k, cp) {}
};

}  // namespace rabitqlib::qgkmeans
