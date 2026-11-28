#pragma once

// QGEMeans' copy of the QGKMeans centroid-graph assigner and of the LloydKMeans
// validation and assignment checks, kept separate so those classes are unchanged.

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/clustering/detail/kmeans.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/index/detail/parallel_exception.hpp"
#include "rabitqlib/index/detail/visited_set.hpp"
#include "rabitqlib/index/symqg/detail/search_scratch.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/input_checks.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::qgemeans::detail {

// Checks n row-major vectors of d coordinates: n * d must fit in ptrdiff_t, and
// every coordinate must be finite and within the clustering bound. component and
// name prefix the error messages. Requires a non-null x and d > 0.
inline void validate_points(
    const char* component,
    size_t n,
    size_t d,
    const float* x,
    uint32_t num_workers,
    const char* name
) {
    if (n > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / d) {
        throw std::invalid_argument(
            std::string(component) + " input dimensions overflow ptrdiff_t"
        );
    }
    // The index bound, so training accepts every point the indexes accept.
    const float max_coordinate = max_coordinate_magnitude(d);
    bool in_range = true;
    bool finite = true;
    static_cast<void>(num_workers);
#pragma omp parallel for num_threads(num_workers) reduction(&& : finite, in_range) \
    schedule(static)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(n * d); ++index) {
        const size_t i = static_cast<size_t>(index);
        finite = finite && std::isfinite(x[i]);
        in_range = in_range && std::abs(x[i]) <= max_coordinate;
    }
    if (!finite) {
        throw std::invalid_argument(
            std::string(component) + " " + name + " must contain only finite values"
        );
    }
    if (!in_range) {
        throw std::invalid_argument(
            std::string(component) + " " + name +
            " exceeds the safe float32 coordinate range"
        );
    }
}

// Checks the settings every clustering class shares. iterations is niter or epochs,
// named by iterations_name in the error message.
inline void validate_common_parameters(
    const char* component,
    size_t d,
    size_t k,
    size_t iterations,
    const char* iterations_name,
    uint32_t num_threads,
    clustering::FinalAssignmentMode final_assignment
) {
    if (d == 0 || k == 0 || iterations == 0) {
        throw std::invalid_argument(
            std::string(component) + " d, k, and " + iterations_name + " must be positive"
        );
    }
    if (d < 64 || d > rotator_impl::FhtKacRotator::kMaxDim) {
        throw std::invalid_argument(
            std::string(component) +
            " currently requires dimensions in the range [64, 65536]"
        );
    }
    if (num_threads > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            std::string(component) + " num_threads must fit in an OpenMP int"
        );
    }
    if (final_assignment != clustering::FinalAssignmentMode::Approximate &&
        final_assignment != clustering::FinalAssignmentMode::Exact) {
        throw std::invalid_argument(
            std::string(component) + " final_assignment mode is invalid"
        );
    }
}

// The shared public assign(): checks that a model is trained, with k * d valid
// centroids, and validates the inputs, then calls assign_points() when n > 0.
template <typename AssignPoints>
void assign_trained(
    const char* component,
    bool trained,
    size_t d,
    size_t k,
    const std::vector<float>& centroids,
    size_t n,
    const float* x,
    const PID* point_assignments,
    const float* point_distances,
    uint32_t num_workers,
    AssignPoints&& assign_points
) {
    if (!trained || centroids.size() != k * d) {
        throw std::runtime_error(std::string(component) + " has not been trained");
    }
    validate_points(component, k, d, centroids.data(), num_workers, "centroids");
    if (n == 0) {
        return;
    }
    if (x == nullptr || point_assignments == nullptr || point_distances == nullptr) {
        throw std::invalid_argument(
            std::string(component) + " assign input and output pointers must not be null"
        );
    }
    validate_points(component, n, d, x, num_workers, "x");
    std::forward<AssignPoints>(assign_points)();
}

// Final assignment of n validated points: exhaustive for Exact, approximate()
// otherwise. Requires n > 0 and outputs that do not overlap x or centroids.
template <typename Approximate>
void assign_with_mode(
    clustering::FinalAssignmentMode final_assignment,
    const float* x,
    const float* centroids,
    size_t n,
    size_t d,
    size_t k,
    bool spherical,
    uint32_t num_workers,
    PID* point_assignments,
    float* point_distances,
    Approximate&& approximate
) {
    if (final_assignment == clustering::FinalAssignmentMode::Exact) {
        clustering::detail::exact_assign(
            x,
            centroids,
            n,
            d,
            k,
            spherical,
            num_workers,
            point_assignments,
            point_distances
        );
    } else {
        std::forward<Approximate>(approximate)();
    }
}

// Padded dimensions from which assignment caches each query's sum of coordinates.
constexpr size_t kMinQuerySumDim = 512;

// Checks the centroid-graph settings; component prefixes the error messages.
inline void validate_graph_parameters(
    const char* component,
    size_t d,
    size_t k,
    uint32_t max_degree,
    uint32_t ef_construction,
    uint32_t ef_search,
    uint32_t graph_build_iterations
) {
    if (d < 64 || d > rotator_impl::FhtKacRotator::kMaxDim) {
        throw std::invalid_argument(
            std::string(component) +
            " currently requires dimensions in the range [64, 65536]"
        );
    }
    if (max_degree == 0 || max_degree % fastscan::kBatchSize != 0) {
        throw std::invalid_argument(
            std::string(component) + " max_degree must be a positive multiple of 32"
        );
    }
    if (k <= max_degree) {
        throw std::invalid_argument(
            std::string(component) + " requires more centroids than max_degree"
        );
    }
    if (ef_construction == 0 || ef_search == 0) {
        throw std::invalid_argument(
            std::string(component) + " ef_construction and ef_search must be positive"
        );
    }
    if (graph_build_iterations == 0) {
        throw std::invalid_argument(
            std::string(component) + " graph_build_iterations must be positive"
        );
    }
}

// Borrows caller storage for the query sums of one training call. State is
// read-only while workers run.
class QuerySumCache {
    enum class State { Disabled, Pending, Ready };
    State state_ = State::Disabled;
    float* values_ = nullptr;

   public:
    QuerySumCache(float* values, bool ready) : values_(values) {
        if (values != nullptr) {
            state_ = ready ? State::Ready : State::Pending;
        }
    }

    QuerySumCache(const QuerySumCache&) = delete;
    QuerySumCache& operator=(const QuerySumCache&) = delete;

    [[nodiscard]] const float* cached(size_t i) const {
        return state_ == State::Ready ? values_ + i : nullptr;
    }

    void store(size_t i, float value) const {
        if (state_ == State::Pending) {
            values_[i] = value;
        }
    }
};

/**
 * Builds a raw SymphonyQG graph over a fixed number of centroids and assigns
 * vectors to them, reusing the graph allocation and per-thread search scratch.
 */
class QGAssigner {
    friend class BatchAssignment;

   public:
    QGAssigner(
        size_t d,
        size_t k,
        uint32_t max_degree,
        uint32_t ef_construction,
        uint32_t ef_search,
        uint32_t graph_build_iterations,
        uint32_t num_threads = 0,
        uint32_t seed = 42,
        MetricType metric_type = METRIC_L2,
        uint32_t quantization_bits = 0,
        const char* component = "QGEMeans"
    )
        : dim_(d)
        , num_clusters_(k)
        , max_degree_(max_degree)
        , ef_construction_(ef_construction)
        , ef_search_(ef_search)
        , graph_build_iterations_(graph_build_iterations)
        , seed_(seed) {
        validate_configuration(num_threads, component);
        // Cosine belongs to the indexes, which normalize vectors; the assigner's exact
        // distances are raw L2 or inner product.
        validate_metric_type(metric_type);
        num_workers_ = static_cast<uint32_t>(resolve_num_threads(num_threads));

        graph_ = std::make_unique<symqg::QuantizedGraph<float>>(
            num_clusters_,
            dim_,
            max_degree_,
            metric_type,
            RotatorType::FhtKacRotator,
            quantization_bits,
            seed_
        );
        graph_->set_ef(ef_search_);

        scratch_.reserve(num_workers_);
        for (uint32_t thread = 0; thread < num_workers_; ++thread) {
            scratch_.emplace_back(
                std::make_unique<symqg::detail::SearchScratch>(
                    graph_->padded_dim(), max_degree_, ef_search_, 1
                )
            );
        }
    }

   private:
    void rebuild_centroid_graph(const float* centroids) {
        if (builder_) {
            builder_->reset(centroids);
        } else {
            // Centroid graphs have much less work than the training-vector batch.
            // Amortize their short phases without reducing assignment parallelism.
            constexpr size_t kMinVerticesPerBuildThread = 128;
            const size_t build_threads = std::min<size_t>(
                num_workers_,
                std::max<size_t>(1, num_clusters_ / kMinVerticesPerBuildThread)
            );
            builder_ = std::make_unique<symqg::QGBuilder>(
                *graph_,
                ef_construction_,
                centroids,
                build_threads,
                symqg::QGInitialization::PiPNN,
                seed_,
                true
            );
        }
        if (graph_build_iterations_ == 1) {
            builder_->build();
        } else {
            builder_->build(graph_build_iterations_);
        }
    }

    // Searches each point independently. A non-empty query_order only changes
    // scheduling; hints, if given, hold one centroid ID per point. Non-null rows
    // map output i to input row rows[i], which also indexes the query-sum cache.
    void search_points(
        const float* centroids,
        const float* x,
        size_t n,
        PID* assignments,
        float* distances,
        const std::vector<uint32_t>& query_order,
        const PID* hints,
        const QuerySumCache& query_sums,
        const std::vector<std::unique_ptr<symqg::detail::SearchScratch>>& scratch,
        const size_t* rows = nullptr
    ) const {
        const auto distance_func = graph_->metric_type() == METRIC_IP
                                       ? dot_product_dis<float>
                                       : euclidean_sqr<float>;
        index_detail::ParallelException failure;
#pragma omp parallel for num_threads(num_workers_) schedule(dynamic, 64)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(n); ++index) {
            const size_t i = query_order.empty() ? static_cast<size_t>(index)
                                                 : query_order[static_cast<size_t>(index)];
            const int thread_id = omp_get_thread_num();
            if (failure.failed()) {
                continue;
            }
            try {
                auto& worker_scratch = *scratch[static_cast<size_t>(thread_id)];
                // Keep visited storage with the worker, as in QuantizedGraph::search.
                thread_local VisitedSet visited;
                thread_local size_t visited_size = 0;
                thread_local size_t visited_capacity = 0;
                const size_t capacity = std::min(
                    static_cast<size_t>(ef_search_) * ef_search_, num_clusters_ / 10
                );
                if (visited_size != num_clusters_ || visited_capacity != capacity) {
                    visited.initialize(num_clusters_, capacity);
                    visited_size = num_clusters_;
                    visited_capacity = capacity;
                }
                const PID hint = hints == nullptr ? kPidMax : hints[i];
                const size_t row = rows == nullptr ? i : rows[i];
                symqg::detail::search_with_workspace(
                    *graph_,
                    x + row * dim_,
                    1,
                    assignments + i,
                    distances + i,
                    worker_scratch.view(visited),
                    hint,
                    query_sums.cached(row)
                );
                query_sums.store(row, worker_scratch.batch_query.g_k1xsumq());
                if (graph_->is_quantized()) {
                    // Keep objectives and returned distances relative to the original
                    // centroids; graph search scores are quantized estimates.
                    distances[i] = distance_func(
                        x + row * dim_, centroids + assignments[i] * dim_, dim_
                    );
                }
            } catch (...) { failure.capture(); }
        }

        failure.rethrow();
    }

    void validate_configuration(uint32_t num_threads, const char* component) const {
        if (num_threads > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("QGAssigner num_threads must fit in an OpenMP int");
        }
        validate_graph_parameters(
            component,
            dim_,
            num_clusters_,
            max_degree_,
            ef_construction_,
            ef_search_,
            graph_build_iterations_
        );
    }

    size_t dim_;
    size_t num_clusters_;
    uint32_t max_degree_;
    uint32_t ef_construction_;
    uint32_t ef_search_;
    uint32_t graph_build_iterations_;
    uint32_t num_workers_ = 0;
    uint32_t seed_;
    std::unique_ptr<symqg::QuantizedGraph<float>> graph_;
    std::unique_ptr<symqg::QGBuilder> builder_;
    std::vector<std::unique_ptr<symqg::detail::SearchScratch>> scratch_;
};

}  // namespace rabitqlib::qgemeans::detail
