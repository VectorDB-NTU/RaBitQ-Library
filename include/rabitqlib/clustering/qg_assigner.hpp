#pragma once

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/clustering/detail/kmeans.hpp"  // IWYU pragma: export
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"
#include "rabitqlib/index/detail/parallel_exception.hpp"
#include "rabitqlib/index/detail/visited_set.hpp"
#include "rabitqlib/index/symqg/detail/search_scratch.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::qgemeans::detail {
class BatchAssignment;
}  // namespace rabitqlib::qgemeans::detail

namespace rabitqlib::qgkmeans {

namespace detail {

class CentroidGraphCache;
class QGAssignment;

}  // namespace detail

namespace detail {

// Padded dimensions from which assignment caches each query's sum of coordinates.
constexpr size_t kMinQuerySumDim = 512;

// Checks the centroid-graph settings; component prefixes the error messages.
inline void validate_graph_parameters(
    const char* component,
    size_t d,
    size_t k,
    uint32_t graph_degree,
    uint32_t ef_build,
    uint32_t ef_search,
    uint32_t graph_build_iterations
) {
    if (d < 64 || d > rotator_impl::FhtKacRotator::kMaxDim) {
        throw std::invalid_argument(
            std::string(component) +
            " currently requires dimensions in the range [64, 65536]"
        );
    }
    if (graph_degree == 0 || graph_degree % fastscan::kBatchSize != 0) {
        throw std::invalid_argument(
            std::string(component) + " graph degree must be a positive multiple of 32"
        );
    }
    if (k <= graph_degree) {
        throw std::invalid_argument(
            std::string(component) + " requires more centroids than the graph degree"
        );
    }
    if (ef_build == 0 || ef_search == 0) {
        throw std::invalid_argument(
            std::string(component) + " ef_build and ef_search must be positive"
        );
    }
    if (graph_build_iterations == 0) {
        throw std::invalid_argument(
            std::string(component) + " graph_build_iterations must be positive"
        );
    }
}

// One training call owns the storage; the public assign adapter can instead
// borrow caller storage for one batch. State is read-only while workers run.
class QuerySumCache {
    enum class State { Disabled, Pending, Ready };
    State state_ = State::Disabled;
    std::vector<float> storage_;
    float* values_ = nullptr;

   public:
    // Owns storage, enabled from kMinQuerySumDim when the caller reads the sums
    // back after filling them.
    QuerySumCache(size_t padded_dim, bool reused) {
        if (padded_dim >= kMinQuerySumDim && reused) {
            state_ = State::Pending;
        }
    }

    QuerySumCache(float* values, bool ready) : values_(values) {
        if (values != nullptr) {
            state_ = ready ? State::Ready : State::Pending;
        }
    }

    QuerySumCache(const QuerySumCache&) = delete;
    QuerySumCache& operator=(const QuerySumCache&) = delete;

    // Called after centroid initialization releases its point permutation.
    // Training keeps n and the input/rotation fixed throughout this lifetime.
    void prepare(size_t n) {
        if (state_ == State::Pending && values_ == nullptr) {
            storage_.resize(n);
            values_ = storage_.data();
        }
    }

    [[nodiscard]] const float* cached(size_t i) const {
        return state_ == State::Ready ? values_ + i : nullptr;
    }

    void store(size_t i, float value) const {
        if (state_ == State::Pending) {
            values_[i] = value;
        }
    }

    // Commit only after every worker and the assignment's state update succeed.
    void complete() {
        if (state_ == State::Pending) {
            state_ = State::Ready;
        }
    }
};

}  // namespace detail

/**
 * Builds a SymphonyQG over a fixed number of centroids and assigns vectors to
 * them. The graph allocation and per-thread search scratch are reused between
 * calls, and the previous assignment is used as the next call's search hint.
 */
class QGAssigner {
    friend class detail::CentroidGraphCache;
    friend class detail::QGAssignment;
    friend class qgemeans::detail::BatchAssignment;

   public:
    QGAssigner(
        size_t d,
        size_t k,
        uint32_t graph_degree,
        uint32_t ef_build,
        uint32_t ef_search,
        uint32_t graph_build_iterations,
        uint32_t num_threads = 0,
        uint32_t seed = 42,
        MetricType metric_type = METRIC_L2,
        uint32_t quantization_bits = 0,
        const char* component = "QGKMeans"
    )
        : dim_(d)
        , num_clusters_(k)
        , graph_degree_(graph_degree)
        , ef_build_(ef_build)
        , ef_search_(ef_search)
        , graph_build_iterations_(graph_build_iterations)
        , seed_(seed) {
        validate_configuration(num_threads, component);
        num_workers_ = static_cast<uint32_t>(resolve_num_threads(num_threads));

        graph_ = std::make_unique<symqg::QuantizedGraph<float>>(
            num_clusters_,
            dim_,
            graph_degree_,
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
                    graph_->padded_dim(), graph_degree_, ef_search_, 1
                )
            );
        }
    }

    // Optional n-element cache: fill it on the first call, then reuse it only
    // for the same input vectors and this assigner. A null pointer disables it.
    void assign(
        const float* centroids,
        const float* x,
        size_t n,
        PID* assignments,
        float* distances,
        float* query_sums = nullptr,
        bool sums_ready = false
    ) {
        detail::QuerySumCache cache(query_sums, sums_ready);
        assign_with_cache(centroids, x, n, assignments, distances, cache);
    }

    void clear_hints() { previous_assignments_.clear(); }

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
                ef_build_,
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

    void assign_with_cache(
        const float* centroids,
        const float* x,
        size_t n,
        PID* assignments,
        float* distances,
        detail::QuerySumCache& query_sums
    ) {
        if (centroids == nullptr || x == nullptr || assignments == nullptr ||
            distances == nullptr) {
            throw std::invalid_argument(
                "QGAssigner input and output pointers must not be null"
            );
        }

        if (n > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / dim_) {
            throw std::invalid_argument("QGAssigner input dimensions overflow ptrdiff_t");
        }

        rebuild_centroid_graph(centroids);

        // Stable grouping changes only the order of independent full searches.
        std::vector<uint32_t> query_order;
        if (previous_assignments_.size() == n &&
            n <= std::numeric_limits<uint32_t>::max()) {
            query_order.resize(n);
            std::vector<size_t> offsets(num_clusters_ + 1, 0);
            for (PID cluster_id : previous_assignments_) {
                ++offsets[cluster_id + 1];
            }
            std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
            for (size_t point = 0; point < n; ++point) {
                query_order[offsets[previous_assignments_[point]]++] =
                    static_cast<uint32_t>(point);
            }
        }
        search_points(
            centroids,
            x,
            n,
            assignments,
            distances,
            query_order,
            previous_assignments_.size() == n ? previous_assignments_.data() : nullptr,
            query_sums,
            scratch_
        );

        previous_assignments_.assign(assignments, assignments + n);
        query_sums.complete();
    }

    // Requires a graph built from centroids. Uses no hints or query-sum cache
    // and allocates its own scratch, so concurrent calls only read the graph.
    void search_built_graph(
        const float* centroids, const float* x, size_t n, PID* assignments, float* distances
    ) const {
        std::vector<std::unique_ptr<symqg::detail::SearchScratch>> scratch;
        scratch.reserve(num_workers_);
        for (uint32_t thread = 0; thread < num_workers_; ++thread) {
            scratch.emplace_back(
                std::make_unique<symqg::detail::SearchScratch>(
                    graph_->padded_dim(), graph_degree_, ef_search_, 1
                )
            );
        }
        const detail::QuerySumCache query_sums(nullptr, false);
        search_points(
            centroids, x, n, assignments, distances, {}, nullptr, query_sums, scratch
        );
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
        const detail::QuerySumCache& query_sums,
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
        detail::validate_graph_parameters(
            component,
            dim_,
            num_clusters_,
            graph_degree_,
            ef_build_,
            ef_search_,
            graph_build_iterations_
        );
    }

    size_t dim_;
    size_t num_clusters_;
    uint32_t graph_degree_;
    uint32_t ef_build_;
    uint32_t ef_search_;
    uint32_t graph_build_iterations_;
    uint32_t num_workers_ = 0;
    uint32_t seed_;
    std::unique_ptr<symqg::QuantizedGraph<float>> graph_;
    std::unique_ptr<symqg::QGBuilder> builder_;
    std::vector<std::unique_ptr<symqg::detail::SearchScratch>> scratch_;
    std::vector<PID> previous_assignments_;
};

}  // namespace rabitqlib::qgkmeans
