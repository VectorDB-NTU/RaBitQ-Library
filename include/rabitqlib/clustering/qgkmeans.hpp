#pragma once

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
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

namespace rabitqlib::qgkmeans {

using clustering::FinalAssignmentMode;
using QGKMeansIterationStats = clustering::KMeansIterationStats;

namespace detail {

class CentroidGraphCache;
class QGAssignment;

}  // namespace detail

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

// One training call owns the storage; the public assign adapter can instead
// borrow caller storage for one batch. State is read-only while workers run.
class QuerySumCache {
    enum class State { Disabled, Pending, Ready };
    State state_ = State::Disabled;
    std::vector<float> storage_;
    float* values_ = nullptr;

   public:
    QuerySumCache(size_t padded_dim, const QGKMeansParameters& parameters) {
        constexpr size_t kMinCachedDim = 512;
        if (padded_dim >= kMinCachedDim &&
            (parameters.niter > 1 ||
             parameters.final_assignment != FinalAssignmentMode::Exact)) {
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
        uint32_t quantization_bits = 0
    )
        : dim_(d)
        , num_clusters_(k)
        , graph_degree_(graph_degree)
        , ef_build_(ef_build)
        , ef_search_(ef_search)
        , graph_build_iterations_(graph_build_iterations)
        , seed_(seed) {
        validate_configuration(num_threads);
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
    // scheduling; hints, if given, hold one centroid ID per point.
    void search_points(
        const float* centroids,
        const float* x,
        size_t n,
        PID* assignments,
        float* distances,
        const std::vector<uint32_t>& query_order,
        const PID* hints,
        const detail::QuerySumCache& query_sums,
        const std::vector<std::unique_ptr<symqg::detail::SearchScratch>>& scratch
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
                symqg::detail::search_with_workspace(
                    *graph_,
                    x + i * dim_,
                    1,
                    assignments + i,
                    distances + i,
                    worker_scratch.view(visited),
                    hint,
                    query_sums.cached(i)
                );
                query_sums.store(i, worker_scratch.batch_query.g_k1xsumq());
                if (graph_->is_quantized()) {
                    // Keep objectives and returned distances relative to the original
                    // centroids; graph search scores are quantized estimates.
                    distances[i] = distance_func(
                        x + i * dim_, centroids + assignments[i] * dim_, dim_
                    );
                }
            } catch (...) { failure.capture(); }
        }

        failure.rethrow();
    }

    void validate_configuration(uint32_t num_threads) const {
        if (num_threads > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("QGAssigner num_threads must fit in an OpenMP int");
        }
        if (dim_ < 64 || dim_ > rotator_impl::FhtKacRotator::kMaxDim) {
            throw std::invalid_argument(
                "QGKMeans currently requires dimensions in the range [64, 65536]"
            );
        }
        if (graph_degree_ == 0 || graph_degree_ % fastscan::kBatchSize != 0) {
            throw std::invalid_argument(
                "QGKMeans graph degree must be a positive multiple of 32"
            );
        }
        if (num_clusters_ <= graph_degree_) {
            throw std::invalid_argument(
                "QGKMeans requires more centroids than the graph degree"
            );
        }
        if (ef_build_ == 0 || ef_search_ == 0) {
            throw std::invalid_argument("QGKMeans ef_build and ef_search must be positive");
        }
        if (graph_build_iterations_ == 0) {
            throw std::invalid_argument("QGKMeans graph_build_iterations must be positive");
        }
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
        , query_sums_(assigner_.graph_->padded_dim(), cp) {}

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
