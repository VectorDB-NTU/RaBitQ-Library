#pragma once

#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/clustering/detail/emeans.hpp"
#include "rabitqlib/clustering/detail/kmeans.hpp"  // IWYU pragma: export
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/index/detail/parallel_exception.hpp"
#include "rabitqlib/index/detail/visited_set.hpp"
#include "rabitqlib/index/symqg/detail/search_scratch.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::qgemeans {

using clustering::FinalAssignmentMode;

struct QGEMeansParameters {
    // Number of clustering iterations, each one pass over the training vectors.
    size_t niter = 8;
    // Vectors per mini-batch; 0 selects the batch-size law.
    size_t batch_size = 0;
    bool verbose = false;
    uint32_t seed = 42;
    FinalAssignmentMode final_assignment = FinalAssignmentMode::Approximate;

    // SymphonyQG assignment parameters, as in QGKMeansParameters.
    uint32_t max_degree = 32;
    uint32_t ef_construction = 240;
    uint32_t ef_search = 16;
    // Graph-building passes after PiPNN initialization; the last refines the graph.
    uint32_t graph_build_iterations = 1;
    uint32_t num_threads = 0;
};

namespace detail {

// Prefixes error messages and progress output.
constexpr const char* kName = "QGEMeans";

// QGEMeans' copy of the QGKMeans centroid-graph assigner and graph cache, kept
// separate so QGKMeans is unchanged. Training and assign() both search through it.

// Padded dimensions from which assignment caches each query's sum of coordinates.
constexpr size_t kMinQuerySumDim = 512;

// Centroids each search returns: the nearest centroid labels the vector, and the
// second gives the removal cost of cost-based revival.
constexpr uint32_t kSearchResults = 2;

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
    // A search pool smaller than its result count is rejected by the graph.
    if (ef_search < kSearchResults) {
        throw std::invalid_argument(
            std::string(component) + " ef_search must be at least 2"
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

// Graph settings that determine a centroid graph and its search results.
struct GraphSettings {
    uint32_t max_degree;
    uint32_t ef_construction;
    uint32_t ef_search;
    uint32_t graph_build_iterations;
    uint32_t num_threads;
    uint32_t seed;

    [[nodiscard]] bool operator==(const GraphSettings& other) const {
        return max_degree == other.max_degree && ef_construction == other.ef_construction &&
               ef_search == other.ef_search &&
               graph_build_iterations == other.graph_build_iterations &&
               num_threads == other.num_threads && seed == other.seed;
    }
};

/**
 * Builds a raw-vector L2 SymphonyQG graph over a fixed number of centroids and
 * finds each vector's two nearest centroids, reusing the graph allocation and
 * per-thread search scratch. Copied from qgkmeans::QGAssigner, without hints,
 * quantized storage or inner product, so QGKMeans is unchanged.
 */
class QGAssigner {
    friend class QGEAssignment;
    friend class GraphCache;

   public:
    static constexpr uint32_t kResults = kSearchResults;

    QGAssigner(size_t d, size_t k, const GraphSettings& settings)
        : dim_(d)
        , num_clusters_(k)
        , max_degree_(settings.max_degree)
        , ef_construction_(settings.ef_construction)
        , ef_search_(settings.ef_search)
        , graph_build_iterations_(settings.graph_build_iterations)
        , seed_(settings.seed) {
        validate_configuration(settings.num_threads);
        num_workers_ = static_cast<uint32_t>(resolve_num_threads(settings.num_threads));

        graph_ = std::make_unique<symqg::QuantizedGraph<float>>(
            num_clusters_,
            dim_,
            max_degree_,
            METRIC_L2,
            RotatorType::FhtKacRotator,
            0,
            seed_
        );
        graph_->set_ef(ef_search_);
        scratch_ = make_scratch();
    }

   private:
    [[nodiscard]] std::vector<std::unique_ptr<symqg::detail::SearchScratch>>
    make_scratch() const {
        std::vector<std::unique_ptr<symqg::detail::SearchScratch>> scratch;
        scratch.reserve(num_workers_);
        for (uint32_t thread = 0; thread < num_workers_; ++thread) {
            scratch.emplace_back(
                std::make_unique<symqg::detail::SearchScratch>(
                    graph_->padded_dim(), max_degree_, ef_search_, kResults
                )
            );
        }
        return scratch;
    }

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

    // Searches each point independently for its nearest centroid and squared
    // distance, and the squared distance to the second nearest when
    // second_distances is non-null (infinity if the search reaches one centroid).
    // Non-null rows map output i to input row rows[i], which also indexes the
    // query-sum cache.
    void search_points(
        const float* x,
        size_t n,
        const size_t* rows,
        PID* assignments,
        float* distances,
        float* second_distances,
        const QuerySumCache& query_sums,
        const std::vector<std::unique_ptr<symqg::detail::SearchScratch>>& scratch
    ) const {
        index_detail::ParallelException failure;
#pragma omp parallel for num_threads(num_workers_) schedule(dynamic, 64)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(n); ++index) {
            const auto i = static_cast<size_t>(index);
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
                const size_t row = rows == nullptr ? i : rows[i];
                std::array<PID, kResults> ids{};
                std::array<float, kResults> found{};
                symqg::detail::search_with_workspace(
                    *graph_,
                    x + row * dim_,
                    kResults,
                    ids.data(),
                    found.data(),
                    worker_scratch.view(visited),
                    kPidMax,
                    query_sums.cached(row)
                );
                query_sums.store(row, worker_scratch.batch_query.g_k1xsumq());
                assignments[i] = ids[0];
                distances[i] = found[0];
                if (second_distances != nullptr) {
                    second_distances[i] = found[1];
                }
            } catch (...) { failure.capture(); }
        }

        failure.rethrow();
    }

    // Requires a graph built from the centroids. Uses no query-sum cache and
    // allocates its own scratch, so concurrent calls only read the graph.
    void search_built_graph(
        const float* x, size_t n, PID* assignments, float* distances
    ) const {
        const auto scratch = make_scratch();
        const QuerySumCache query_sums(nullptr, false);
        search_points(x, n, nullptr, assignments, distances, nullptr, query_sums, scratch);
    }

    void validate_configuration(uint32_t num_threads) const {
        if (num_threads > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("QGEMeans num_threads must fit in an OpenMP int");
        }
        validate_graph_parameters(
            "QGEMeans",
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

// Graph over a model's final centroids for assign(), rebuilt when the centroids
// or graph settings differ from those it was built from. Copies share the
// immutable graph; each object has its own lock. Copied from
// qgkmeans::detail::CentroidGraphCache, so assign() searches with the same
// QGAssigner as training.
class GraphCache {
    struct Entry {
        std::vector<float> centroids;
        GraphSettings settings;
        QGAssigner assigner;

        Entry(size_t d, size_t k, const float* source, const GraphSettings& graph)
            : centroids(source, source + k * d), settings(graph), assigner(d, k, graph) {
            // The builder borrows this copy, which lives as long as the graph.
            assigner.rebuild_centroid_graph(centroids.data());
        }

        [[nodiscard]] bool matches(
            size_t d, size_t k, const float* source, const GraphSettings& graph
        ) const {
            return centroids.size() == k * d &&
                   std::equal(centroids.begin(), centroids.end(), source) &&
                   settings == graph;
        }
    };

    mutable std::mutex mutex_;
    mutable std::shared_ptr<const Entry> entry_;

    [[nodiscard]] std::shared_ptr<const Entry> snapshot() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return entry_;
    }

   public:
    GraphCache() = default;

    GraphCache(const GraphCache& other) : entry_(other.snapshot()) {}

    GraphCache& operator=(const GraphCache& other) {
        if (this != &other) {
            auto entry = other.snapshot();
            const std::lock_guard<std::mutex> lock(mutex_);
            entry_ = std::move(entry);
        }
        return *this;
    }

    ~GraphCache() = default;

    void clear() {
        const std::lock_guard<std::mutex> lock(mutex_);
        entry_.reset();
    }

    // Requires k * d validated centroid coordinates.
    void assign(
        size_t d,
        size_t k,
        const float* centroids,
        const GraphSettings& settings,
        const float* x,
        size_t n,
        PID* assignments,
        float* distances
    ) const {
        std::shared_ptr<const Entry> entry;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (!entry_ || !entry_->matches(d, k, centroids, settings)) {
                entry_ = std::make_shared<const Entry>(d, k, centroids, settings);
            }
            entry = entry_;
        }
        entry->assigner.search_built_graph(x, n, assignments, distances);
    }
};

[[nodiscard]] inline GraphSettings graph_settings(const QGEMeansParameters& cp) {
    return {
        cp.max_degree,
        cp.ef_construction,
        cp.ef_search,
        cp.graph_build_iterations,
        cp.num_threads,
        cp.seed};
}

// SymphonyQG assignment for EMeans: a raw-vector graph over the current centroids,
// rebuilt by every prepare() and searched for each row's two nearest centroids,
// without hints. The query-sum cache is filled during the first pass, which visits
// every row once, and reused afterwards. assign() searches a cached graph over
// the final centroids.
class QGEAssignment {
    const float* data_;
    QGAssigner assigner_;
    std::vector<float> query_sums_;
    bool sums_ready_ = false;

   public:
    static constexpr const char* kName = detail::kName;

    static void validate_parameters(size_t d, size_t k, const QGEMeansParameters& cp) {
        validate_graph_parameters(
            kName,
            d,
            k,
            cp.max_degree,
            cp.ef_construction,
            cp.ef_search,
            cp.graph_build_iterations
        );
    }

    QGEAssignment(
        size_t d, size_t k, const float* x, size_t n, uint32_t, const QGEMeansParameters& cp
    )
        : data_(x), assigner_(d, k, graph_settings(cp)) {
        if (assigner_.graph_->padded_dim() >= detail::kMinQuerySumDim) {
            query_sums_.resize(n);
        }
    }

    void prepare(const float* centroids) { assigner_.rebuild_centroid_graph(centroids); }

    void assign_batch(
        const size_t* rows,
        size_t count,
        PID* assignments,
        float* distances,
        float* second_distances
    ) {
        const detail::QuerySumCache cache(
            query_sums_.empty() ? nullptr : query_sums_.data(), sums_ready_
        );
        assigner_.search_points(
            data_,
            count,
            rows,
            assignments,
            distances,
            second_distances,
            cache,
            assigner_.scratch_
        );
    }

    // Every row has been searched once, so its query sum is cached.
    void end_pass() { sums_ready_ = true; }

    struct ModelState {
        bool trained = false;
        GraphCache graph;
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
        const QGEMeansParameters& cp,
        const ModelState& state,
        PID* assignments,
        float* distances
    ) {
        state.graph.assign(
            d, k, centroids, graph_settings(cp), x, n, assignments, distances
        );
    }
};

}  // namespace detail

/**
 * Mini-batch e-means with SymphonyQG graph assignment. Each mini-batch is assigned
 * through a raw-vector graph over the current centroids, rebuilt before every
 * batch; centroids follow the e-means mass-weighted update and cost-based revival
 * of cheap centroids. train() then assigns every vector to the final centroids.
 */
class QGEMeans : public detail::EMeans<QGEMeansParameters, detail::QGEAssignment> {
    using Base = detail::EMeans<QGEMeansParameters, detail::QGEAssignment>;

   public:
    QGEMeans(size_t d, size_t k) : QGEMeans(d, k, QGEMeansParameters{}) {}

    QGEMeans(size_t d, size_t k, const QGEMeansParameters& cp) : Base(d, k, cp) {}
};

}  // namespace rabitqlib::qgemeans
