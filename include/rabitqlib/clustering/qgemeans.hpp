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
    friend class BatchAssignment;
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

// Raw-vector centroid graph for one training call: rebuilt from the current
// centroids, then searched for the nearest centroid of each listed row, without
// hints. The query-sum cache is filled during the first iteration, which visits
// every row once, and reused afterwards.
class BatchAssignment {
    QGAssigner assigner_;
    std::vector<float> query_sums_;
    bool sums_ready_ = false;

   public:
    BatchAssignment(size_t d, size_t k, size_t n, const QGEMeansParameters& cp)
        : assigner_(d, k, graph_settings(cp)) {
        if (assigner_.graph_->padded_dim() >= detail::kMinQuerySumDim) {
            query_sums_.resize(n);
        }
    }

    void rebuild(const float* centroids) { assigner_.rebuild_centroid_graph(centroids); }

    // Requires a graph rebuilt from the current centroids. Null rows assign rows 0
    // to count - 1; a non-null second_distances receives each row's squared
    // distance to its second-nearest centroid.
    void search(
        const float* x,
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
            x,
            count,
            rows,
            assignments,
            distances,
            second_distances,
            cache,
            assigner_.scratch_
        );
    }

    // Called after a pass that searched every row.
    void complete_query_sums() { sums_ready_ = true; }
};

}  // namespace detail

/**
 * Mini-batch e-means with SymphonyQG graph assignment. Each mini-batch is assigned
 * through a raw-vector graph over the current centroids, rebuilt before every
 * batch; centroids follow the e-means mass-weighted update and cost-based revival
 * of cheap centroids. train() then assigns every vector to the final centroids.
 */
class QGEMeans : public QGEMeansParameters {
   public:
    QGEMeans(size_t d, size_t k) : QGEMeans(d, k, QGEMeansParameters{}) {}

    QGEMeans(size_t d, size_t k, const QGEMeansParameters& cp)
        : QGEMeansParameters(cp), d(d), k(k) {
        validate_parameters();
    }

    void train(size_t n, const float* x) { fit(n, x, true); }

    /**
     * Trains the same centroids as train() but skips its final assignment of all n
     * vectors. Leaves assignments and distances empty and final_obj zero;
     * centroids, iteration_stats and assign() match a train() call.
     */
    void train_centroids(size_t n, const float* x) { fit(n, x, false); }

    /**
     * Assigns n row-major vectors to the current centroids with final_assignment.
     * Each vector's result is independent of the others in the call. Requires a
     * successful train() or train_centroids(); the outputs hold n elements each
     * and must not overlap x or centroids.
     */
    void assign(
        size_t n, const float* x, PID* point_assignments, float* point_distances
    ) const {
        validate_parameters();
        const uint32_t num_workers = resolved_num_workers();
        detail::assign_trained(
            detail::kName,
            trained_,
            d,
            k,
            centroids,
            n,
            x,
            point_assignments,
            point_distances,
            num_workers,
            [&] { assign_points(n, x, num_workers, point_assignments, point_distances); }
        );
    }

    size_t d;
    size_t k;
    std::vector<float> centroids;
    std::vector<PID> assignments;
    std::vector<float> distances;
    // One entry per iteration: obj sums the mini-batch assignment distances of that
    // iteration, shift is the squared centroid displacement over it, and nsplit counts
    // revived centroids.
    std::vector<clustering::KMeansIterationStats> iteration_stats;
    // Objective of the final returned assignments against the final centroids.
    double final_obj = 0.0;

   private:
    void fit(size_t n, const float* x, bool assign_all) {
        validate_parameters();
        if (x == nullptr) {
            throw std::invalid_argument(std::string(detail::kName) + " x must not be null");
        }
        if (n < k) {
            throw std::invalid_argument(std::string(detail::kName) + " requires n >= k");
        }
        const uint32_t num_workers = resolved_num_workers();
        detail::validate_points(detail::kName, n, d, x, num_workers, "x");

        const detail::Recipe recipe = detail::resolve_recipe(n, k, niter, batch_size);
        if (verbose) {
            std::cout << detail::kName << " mini-batches of " << recipe.batch_size
                      << " training points\n";
        }
        detail::BatchAssignment graph(d, k, n, *this);
        detail::EMeansState state;
        detail::initialize(state, d, k, n, x, recipe.batch_size, seed);

        std::vector<clustering::KMeansIterationStats> stats;
        stats.reserve(niter);
        std::vector<size_t> order(n);
        std::vector<float> iteration_start(state.centroids.size());
        size_t step = 0;
        for (size_t iteration = 0; iteration < niter; ++iteration) {
            const bool last = iteration + 1 == niter;
            std::copy(
                state.centroids.begin(), state.centroids.end(), iteration_start.begin()
            );
            detail::start_iteration(state, last);
            std::iota(order.begin(), order.end(), size_t{0});
            std::mt19937_64 rng(detail::iteration_seed(seed, iteration));
            std::shuffle(order.begin(), order.end(), rng);
            double objective = 0.0;
            size_t nsplit = 0;
            for (size_t begin = 0; begin < n; begin += recipe.batch_size) {
                const size_t rows = std::min(recipe.batch_size, n - begin);
                const size_t* batch = order.data() + begin;
                graph.rebuild(state.centroids.data());
                graph.search(
                    x,
                    batch,
                    rows,
                    state.batch_assignments.data(),
                    state.batch_distances.data(),
                    state.batch_second_distances.data()
                );
                objective += detail::accumulate(state, x, batch, rows, num_workers);
                detail::apply_statistics(
                    state,
                    detail::scheduled_alpha(recipe.alpha0, step, recipe.total_steps),
                    last,
                    num_workers
                );
                if ((step + 1) % detail::kReviveEvery == 0) {
                    nsplit += detail::revive(state, last);
                }
                ++step;
            }
            if (iteration == 0) {
                graph.complete_query_sums();
            }
            if (last) {
                detail::close_means(state, num_workers);
            }
            double shift = 0.0;
            for (size_t cluster = 0; cluster < k; ++cluster) {
                shift += clustering::detail::euclidean_sqr_f64(
                    state.centroids.data() + cluster * d,
                    iteration_start.data() + cluster * d,
                    d
                );
            }
            stats.push_back({iteration + 1, objective, shift, nsplit});
            if (verbose) {
                std::cout << detail::kName << " iteration " << iteration + 1 << "/" << niter
                          << " | objective=" << objective << " | shift=" << shift
                          << " | splits=" << nsplit << '\n';
            }
        }

        std::vector<PID> final_assignments;
        std::vector<float> final_distances;
        double objective = 0.0;
        if (assign_all) {
            final_assignments.resize(n);
            final_distances.resize(n);
            // The approximate pass reuses the training graph and its query sums.
            detail::assign_with_mode(
                final_assignment,
                x,
                state.centroids.data(),
                n,
                d,
                k,
                false,
                num_workers,
                final_assignments.data(),
                final_distances.data(),
                [&] {
                    graph.rebuild(state.centroids.data());
                    graph.search(
                        x,
                        nullptr,
                        n,
                        final_assignments.data(),
                        final_distances.data(),
                        nullptr
                    );
                }
            );
            objective =
                std::accumulate(final_distances.begin(), final_distances.end(), 0.0);
        }

        // Commit: the previous results stay intact if anything above throws.
        centroids.swap(state.centroids);
        assignments.swap(final_assignments);
        distances.swap(final_distances);
        iteration_stats.swap(stats);
        final_obj = objective;
        graph_cache_.clear();
        trained_ = true;
    }

    // Assigns validated points to the current centroids without training state, so
    // each point's result is independent of the batch. Requires n > 0 and outputs
    // that do not overlap x or centroids.
    void assign_points(
        size_t n,
        const float* x,
        uint32_t num_workers,
        PID* point_assignments,
        float* point_distances
    ) const {
        detail::assign_with_mode(
            final_assignment,
            x,
            centroids.data(),
            n,
            d,
            k,
            false,
            num_workers,
            point_assignments,
            point_distances,
            [&] {
                graph_cache_.assign(
                    d,
                    k,
                    centroids.data(),
                    detail::graph_settings(*this),
                    x,
                    n,
                    point_assignments,
                    point_distances
                );
            }
        );
    }

    void validate_parameters() const {
        detail::validate_common_parameters(
            detail::kName, d, k, niter, "niter", num_threads, final_assignment
        );
        detail::validate_graph_parameters(
            detail::kName,
            d,
            k,
            max_degree,
            ef_construction,
            ef_search,
            graph_build_iterations
        );
    }

    [[nodiscard]] uint32_t resolved_num_workers() const {
        return static_cast<uint32_t>(resolve_num_threads(num_threads));
    }

    bool trained_ = false;
    // Centroid graph for assign(), built on first use after training.
    detail::GraphCache graph_cache_;
};

}  // namespace rabitqlib::qgemeans
