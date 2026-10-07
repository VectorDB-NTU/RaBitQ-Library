#pragma once

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "rabitqlib/clustering/detail/kmeans.hpp"
#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"
#include "rabitqlib/utils/visited_set.hpp"

namespace rabitqlib::qgkmeans {

using clustering::FinalAssignmentMode;
using QGKMeansIterationStats = clustering::KMeansIterationStats;

namespace detail {

using clustering::detail::exact_assign;
using clustering::detail::should_terminate_by_objective;
class QGAssignment;

}  // namespace detail

struct QGKMeansParameters {
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

    // SymphonyQG assignment parameters.
    uint32_t graph_degree = 32;
    uint32_t ef_build = 240;
    uint32_t ef_search = 16;
    /// Graph-building passes after PiPNN initialization; the last refines the graph.
    uint32_t graph_build_iterations = 1;
    uint32_t num_threads = 0;
    /// Centroid storage in QG: 0 for raw float32, or 4/8-bit RaBitQ.
    uint32_t quantization_bits = 0;
};

/**
 * Builds a SymphonyQG over a fixed number of centroids and assigns vectors to
 * them. The graph allocation and per-thread search scratch are reused between
 * calls, and the previous assignment is used as the next call's search hint.
 */
class QGAssigner {
    friend class detail::QGAssignment;

   private:
    struct ThreadScratch {
        std::vector<float> rotated_query;
        std::vector<float> estimated_distances;
        std::vector<float> lookup_table;
        BatchQuery<float> batch_query;
        buffer::SearchBuffer<float> search_pool;
        buffer::SearchBuffer<float> result_pool;

        ThreadScratch(size_t padded_dim, size_t M, size_t ef_search)
            : rotated_query(padded_dim)
            , estimated_distances(M)
            , lookup_table(padded_dim << 2)
            , search_pool(ef_search)
            , result_pool(1) {}
    };

   public:
    QGAssigner(
        size_t d,
        size_t k,
        uint32_t M,
        uint32_t ef_construction,
        uint32_t ef_search,
        uint32_t niter,
        uint32_t num_threads = 0,
        uint32_t seed = 42,
        MetricType metric_type = METRIC_L2,
        uint32_t quantization_bits = 0
    )
        : d_(d)
        , k_(k)
        , M_(M)
        , ef_construction_(ef_construction)
        , ef_search_(ef_search)
        , niter_(niter)
        , num_threads_(num_threads)
        , seed_(seed) {
        validate_configuration();
        num_threads_ = static_cast<uint32_t>(resolve_num_threads(num_threads_));

        graph_ = std::make_unique<symqg::QuantizedGraph<float>>(
            k_, d_, M_, metric_type, RotatorType::FhtKacRotator, quantization_bits, seed_
        );
        graph_->set_ef(ef_search_);

        scratch_.reserve(num_threads_);
        for (uint32_t thread = 0; thread < num_threads_; ++thread) {
            scratch_.emplace_back(
                std::make_unique<ThreadScratch>(graph_->padded_dim(), M_, ef_search_)
            );
        }
    }

    // Optional n-element cache: fill it on the first call, then reuse it only
    // for the same input vectors and this assigner. A null pointer disables it.
    void assign(
        const float* centroids,
        const float* x,
        size_t n,
        PID* labels,
        float* distances,
        float* query_sums = nullptr,
        bool sums_ready = false
    ) {
        if (centroids == nullptr || x == nullptr || labels == nullptr ||
            distances == nullptr) {
            throw std::invalid_argument(
                "QGAssigner input and output pointers must not be null"
            );
        }

        if (n > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / d_) {
            throw std::invalid_argument("QGAssigner input dimensions overflow ptrdiff_t");
        }

        sums_ready = query_sums != nullptr && sums_ready;
        if (builder_) {
            builder_->reset(centroids);
        } else {
            // Centroid graphs have much less work than the training-vector batch.
            // Amortize their short phases without reducing assignment parallelism.
            constexpr size_t kMinVerticesPerBuildThread = 128;
            const size_t build_threads = std::min<size_t>(
                num_threads_, std::max<size_t>(1, k_ / kMinVerticesPerBuildThread)
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
        if (niter_ == 1) {
            builder_->build();
        } else {
            builder_->build(niter_);
        }
        const auto distance_func = graph_->metric_type() == METRIC_IP
                                       ? dot_product_dis<float>
                                       : euclidean_sqr<float>;

        // Stable grouping changes only the order of independent full searches.
        std::vector<uint32_t> query_order;
        if (previous_labels_.size() == n && n <= std::numeric_limits<uint32_t>::max()) {
            query_order.resize(n);
            std::vector<size_t> offsets(k_ + 1, 0);
            for (PID label : previous_labels_) {
                ++offsets[label + 1];
            }
            std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
            for (size_t point = 0; point < n; ++point) {
                query_order[offsets[previous_labels_[point]]++] =
                    static_cast<uint32_t>(point);
            }
        }
        std::vector<std::exception_ptr> errors(num_threads_);
#pragma omp parallel for num_threads(num_threads_) schedule(dynamic, 64)
        for (std::ptrdiff_t loop_index = 0; loop_index < static_cast<std::ptrdiff_t>(n);
             ++loop_index) {
            const size_t i = query_order.empty()
                                 ? static_cast<size_t>(loop_index)
                                 : query_order[static_cast<size_t>(loop_index)];
            const int thread_id = omp_get_thread_num();
            if (errors[static_cast<size_t>(thread_id)]) {
                continue;
            }
            try {
                ThreadScratch& scratch = *scratch_[static_cast<size_t>(thread_id)];
                // Keep visited storage with the worker, as in QuantizedGraph::search.
                thread_local VisitedSet visited;
                thread_local size_t visited_size = 0;
                thread_local size_t visited_capacity = 0;
                const size_t capacity =
                    std::min(static_cast<size_t>(ef_search_) * ef_search_, k_ / 10);
                if (visited_size != k_ || visited_capacity != capacity) {
                    visited.initialize(k_, capacity);
                    visited_size = k_;
                    visited_capacity = capacity;
                }
                const PID hint =
                    previous_labels_.size() == n ? previous_labels_[i] : kPidMax;
                graph_->search_with_scratch(
                    x + i * d_,
                    1,
                    labels + i,
                    distances + i,
                    scratch.rotated_query.data(),
                    scratch.estimated_distances.data(),
                    scratch.lookup_table.data(),
                    scratch.batch_query,
                    scratch.search_pool,
                    scratch.result_pool,
                    visited,
                    hint,
                    sums_ready ? query_sums + i : nullptr
                );
                if (query_sums != nullptr && !sums_ready) {
                    query_sums[i] = scratch.batch_query.k1xsumq();
                }
                if (graph_->is_quantized()) {
                    // Keep objectives and returned distances relative to the original
                    // centroids; graph search scores are quantized estimates.
                    distances[i] =
                        distance_func(x + i * d_, centroids + labels[i] * d_, d_);
                }
            } catch (...) {
                errors[static_cast<size_t>(thread_id)] = std::current_exception();
            }
        }

        for (const auto& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }

        previous_labels_.assign(labels, labels + n);
    }

    void clear_hints() { previous_labels_.clear(); }

   private:
    void validate_configuration() const {
        if (num_threads_ > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("QGAssigner num_threads must fit in an OpenMP int");
        }
        if (d_ < 64 || d_ > rotator_impl::FhtKacRotator::kMaxDim) {
            throw std::invalid_argument(
                "QGKMeans currently requires dimensions in the range [64, 65536]"
            );
        }
        if (M_ == 0 || M_ % fastscan::kBatchSize != 0) {
            throw std::invalid_argument(
                "QGKMeans graph degree must be a positive multiple of 32"
            );
        }
        if (k_ <= M_) {
            throw std::invalid_argument(
                "QGKMeans requires more centroids than the graph degree"
            );
        }
        if (ef_construction_ == 0 || ef_search_ == 0) {
            throw std::invalid_argument("QGKMeans ef_build and ef_search must be positive");
        }
        if (niter_ == 0) {
            throw std::invalid_argument("QGKMeans graph_build_iterations must be positive");
        }
    }

    size_t d_;
    size_t k_;
    uint32_t M_;
    uint32_t ef_construction_;
    uint32_t ef_search_;
    uint32_t niter_;
    uint32_t num_threads_;
    uint32_t seed_;
    std::unique_ptr<symqg::QuantizedGraph<float>> graph_;
    std::unique_ptr<symqg::QGBuilder> builder_;
    std::vector<std::unique_ptr<ThreadScratch>> scratch_;
    std::vector<PID> previous_labels_;
};

namespace detail {

// Retain graph-specific state for one synchronous training call.
class QGAssignment {
    size_t d_;
    const float* x_;
    size_t n_;
    uint32_t num_threads_;
    MetricType metric_type_;
    bool quantized_;
    QGAssigner assigner_;
    bool cache_query_sums_;
    std::vector<float> query_sums_;
    bool query_sums_ready_ = false;

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
        uint32_t num_threads,
        const QGKMeansParameters& cp
    )
        : d_(d)
        , x_(x)
        , n_(n)
        , num_threads_(num_threads)
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
        // Inputs and the assigner's rotator are immutable for this training call.
        , cache_query_sums_(
              assigner_.graph_->padded_dim() >= 1024 &&
              (cp.niter > 1 || cp.final_assignment != FinalAssignmentMode::Exact)
          ) {}

    void assign(const float* centroids, PID* labels, float* distances, bool final = false) {
        if (cache_query_sums_ && query_sums_.empty()) {
            // Allocate after centroid initialization releases its point permutation.
            query_sums_.resize(n_);
        }
        std::vector<PID> previous_labels;
        if (final && quantized_) {
            previous_labels.assign(labels, labels + n_);
        }
        assigner_.assign(
            centroids,
            x_,
            n_,
            labels,
            distances,
            query_sums_.empty() ? nullptr : query_sums_.data(),
            query_sums_ready_
        );
        query_sums_ready_ = true;
        if (previous_labels.empty()) {
            return;
        }

        // Protect the returned assignments without changing exploration during
        // Lloyd iterations: uphill approximate moves can improve final centroids.
        const auto distance_func =
            metric_type_ == METRIC_IP ? dot_product_dis<float> : euclidean_sqr<float>;
#pragma omp parallel for num_threads(num_threads_) schedule(static)
        for (std::ptrdiff_t loop_index = 0; loop_index < static_cast<std::ptrdiff_t>(n_);
             ++loop_index) {
            const size_t point = static_cast<size_t>(loop_index);
            const PID previous = previous_labels[point];
            if (previous == labels[point]) {
                continue;
            }
            const float distance =
                distance_func(x_ + point * d_, centroids + previous * d_, d_);
            if (distance < distances[point]) {
                labels[point] = previous;
                distances[point] = distance;
            }
        }
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
