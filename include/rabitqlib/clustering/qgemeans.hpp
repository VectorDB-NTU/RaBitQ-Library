#pragma once

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/clustering/detail/kmeans.hpp"  // IWYU pragma: export
#include "rabitqlib/clustering/qg_assigner.hpp"
#include "rabitqlib/clustering/qgkmeans.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::qgemeans {

using clustering::FinalAssignmentMode;

struct QGEMeansParameters {
    // Passes over the training vectors.
    size_t epochs = 8;
    // Vectors per mini-batch; 0 selects the batch-size law.
    size_t batch_size = 0;
    bool verbose = false;
    uint32_t seed = 42;
    FinalAssignmentMode final_assignment = FinalAssignmentMode::Approximate;

    // SymphonyQG assignment parameters, as in QGKMeansParameters.
    uint32_t graph_degree = 32;
    uint32_t ef_build = 240;
    uint32_t ef_search = 16;
    // Graph-building passes after PiPNN initialization; the last refines the graph.
    uint32_t graph_build_iterations = 1;
    uint32_t num_threads = 0;
};

namespace detail {

// Mini-batch e-means recipe of Algorithm 1 in arXiv 2611.09501, with the constants of
// the FAISS reference implementation (faiss/contrib/e_means, MIT license): the
// throughput batch-size law (d > 16), the step law, and a cosine schedule.
// Prefixes error messages and progress output.
constexpr const char* kName = "QGEMeans";
constexpr double kAlphaMinFactor = 0.1;
constexpr size_t kReviveEvery = 5;
constexpr float kInitialMass = 1.0F / 65536.0F;
constexpr float kSplitEpsilon = 1e-4F;
constexpr float kUnusedThreshold = 0.1F;
constexpr double kBatchC = 0.035134;
constexpr double kBatchK = 0.134749;
constexpr double kBatchN = 0.876473;
constexpr double kBatchT = 0.813344;
constexpr double kAlphaC = 0.8244;
constexpr double kAlphaK = -0.4617;
constexpr double kAlphaB = 0.9003;
constexpr double kAlphaN = -0.4498;
constexpr double kAlphaT = -0.1057;
constexpr double kPi = 3.14159265358979323846;

// Nearest power of two to value, 1 below 1.
[[nodiscard]] inline size_t round_power_of_two(double value) {
    if (value <= 1) {
        return 1;
    }
    const size_t low = size_t{1} << static_cast<int>(std::floor(std::log2(value)));
    const auto high = static_cast<double>(2 * low);
    return value / static_cast<double>(low) < high / value ? low : 2 * low;
}

struct Recipe {
    size_t batch_size;
    double alpha0;
    size_t steps_per_epoch;
    size_t total_steps;
};

// Requires n >= 1 and epochs >= 1.
[[nodiscard]] inline Recipe resolve_recipe(
    size_t n, size_t k, size_t epochs, size_t batch_size
) {
    const auto nd = static_cast<double>(n);
    const auto kd = static_cast<double>(k);
    const auto td = static_cast<double>(epochs);
    if (batch_size == 0) {
        const double law =
            kBatchC * std::pow(kd, kBatchK) * std::pow(nd, kBatchN) * std::pow(td, kBatchT);
        batch_size = std::min(round_power_of_two(law), round_power_of_two(nd / 2));
    }
    batch_size = std::max<size_t>(1, std::min(batch_size, n));
    double alpha0 = kAlphaC * std::pow(kd, kAlphaK) *
                    std::pow(static_cast<double>(batch_size), kAlphaB) *
                    std::pow(nd, kAlphaN) * std::pow(td, kAlphaT);
    alpha0 = std::min(std::max(alpha0, 1e-4), 1.0);
    const size_t steps = (n + batch_size - 1) / batch_size;
    return {batch_size, alpha0, steps, steps * epochs};
}

[[nodiscard]] inline double scheduled_alpha(
    double alpha0, size_t step, size_t total_steps
) {
    const double fraction = static_cast<double>(std::min(step, total_steps)) /
                            static_cast<double>(std::max<size_t>(1, total_steps));
    const double cosine = 0.5 * (1.0 + std::cos(kPi * fraction));
    return alpha0 * (kAlphaMinFactor + ((1.0 - kAlphaMinFactor) * cosine));
}

[[nodiscard]] inline uint32_t epoch_seed(uint64_t seed, uint64_t epoch) {
    return static_cast<uint32_t>((seed * 1000003ULL + epoch) % (1ULL << 32));
}

// The QGKMeans settings that build and search the same centroid graph.
[[nodiscard]] inline qgkmeans::QGKMeansParameters graph_parameters(
    const QGEMeansParameters& cp
) {
    qgkmeans::QGKMeansParameters graph;
    graph.seed = cp.seed;
    graph.graph_degree = cp.graph_degree;
    graph.ef_build = cp.ef_build;
    graph.ef_search = cp.ef_search;
    graph.graph_build_iterations = cp.graph_build_iterations;
    graph.num_threads = cp.num_threads;
    return graph;
}

// Raw-vector centroid graph for one training call: rebuilt from the current
// centroids, then searched for the nearest centroid of each listed row, without
// hints. The query-sum cache is filled during the first epoch, which visits
// every row once, and reused afterwards.
class BatchAssignment {
    qgkmeans::QGAssigner assigner_;
    std::vector<float> query_sums_;
    bool sums_ready_ = false;

   public:
    BatchAssignment(size_t d, size_t k, size_t n, const QGEMeansParameters& cp)
        : assigner_(
              d,
              k,
              cp.graph_degree,
              cp.ef_build,
              cp.ef_search,
              cp.graph_build_iterations,
              cp.num_threads,
              cp.seed,
              METRIC_L2,
              0,
              kName
          ) {
        if (assigner_.graph_->padded_dim() >= qgkmeans::detail::kMinQuerySumDim) {
            query_sums_.resize(n);
        }
    }

    void rebuild(const float* centroids) { assigner_.rebuild_centroid_graph(centroids); }

    // Requires a graph rebuilt from centroids. Null rows assign rows 0 to count - 1.
    void search(
        const float* centroids,
        const float* x,
        const size_t* rows,
        size_t count,
        PID* assignments,
        float* distances
    ) {
        const qgkmeans::detail::QuerySumCache cache(
            query_sums_.empty() ? nullptr : query_sums_.data(), sums_ready_
        );
        assigner_.search_points(
            centroids,
            x,
            count,
            assignments,
            distances,
            {},
            nullptr,
            cache,
            assigner_.scratch_,
            rows
        );
    }

    // Called after a pass that searched every row.
    void complete_query_sums() { sums_ready_ = true; }
};

}  // namespace detail

/**
 * Mini-batch e-means with SymphonyQG graph assignment. Each mini-batch is assigned
 * through a raw-vector graph over the current centroids, rebuilt before every
 * batch; centroids follow the e-means mass-weighted update and revival of unused
 * centroids. train() then assigns every vector to the final centroids.
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
        clustering::detail::assign_trained(
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
    // One entry per epoch: obj sums the mini-batch assignment distances of that
    // epoch, shift is the squared centroid displacement over it, and nsplit counts
    // revived centroids.
    std::vector<clustering::KMeansIterationStats> iteration_stats;
    // Objective of the final returned assignments against the final centroids.
    double final_obj = 0.0;

   private:
    // Training state of one fit; committed to the public members at the end. The
    // vectors after split_rng are scratch reused across mini-batches.
    struct State {
        std::vector<float> centroids;
        std::vector<float> mass;
        std::vector<double> pass_counts;
        std::vector<float> pass_sums;
        std::vector<float> batch_sums;
        std::vector<float> batch_counts;
        std::vector<PID> batch_assignments;
        std::vector<float> batch_distances;
        std::vector<size_t> offsets;
        std::vector<size_t> members;
        std::vector<double> cluster_objectives;
        std::mt19937_64 split_rng;
        std::vector<size_t> cursor;
        std::vector<size_t> unused;
        std::vector<double> draws;
        std::vector<std::pair<double, size_t>> keys;
    };

    void fit(size_t n, const float* x, bool assign_all) {
        validate_parameters();
        if (x == nullptr) {
            throw std::invalid_argument(std::string(detail::kName) + " x must not be null");
        }
        if (n < k) {
            throw std::invalid_argument(std::string(detail::kName) + " requires n >= k");
        }
        const uint32_t num_workers = resolved_num_workers();
        clustering::detail::validate_points(detail::kName, n, d, x, num_workers, "x");

        const detail::Recipe recipe = detail::resolve_recipe(n, k, epochs, batch_size);
        detail::BatchAssignment graph(d, k, n, *this);
        State state;
        initialize(state, n, x, recipe.batch_size);

        std::vector<clustering::KMeansIterationStats> stats;
        stats.reserve(epochs);
        std::vector<size_t> order(n);
        std::vector<float> epoch_start(state.centroids.size());
        size_t step = 0;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            const bool last = epoch + 1 == epochs;
            std::copy(state.centroids.begin(), state.centroids.end(), epoch_start.begin());
            std::fill(state.pass_counts.begin(), state.pass_counts.end(), 0.0);
            if (last) {
                state.pass_sums.assign(k * d, 0.0F);
            }
            std::iota(order.begin(), order.end(), size_t{0});
            std::mt19937_64 rng(detail::epoch_seed(seed, epoch));
            std::shuffle(order.begin(), order.end(), rng);
            double objective = 0.0;
            size_t nsplit = 0;
            for (size_t begin = 0; begin < n; begin += recipe.batch_size) {
                const size_t rows = std::min(recipe.batch_size, n - begin);
                const size_t* batch = order.data() + begin;
                graph.rebuild(state.centroids.data());
                graph.search(
                    state.centroids.data(),
                    x,
                    batch,
                    rows,
                    state.batch_assignments.data(),
                    state.batch_distances.data()
                );
                objective += accumulate(state, x, batch, rows, num_workers);
                apply_statistics(
                    state,
                    detail::scheduled_alpha(recipe.alpha0, step, recipe.total_steps),
                    last,
                    num_workers
                );
                if ((step + 1) % detail::kReviveEvery == 0) {
                    nsplit += revive_unused(state);
                }
                ++step;
            }
            if (epoch == 0) {
                graph.complete_query_sums();
            }
            if (last) {
                close_means(state, num_workers);
            }
            double shift = 0.0;
            for (size_t cluster = 0; cluster < k; ++cluster) {
                shift += clustering::detail::euclidean_sqr_f64(
                    state.centroids.data() + cluster * d,
                    epoch_start.data() + cluster * d,
                    d
                );
            }
            stats.push_back({epoch + 1, objective, shift, nsplit});
            if (verbose) {
                std::cout << detail::kName << " epoch " << epoch + 1 << "/" << epochs
                          << " | batch_size=" << recipe.batch_size
                          << " | objective=" << objective << " | shift=" << shift
                          << " | revived=" << nsplit << '\n';
            }
        }

        std::vector<PID> final_assignments;
        std::vector<float> final_distances;
        double objective = 0.0;
        if (assign_all) {
            final_assignments.resize(n);
            final_distances.resize(n);
            // The approximate pass reuses the training graph and its query sums.
            clustering::detail::assign_with_mode(
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
                        state.centroids.data(),
                        x,
                        nullptr,
                        n,
                        final_assignments.data(),
                        final_distances.data()
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

    // Initial centroids are k distinct random rows; every centroid starts with the
    // reference's small initial mass.
    void initialize(State& state, size_t n, const float* x, size_t batch) const {
        state.centroids.resize(k * d);
        {
            std::mt19937_64 rng(seed);
            std::vector<size_t> rows(n);
            std::iota(rows.begin(), rows.end(), size_t{0});
            for (size_t cluster = 0; cluster < k; ++cluster) {
                std::uniform_int_distribution<size_t> pick(cluster, n - 1);
                std::swap(rows[cluster], rows[pick(rng)]);
                std::copy_n(x + rows[cluster] * d, d, state.centroids.data() + cluster * d);
            }
        }
        state.mass.assign(k, detail::kInitialMass);
        state.split_rng.seed(seed);
        state.pass_counts.assign(k, 0.0);
        state.batch_sums.resize(k * d);
        state.batch_counts.resize(k);
        state.batch_assignments.resize(batch);
        state.batch_distances.resize(batch);
        state.offsets.resize(k + 1);
        state.members.resize(batch);
        state.cluster_objectives.resize(k);
        state.cursor.resize(k);
        state.unused.reserve(k);
        state.draws.resize(k);
        state.keys.reserve(k);
    }

    // Per-centroid sums and counts of one batch, in float32 as the reference, over
    // members in batch order. Returns the batch's assignment objective.
    double accumulate(
        State& state, const float* x, const size_t* rows, size_t count, uint32_t num_workers
    ) const {
        auto& offsets = state.offsets;
        std::fill(offsets.begin(), offsets.end(), size_t{0});
        for (size_t i = 0; i < count; ++i) {
            ++offsets[state.batch_assignments[i] + 1];
        }
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        auto& cursor = state.cursor;
        std::copy(offsets.begin(), offsets.end() - 1, cursor.begin());
        for (size_t i = 0; i < count; ++i) {
            state.members[cursor[state.batch_assignments[i]]++] = i;
        }
        static_cast<void>(num_workers);
#pragma omp parallel for num_threads(num_workers) schedule(dynamic, 16)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(k); ++index) {
            const auto cluster = static_cast<size_t>(index);
            float* sums = state.batch_sums.data() + cluster * d;
            std::fill_n(sums, d, 0.0F);
            double cluster_objective = 0.0;
            for (size_t m = offsets[cluster]; m < offsets[cluster + 1]; ++m) {
                const size_t i = state.members[m];
                const float* row = x + rows[i] * d;
                for (size_t dim = 0; dim < d; ++dim) {
                    sums[dim] += row[dim];
                }
                cluster_objective += state.batch_distances[i];
            }
            state.cluster_objectives[cluster] = cluster_objective;
            state.batch_counts[cluster] =
                static_cast<float>(offsets[cluster + 1] - offsets[cluster]);
        }
        // Summed in cluster order, so the objective does not depend on scheduling.
        return std::accumulate(
            state.cluster_objectives.begin(), state.cluster_objectives.end(), 0.0
        );
    }

    // Algorithm 1, line 8: mass' = (1 - alpha) mass + alpha count, and each
    // centroid moves toward its batch mean by alpha count / mass'.
    void apply_statistics(
        State& state, double alpha_value, bool last, uint32_t num_workers
    ) const {
        const auto alpha = static_cast<float>(alpha_value);
        const auto decay = static_cast<float>(1.0 - alpha_value);
        const bool replace_mass = alpha_value == 1.0;
        static_cast<void>(num_workers);
#pragma omp parallel for num_threads(num_workers) schedule(static)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(k); ++index) {
            const auto cluster = static_cast<size_t>(index);
            const float count = state.batch_counts[cluster];
            state.pass_counts[cluster] += count;
            const float* sums = state.batch_sums.data() + cluster * d;
            if (last) {
                float* pass = state.pass_sums.data() + cluster * d;
                for (size_t dim = 0; dim < d; ++dim) {
                    pass[dim] += sums[dim];
                }
            }
            if (!replace_mass) {
                state.mass[cluster] *= decay;
            }
            if (count <= 0) {
                continue;
            }
            const float retained = replace_mass ? 0.0F : state.mass[cluster];
            const float mass = retained + (alpha * count);
            const float weight = alpha * count / mass;
            float* centroid = state.centroids.data() + cluster * d;
            for (size_t dim = 0; dim < d; ++dim) {
                const float mean = sums[dim] / count;
                centroid[dim] = weight == 1.0F
                                    ? mean
                                    : centroid[dim] + (weight * (mean - centroid[dim]));
            }
            state.mass[cluster] = mass;
        }
    }

    // Algorithm 1, line 9: centroids whose mass falls below a tenth of the mean are
    // replaced by splitting donors drawn in proportion to their mass (Gumbel top-k).
    [[nodiscard]] size_t revive_unused(State& state) const {
        auto& mass = state.mass;
        double total = 0.0;
        for (const float value : mass) {
            total += value;
        }
        const float threshold =
            static_cast<float>(total) * detail::kUnusedThreshold / static_cast<float>(k);
        auto& unused = state.unused;
        unused.clear();
        size_t donors = 0;
        for (size_t cluster = 0; cluster < k; ++cluster) {
            if (mass[cluster] < threshold) {
                unused.push_back(cluster);
            } else if (mass[cluster] > 0) {
                ++donors;
            }
        }
        const size_t nsplit = std::min(unused.size(), donors);
        if (nsplit == 0) {
            return 0;
        }
        // Drawn only when something splits, as the reference does.
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        auto& draws = state.draws;
        for (auto& value : draws) {
            value = std::max(uniform(state.split_rng), std::numeric_limits<double>::min());
        }
        auto& keys = state.keys;
        keys.clear();
        for (size_t cluster = 0; cluster < k; ++cluster) {
            if (mass[cluster] >= threshold && mass[cluster] > 0) {
                keys.emplace_back(
                    std::log(mass[cluster]) - std::log(-std::log(draws[cluster])), cluster
                );
            }
        }
        std::partial_sort(
            keys.begin(),
            keys.begin() + static_cast<std::ptrdiff_t>(nsplit),
            keys.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
        for (size_t split = 0; split < nsplit; ++split) {
            const size_t source = keys[split].second;
            const size_t target = unused[split];
            float* donor = state.centroids.data() + source * d;
            float* child = state.centroids.data() + target * d;
            bool collapsed = true;
            for (size_t dim = 0; dim < d; ++dim) {
                const float offset = detail::kSplitEpsilon * (dim % 2 == 0 ? 1.0F : -1.0F);
                const float value = donor[dim];
                donor[dim] = value * (1.0F - offset);
                child[dim] = value * (1.0F + offset);
                collapsed = collapsed && donor[dim] == child[dim];
            }
            if (collapsed) {
                // A zero donor splits into equal halves; separate the child.
                const float next =
                    std::nextafter(child[0], std::numeric_limits<float>::infinity());
                child[0] =
                    std::isfinite(next)
                        ? next
                        : std::nextafter(child[0], -std::numeric_limits<float>::infinity());
            }
            const float half = mass[source] / 2.0F;
            mass[source] = half;
            mass[target] = half;
        }
        return nsplit;
    }

    // Closing means: centroids assigned during the last epoch become the mean of
    // their last-epoch members; the others keep their positions.
    void close_means(State& state, uint32_t num_workers) const {
        static_cast<void>(num_workers);
#pragma omp parallel for num_threads(num_workers) schedule(static)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(k); ++index) {
            const auto cluster = static_cast<size_t>(index);
            if (state.pass_counts[cluster] > 0) {
                const auto count = static_cast<float>(state.pass_counts[cluster]);
                for (size_t dim = 0; dim < d; ++dim) {
                    state.centroids[cluster * d + dim] =
                        state.pass_sums[cluster * d + dim] / count;
                }
            }
        }
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
        clustering::detail::assign_with_mode(
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
                    detail::graph_parameters(*this),
                    x,
                    n,
                    point_assignments,
                    point_distances
                );
            }
        );
    }

    void validate_parameters() const {
        clustering::detail::validate_common_parameters(
            detail::kName, d, k, epochs, "epochs", num_threads, final_assignment
        );
        qgkmeans::detail::validate_graph_parameters(
            detail::kName, d, k, graph_degree, ef_build, ef_search, graph_build_iterations
        );
    }

    [[nodiscard]] uint32_t resolved_num_workers() const {
        return static_cast<uint32_t>(resolve_num_threads(num_threads));
    }

    bool trained_ = false;
    // Centroid graph for assign(), built on first use after training.
    qgkmeans::detail::CentroidGraphCache graph_cache_;
};

}  // namespace rabitqlib::qgemeans
