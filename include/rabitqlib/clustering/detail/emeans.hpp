#pragma once

// The mini-batch e-means algorithm QGEMeans trains with, independent of how each
// mini-batch is assigned: Algorithm 1 of arXiv 2611.09501, with the constants of
// the FAISS reference implementation (faiss/contrib/e_means at b14b3d693, MIT
// license): the throughput batch-size law (d > 16), the step law, a cosine
// schedule and cost-based revival. EMeans runs it over a pluggable assignment, as
// LloydKMeans does for Lloyd's algorithm. Also QGEMeans' copy of the LloydKMeans
// validation and assignment checks, kept separate so LloydKMeans is unchanged.

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

#include "rabitqlib/clustering/detail/kmeans.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/input_checks.hpp"
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

// Checks the settings every clustering class shares. iterations is niter,
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

constexpr double kAlphaMinFactor = 0.1;
constexpr size_t kReviveEvery = 5;
constexpr float kInitialMass = 1.0F / 65536.0F;
// Cost-based revival: a centroid may move when its removal cost R, the EMA of
// d2 - d1 over its vectors, is at most zero or below kRemovalThreshold times the
// median, and only if its donor's split gain kSplitGain * |U| exceeds R.
constexpr double kRemovalThreshold = 0.3;
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
constexpr double kSplitGain = 2.0 / kPi;

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
    size_t steps_per_iteration;
    size_t total_steps;
};

// Requires n >= 1 and niter >= 1.
[[nodiscard]] inline Recipe resolve_recipe(
    size_t n, size_t k, size_t niter, size_t batch_size
) {
    const auto nd = static_cast<double>(n);
    const auto kd = static_cast<double>(k);
    const auto td = static_cast<double>(niter);
    if (batch_size == 0) {
        const double law =
            kBatchC * std::pow(kd, kBatchK) * std::pow(nd, kBatchN) * std::pow(td, kBatchT);
        // Rounding is monotone, so capping first gives the same power of two and
        // keeps an extreme law within the shift range of round_power_of_two.
        batch_size = round_power_of_two(std::min(law, nd / 2));
    }
    batch_size = std::max<size_t>(1, std::min(batch_size, n));
    double alpha0 = kAlphaC * std::pow(kd, kAlphaK) *
                    std::pow(static_cast<double>(batch_size), kAlphaB) *
                    std::pow(nd, kAlphaN) * std::pow(td, kAlphaT);
    alpha0 = std::min(std::max(alpha0, 1e-4), 1.0);
    const size_t steps = (n + batch_size - 1) / batch_size;
    return {batch_size, alpha0, steps, steps * niter};
}

[[nodiscard]] inline double scheduled_alpha(
    double alpha0, size_t step, size_t total_steps
) {
    const double fraction = static_cast<double>(std::min(step, total_steps)) /
                            static_cast<double>(std::max<size_t>(1, total_steps));
    const double cosine = 0.5 * (1.0 + std::cos(kPi * fraction));
    return alpha0 * (kAlphaMinFactor + ((1.0 - kAlphaMinFactor) * cosine));
}

// A pass seed equal to the initialization seed moves by the golden-ratio
// constant, so a pass never replays the initialization draw.
[[nodiscard]] inline uint32_t iteration_seed(uint64_t seed, uint64_t iteration) {
    auto value = static_cast<uint32_t>((seed * 1000003ULL + iteration) % (1ULL << 32));
    if (value == static_cast<uint32_t>(seed)) {
        value += 0x9E3779B9U;
    }
    return value;
}

// Uniform double in (0, 1) from 53 random bits, so the axis draws do not depend on
// the standard library. The initial rows and pass orders use std distributions,
// so results are reproducible for one toolchain only.
[[nodiscard]] inline double open_uniform(std::mt19937_64& rng) {
    constexpr double kScale = 1.0 / 9007199254740992.0;  // 2^-53
    return (static_cast<double>(rng() >> 11) + 0.5) * kScale;
}

// Training state of one fit. removal holds k rows [U / axis_scale | R]: the Oja
// long-axis sums U and the removal cost R, the EMA of d2 - d1 over each
// centroid's vectors. The batch_* inputs are filled by the caller's assignment of
// each mini-batch; the vectors after split_rng are scratch reused across
// mini-batches.
struct EMeansState {
    size_t d = 0;
    size_t k = 0;
    std::vector<float> centroids;
    std::vector<float> mass;
    std::vector<double> pass_counts;
    std::vector<float> pass_sums;
    std::vector<float> removal;
    std::vector<float> axes;
    double axis_scale = 1.0;
    std::vector<float> batch_sums;
    std::vector<float> batch_counts;
    std::vector<float> batch_removal;
    std::vector<PID> batch_assignments;
    std::vector<float> batch_distances;
    std::vector<float> batch_second_distances;
    std::vector<size_t> offsets;
    std::vector<size_t> members;
    std::vector<double> cluster_objectives;
    std::mt19937_64 split_rng;
    std::vector<size_t> cursor;
    std::vector<double> costs;
    std::vector<double> axis_norms;
    std::vector<size_t> candidates;
    std::vector<size_t> donors;
    std::vector<double> sorted_costs;
};

// Initial centroids are k distinct random rows; every centroid starts with the
// reference's small initial mass.
inline void initialize(
    EMeansState& state,
    size_t d,
    size_t k,
    size_t n,
    const float* x,
    size_t batch,
    uint32_t seed
) {
    state.d = d;
    state.k = k;
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
    state.mass.assign(k, kInitialMass);
    state.split_rng.seed(seed);
    state.pass_counts.assign(k, 0.0);
    state.removal.assign(k * (d + 1), 0.0F);
    state.axis_scale = 1.0;
    // Random unit initial axes, in the reference's draw order: k rows of d
    // standard normals (Box-Muller), each normalized in double.
    state.axes.resize(k * d);
    std::vector<double> axis(d);
    for (size_t cluster = 0; cluster < k; ++cluster) {
        double norm2 = 0.0;
        for (size_t dim = 0; dim < d; ++dim) {
            const double radius = std::sqrt(-2.0 * std::log(open_uniform(state.split_rng)));
            const double angle = 2.0 * kPi * open_uniform(state.split_rng);
            axis[dim] = radius * std::cos(angle);
            norm2 += axis[dim] * axis[dim];
        }
        const double norm = std::sqrt(norm2);
        for (size_t dim = 0; dim < d; ++dim) {
            state.axes[cluster * d + dim] = static_cast<float>(axis[dim] / norm);
        }
    }
    state.batch_sums.resize(k * d);
    state.batch_counts.resize(k);
    state.batch_removal.assign(k * (d + 1), 0.0F);
    state.batch_assignments.resize(batch);
    state.batch_distances.resize(batch);
    state.batch_second_distances.resize(batch);
    state.offsets.resize(k + 1);
    state.members.resize(batch);
    state.cluster_objectives.resize(k);
    state.cursor.resize(k);
    state.costs.resize(k);
    state.axis_norms.resize(k);
    state.candidates.reserve(k);
    state.donors.reserve(k);
    state.sorted_costs.resize(k);
}

// Starts a pass: resets the pass counts, and in the last pass the pass sums that
// close the means.
inline void start_iteration(EMeansState& state, bool last) {
    std::fill(state.pass_counts.begin(), state.pass_counts.end(), 0.0);
    if (last) {
        state.pass_sums.assign(state.k * state.d, 0.0F);
    }
}

// Per-centroid sums and counts of one batch, in float32 as the reference, over
// members in batch order, and the removal statistics of cost-based revival: for
// a vector with residual y to its centroid and p = <y, axis>, p * y is added to
// U and d2 - d1 to R (zero when not finite). Returns the batch's assignment
// objective.
inline double accumulate(
    EMeansState& state,
    const float* x,
    const size_t* rows,
    size_t count,
    uint32_t num_workers
) {
    const size_t d = state.d;
    const size_t k = state.k;
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
#pragma omp parallel num_threads(num_workers)
    {
        std::vector<float> residual(d);
#pragma omp for schedule(dynamic, 16)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(k); ++index) {
            const auto cluster = static_cast<size_t>(index);
            float* sums = state.batch_sums.data() + cluster * d;
            float* removal = state.batch_removal.data() + cluster * (d + 1);
            const float* centroid = state.centroids.data() + cluster * d;
            const float* axis = state.axes.data() + cluster * d;
            std::fill_n(sums, d, 0.0F);
            std::fill_n(removal, d + 1, 0.0F);
            double cluster_objective = 0.0;
            for (size_t m = offsets[cluster]; m < offsets[cluster + 1]; ++m) {
                const size_t i = state.members[m];
                const float* row = x + rows[i] * d;
                float projection = 0.0F;
                for (size_t dim = 0; dim < d; ++dim) {
                    sums[dim] += row[dim];
                    residual[dim] = row[dim] - centroid[dim];
                    projection += residual[dim] * axis[dim];
                }
                for (size_t dim = 0; dim < d; ++dim) {
                    removal[dim] += projection * residual[dim];
                }
                const float margin =
                    state.batch_second_distances[i] - state.batch_distances[i];
                removal[d] += std::isfinite(margin) ? margin : 0.0F;
                cluster_objective += state.batch_distances[i];
            }
            state.cluster_objectives[cluster] = cluster_objective;
            state.batch_counts[cluster] =
                static_cast<float>(offsets[cluster + 1] - offsets[cluster]);
        }
    }
    // Summed in cluster order, so the objective does not depend on scheduling.
    return std::accumulate(
        state.cluster_objectives.begin(), state.cluster_objectives.end(), 0.0
    );
}

// Algorithm 1, line 8: mass' = (1 - alpha) mass + alpha count, and each
// centroid moves toward its batch mean by alpha count / mass'. The removal
// statistics take the same step: R' = (1 - alpha) R + alpha R_batch, and U
// decays through the shared axis_scale while each assigned centroid adds its
// batch term and renews its unit axis U / |U| (Oja's rule).
inline void apply_statistics(
    EMeansState& state, double alpha_value, bool last, uint32_t num_workers
) {
    const size_t d = state.d;
    const size_t k = state.k;
    const auto alpha = static_cast<float>(alpha_value);
    const auto decay = static_cast<float>(1.0 - alpha_value);
    const bool replace_mass = alpha_value == 1.0;
    double scale = state.axis_scale * (1.0 - alpha_value);
    if (!(scale > 0x1p-32)) {
        const auto factor = static_cast<float>(scale);
        for (size_t cluster = 0; cluster < k; ++cluster) {
            float* u = state.removal.data() + cluster * (d + 1);
            for (size_t dim = 0; dim < d; ++dim) {
                u[dim] *= factor;
            }
        }
        scale = 1.0;
    }
    state.axis_scale = scale;
    const auto axis_weight = static_cast<float>(alpha_value / scale);
    const float bound = max_coordinate_magnitude(d);
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
        float* removal = state.removal.data() + cluster * (d + 1);
        const float* batch_removal = state.batch_removal.data() + cluster * (d + 1);
        removal[d] = (decay * removal[d]) + (alpha * batch_removal[d]);
        if (count <= 0) {
            continue;
        }
        double norm2 = 0.0;
        for (size_t dim = 0; dim < d; ++dim) {
            removal[dim] += axis_weight * batch_removal[dim];
            norm2 += static_cast<double>(removal[dim]) * removal[dim];
        }
        const double norm = std::sqrt(norm2);
        if (norm > 0) {
            float* axis = state.axes.data() + cluster * d;
            for (size_t dim = 0; dim < d; ++dim) {
                axis[dim] = static_cast<float>(removal[dim] / norm);
            }
        }
        const float retained = replace_mass ? 0.0F : state.mass[cluster];
        const float mass = retained + (alpha * count);
        const float weight = alpha * count / mass;
        float* centroid = state.centroids.data() + cluster * d;
        for (size_t dim = 0; dim < d; ++dim) {
            const float mean = sums[dim] / count;
            const float moved =
                weight == 1.0F ? mean : centroid[dim] + (weight * (mean - centroid[dim]));
            // float32 means of points on the bound can round past it.
            centroid[dim] = std::clamp(moved, -bound, bound);
        }
        state.mass[cluster] = mass;
    }
}

// Cost-based revival (Faiss's default): centroids whose removal cost R is at
// most zero or below kRemovalThreshold times the median move, cheapest first,
// to split the donors with the longest axes |U|, longest first. Pairs move in
// that order until the first whose donor's split gain kSplitGain * |U| does not
// exceed the candidate's R; a candidate with R <= 0 always moves. A split puts
// the donor and the candidate at c -/+ s u along the donor's unit axis u, with
// s = sqrt(2 |U| / (pi N)): the half-means of a Gaussian along u. They share the
// donor's halved mass and removal statistics. In the last iteration, both lose their
// pass sums and counts, so each closes from the vectors it receives after the
// split (Faiss's rule without closing handover).
[[nodiscard]] inline size_t revive(EMeansState& state, bool last) {
    const size_t d = state.d;
    const size_t k = state.k;
    auto& costs = state.costs;
    auto& axis_norms = state.axis_norms;
    for (size_t cluster = 0; cluster < k; ++cluster) {
        const float* removal = state.removal.data() + cluster * (d + 1);
        costs[cluster] = removal[d];
        double norm2 = 0.0;
        for (size_t dim = 0; dim < d; ++dim) {
            norm2 += static_cast<double>(removal[dim]) * removal[dim];
        }
        axis_norms[cluster] = state.axis_scale * std::sqrt(norm2);
    }
    auto& sorted = state.sorted_costs;
    std::copy(costs.begin(), costs.end(), sorted.begin());
    const size_t middle = k / 2;
    std::nth_element(
        sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(middle), sorted.end()
    );
    double median = sorted[middle];
    if (k % 2 == 0) {
        median = (median +
                  *std::max_element(
                      sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(middle)
                  )) /
                 2.0;
    }
    const double threshold = kRemovalThreshold * median;

    auto& candidates = state.candidates;
    auto& donors = state.donors;
    candidates.clear();
    donors.clear();
    for (size_t cluster = 0; cluster < k; ++cluster) {
        if (costs[cluster] <= 0 || costs[cluster] < threshold) {
            candidates.push_back(cluster);
        } else if (state.mass[cluster] > 0) {
            donors.push_back(cluster);
        }
    }
    if (candidates.empty()) {
        return 0;
    }
    std::stable_sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
        return costs[a] < costs[b];
    });
    // Donors rank by their float32 axis length, as in the reference.
    std::stable_sort(donors.begin(), donors.end(), [&](size_t a, size_t b) {
        return static_cast<float>(axis_norms[a]) > static_cast<float>(axis_norms[b]);
    });
    size_t nsplit = 0;
    const size_t pairs = std::min(candidates.size(), donors.size());
    while (nsplit < pairs) {
        const double cost = costs[candidates[nsplit]];
        const double gain = kSplitGain * static_cast<float>(axis_norms[donors[nsplit]]);
        if (!(cost <= 0 || gain > cost)) {
            break;
        }
        ++nsplit;
    }

    // Donors and candidates are disjoint, so each split reads its donor's
    // state from before this check. Split coordinates stay within the bound
    // train() and assign() accept, so a trained model always assigns.
    const auto bound = static_cast<double>(max_coordinate_magnitude(d));
    for (size_t split = 0; split < nsplit; ++split) {
        const size_t source = donors[split];
        const size_t target = candidates[split];
        const double norm = axis_norms[source];
        const auto mass = static_cast<double>(state.mass[source]);
        const bool valid =
            norm > 0 && mass > 0 && std::isfinite(norm) && std::isfinite(mass);
        const double spread = valid ? std::sqrt(2.0 * (norm / mass) / kPi) : 0.0;
        float* donor = state.centroids.data() + source * d;
        float* child = state.centroids.data() + target * d;
        const float* axis = state.axes.data() + source * d;
        bool collapsed = true;
        for (size_t dim = 0; dim < d; ++dim) {
            const auto value = static_cast<double>(donor[dim]);
            const double offset = spread * static_cast<double>(axis[dim]);
            donor[dim] = static_cast<float>(std::clamp(value - offset, -bound, bound));
            child[dim] = static_cast<float>(std::clamp(value + offset, -bound, bound));
            collapsed = collapsed && donor[dim] == child[dim];
        }
        if (collapsed) {
            // Coincident copies; separate the child along one coordinate, toward
            // zero so it stays within the bound.
            child[0] = std::nextafter(
                child[0],
                child[0] > 0 ? -std::numeric_limits<float>::infinity()
                             : std::numeric_limits<float>::infinity()
            );
        }
        const float half = state.mass[source] / 2.0F;
        state.mass[source] = half;
        state.mass[target] = half;
        float* source_removal = state.removal.data() + source * (d + 1);
        float* target_removal = state.removal.data() + target * (d + 1);
        for (size_t dim = 0; dim <= d; ++dim) {
            source_removal[dim] /= 2.0F;
            target_removal[dim] = source_removal[dim];
        }
        std::copy_n(axis, d, state.axes.data() + target * d);
        if (last) {
            for (const size_t cluster : {source, target}) {
                state.pass_counts[cluster] = 0.0;
                std::fill_n(state.pass_sums.data() + cluster * d, d, 0.0F);
            }
        }
    }
    return nsplit;
}

// Closing means: centroids assigned during the last iteration become the mean of
// their last-iteration members; the others keep their positions.
inline void close_means(EMeansState& state, uint32_t num_workers) {
    const size_t d = state.d;
    const size_t k = state.k;
    const float bound = max_coordinate_magnitude(d);
    static_cast<void>(num_workers);
#pragma omp parallel for num_threads(num_workers) schedule(static)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(k); ++index) {
        const auto cluster = static_cast<size_t>(index);
        if (state.pass_counts[cluster] > 0) {
            const auto count = static_cast<float>(state.pass_counts[cluster]);
            for (size_t dim = 0; dim < d; ++dim) {
                // As in apply_statistics, keep rounded means within the bound.
                state.centroids[cluster * d + dim] =
                    std::clamp(state.pass_sums[cluster * d + dim] / count, -bound, bound);
            }
        }
    }
}

/**
 * Mini-batch e-means over a pluggable assignment, the counterpart of LloydKMeans.
 * Parameters provides niter, batch_size, verbose, seed, final_assignment and
 * num_threads. Assignment provides:
 *
 *   static constexpr const char* kName;  // prefixes messages and output
 *   static void validate_parameters(size_t d, size_t k, const Parameters&);
 *   Assignment(size_t d, size_t k, const float* x, size_t n, uint32_t num_workers,
 *              const Parameters&);  // one training call; may borrow x
 *   void prepare(const float* centroids);  // before each mini-batch and the final
 *                                          // pass; may borrow centroids until the
 *                                          // next prepare
 *   void assign_batch(const size_t* rows, size_t count, PID* assignments,
 *                     float* distances, float* second_distances);
 *   void end_pass();  // after each pass over all n training rows
 *   struct ModelState;  // kept by the model for assign()
 *   void store_model_state(ModelState&) const;
 *   static bool has_model_state(const ModelState&, size_t d);
 *   static void assign_with_state(size_t d, size_t k, const float* centroids,
 *                                 const float* x, size_t n, uint32_t num_workers,
 *                                 const Parameters&, const ModelState&,
 *                                 PID* assignments, float* distances);
 *
 * assign_batch labels rows[i] (row i when rows is null) of the training x with its
 * nearest prepared centroid, ties to the lower ID, and returns exact squared L2
 * distances to that centroid and, when second_distances is non-null, to the second
 * nearest; infinity when the assignment finds only one centroid. Revival reads
 * second - first as each vector's removal cost.
 */
template <typename Parameters, typename Assignment>
class EMeans : public Parameters {
   public:
    using Parameters::batch_size;
    using Parameters::final_assignment;
    using Parameters::niter;
    using Parameters::num_threads;
    using Parameters::seed;
    using Parameters::verbose;

    EMeans(size_t d, size_t k, const Parameters& cp) : Parameters(cp), d(d), k(k) {
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
        assign_trained(
            Assignment::kName,
            Assignment::has_model_state(model_state_, d),
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
            throw std::invalid_argument(
                std::string(Assignment::kName) + " x must not be null"
            );
        }
        if (n < k) {
            throw std::invalid_argument(
                std::string(Assignment::kName) + " requires n >= k"
            );
        }
        const uint32_t num_workers = resolved_num_workers();
        validate_points(Assignment::kName, n, d, x, num_workers, "x");

        const Recipe recipe = resolve_recipe(n, k, niter, batch_size);
        if (verbose) {
            std::cout << Assignment::kName << " mini-batches of " << recipe.batch_size
                      << " training points\n";
        }
        Assignment graph(d, k, x, n, num_workers, *this);
        EMeansState state;
        initialize(state, d, k, n, x, recipe.batch_size, seed);

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
            start_iteration(state, last);
            std::iota(order.begin(), order.end(), size_t{0});
            std::mt19937_64 rng(iteration_seed(seed, iteration));
            std::shuffle(order.begin(), order.end(), rng);
            double objective = 0.0;
            size_t nsplit = 0;
            for (size_t begin = 0; begin < n; begin += recipe.batch_size) {
                const size_t rows = std::min(recipe.batch_size, n - begin);
                const size_t* batch = order.data() + begin;
                graph.prepare(state.centroids.data());
                graph.assign_batch(
                    batch,
                    rows,
                    state.batch_assignments.data(),
                    state.batch_distances.data(),
                    state.batch_second_distances.data()
                );
                objective += accumulate(state, x, batch, rows, num_workers);
                apply_statistics(
                    state,
                    scheduled_alpha(recipe.alpha0, step, recipe.total_steps),
                    last,
                    num_workers
                );
                if ((step + 1) % kReviveEvery == 0) {
                    nsplit += revive(state, last);
                }
                ++step;
            }
            graph.end_pass();
            if (last) {
                close_means(state, num_workers);
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
                std::cout << Assignment::kName << " iteration " << iteration + 1 << "/"
                          << niter << " | objective=" << objective << " | shift=" << shift
                          << " | splits=" << nsplit << '\n';
            }
        }

        std::vector<PID> final_assignments;
        std::vector<float> final_distances;
        double objective = 0.0;
        if (assign_all) {
            final_assignments.resize(n);
            final_distances.resize(n);
            // The approximate pass reuses the training assignment and its state.
            assign_with_mode(
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
                    graph.prepare(state.centroids.data());
                    graph.assign_batch(
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
        graph.store_model_state(model_state_);
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
        assign_with_mode(
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
                Assignment::assign_with_state(
                    d,
                    k,
                    centroids.data(),
                    x,
                    n,
                    num_workers,
                    *this,
                    model_state_,
                    point_assignments,
                    point_distances
                );
            }
        );
    }

    void validate_parameters() const {
        validate_common_parameters(
            Assignment::kName, d, k, niter, "niter", num_threads, final_assignment
        );
        Assignment::validate_parameters(d, k, *this);
    }

    [[nodiscard]] uint32_t resolved_num_workers() const {
        return static_cast<uint32_t>(resolve_num_threads(num_threads));
    }

    // Kept for assign(); empty until the first successful fit.
    typename Assignment::ModelState model_state_;
};

}  // namespace rabitqlib::qgemeans::detail
