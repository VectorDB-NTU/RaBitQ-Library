#pragma once

#include <omp.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/core/detail/simd/matrix_dispatch.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/index/detail/parallel_exception.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::clustering {

enum class FinalAssignmentMode : uint8_t {
    Approximate = 0,
    SymphonyQG = Approximate,  // Compatibility name for the graph assignment path.
    Exact = 1,
};

struct KMeansIterationStats {
    size_t iteration = 0;
    double obj = 0;
    double shift = 0;
    size_t nsplit = 0;
};

namespace detail {

[[nodiscard]] inline bool should_terminate_by_objective(
    double previous_objective, double objective, double early_stop_threshold
) {
    if (previous_objective == 0.0) {
        return objective == 0.0;
    }
    const double change =
        std::abs(previous_objective - objective) / std::abs(previous_objective);
    return change <= early_stop_threshold;
}

// Float vectors accumulated in double, in index order.
[[nodiscard]] inline double l2norm_sqr_f64(const float* vector, size_t d) {
    double result = 0.0;
    for (size_t dim = 0; dim < d; ++dim) {
        const double value = vector[dim];
        result += value * value;
    }
    return result;
}

[[nodiscard]] inline double euclidean_sqr_f64(const float* a, const float* b, size_t d) {
    double result = 0.0;
    for (size_t dim = 0; dim < d; ++dim) {
        const double difference = static_cast<double>(a[dim]) - b[dim];
        result += difference * difference;
    }
    return result;
}

// Inputs have already passed clustering validation. GEMM only screens candidates;
// final distances and tie breaking retain the scalar double-precision arithmetic.
// Optional previous assignments contain n valid centroid IDs and may alias the
// output. Recompute their distances against the current, updated centroids.
inline void exact_assign(
    const float* x,
    const float* centroids,
    size_t n,
    size_t d,
    size_t k,
    bool spherical,
    uint32_t num_workers,
    PID* assignments,
    float* distances,
    const PID* previous_assignments = nullptr
) {
    constexpr size_t kMaxPointBlock = 128;
    // Larger row blocks amortize packing on the AVX-512 Eigen backend. Retain
    // smaller blocks on other backends and require at least two blocks per
    // worker to avoid doubling the work of a worker handling a small tail.
    const size_t point_block =
        cpu::has_avx512_core() && k >= 256 && n / num_workers >= 2 * kMaxPointBlock
            ? kMaxPointBlock
            : 64;
    constexpr size_t kCentroidBlock = 256;
    const size_t blocks = 1 + (n - 1) / point_block;
    const auto team_size = static_cast<uint32_t>(std::min<size_t>(num_workers, blocks));
    std::vector<double> centroid_norms(k);
    std::vector<std::vector<float>> products(
        team_size, std::vector<float>(point_block * std::min(k, kCentroidBlock))
    );
    index_detail::ParallelException failure;

    const auto exact_distance = [d, spherical](const float* vector, const float* centroid) {
        double distance = 0.0;
        if (spherical) {
            for (size_t dim = 0; dim < d; ++dim) {
                distance -= static_cast<double>(vector[dim]) * centroid[dim];
            }
            distance += 1.0;
        } else {
            distance = euclidean_sqr_f64(vector, centroid, d);
        }
        return distance;
    };

    // For d <= 65536, this exceeds the float dot-product error bound
    // gamma_d * sum(abs(x_i*c_i)), using 2*abs(x_i*c_i) <= x_i^2+c_i^2.
    // The slack also covers double norm/distance rounding and underflow.
    const double error_scale = 8.0 * static_cast<double>(d) *
                               static_cast<double>(std::numeric_limits<float>::epsilon());
    const double error_floor =
        8.0 * static_cast<double>(d) * std::numeric_limits<float>::min() +
        8.0 * std::numeric_limits<double>::epsilon();
#pragma omp parallel num_threads(team_size)
    {
        // Set only this OpenMP task's ICV, not Eigen's process-global thread limit.
        // This also prevents nested GEMM teams when the outer team has one worker.
        omp_set_num_threads(1);
        const auto thread = static_cast<size_t>(omp_get_thread_num());
        float* dots = products[thread].data();
        std::array<double, kMaxPointBlock> point_norms{};
        std::array<double, kMaxPointBlock> best_distances{};
        // Reuse this worker team for norms and assignment. The implicit barrier
        // publishes all centroid norms before any matrix tile is screened.
#pragma omp for schedule(static)
        for (std::ptrdiff_t cluster = 0; cluster < static_cast<std::ptrdiff_t>(k);
             ++cluster) {
            centroid_norms[static_cast<size_t>(cluster)] =
                l2norm_sqr_f64(centroids + static_cast<size_t>(cluster) * d, d);
        }
#pragma omp for schedule(static)
        for (std::ptrdiff_t block = 0; block < static_cast<std::ptrdiff_t>(blocks);
             ++block) {
            if (failure.failed()) {
                continue;
            }
            try {
                const size_t first_point = static_cast<size_t>(block) * point_block;
                const size_t rows = std::min(point_block, n - first_point);
                for (size_t row = 0; row < rows; ++row) {
                    const float* vector = x + (first_point + row) * d;
                    point_norms[row] = l2norm_sqr_f64(vector, d);
                    best_distances[row] = std::numeric_limits<double>::infinity();
                    if (previous_assignments != nullptr) {
                        const size_t point = first_point + row;
                        const PID previous = previous_assignments[point];
                        assert(previous < k);
                        best_distances[row] =
                            exact_distance(vector, centroids + previous * d);
                        assignments[point] = previous;
                    }
                }
                for (size_t first_cluster = 0; first_cluster < k;
                     first_cluster += kCentroidBlock) {
                    const size_t cols = std::min(kCentroidBlock, k - first_cluster);
                    simd::matrix_product_transposed(
                        x + first_point * d,
                        centroids + first_cluster * d,
                        dots,
                        rows,
                        d,
                        cols
                    );
                    for (size_t row = 0; row < rows; ++row) {
                        const size_t point = first_point + row;
                        const float* vector = x + point * d;
                        for (size_t col = 0; col < cols; ++col) {
                            const size_t cluster = first_cluster + col;
                            const double norm_sum =
                                point_norms[row] + centroid_norms[cluster];
                            const double dot = dots[row * cols + col];
                            const double estimate =
                                spherical ? 1.0 - dot : norm_sum - 2.0 * dot;
                            const double error = error_scale * norm_sum + error_floor;
                            if (estimate > best_distances[row] + error) {
                                continue;
                            }
                            const float* centroid = centroids + cluster * d;
                            const double distance = exact_distance(vector, centroid);
                            // The hint may have a higher ID than an equally good
                            // centroid. Preserve the original lowest-ID tie rule.
                            if (distance < best_distances[row] ||
                                (distance == best_distances[row] &&
                                 cluster < assignments[point])) {
                                assignments[point] = static_cast<PID>(cluster);
                                best_distances[row] = distance;
                            }
                        }
                    }
                }
                for (size_t row = 0; row < rows; ++row) {
                    distances[first_point + row] = static_cast<float>(best_distances[row]);
                }
            } catch (...) { failure.capture(); }
        }
    }
    failure.rethrow();
}

// Returns count sorted, distinct row indices in [0, n). The seed sequence keeps
// this stream independent of the centroid initialization stream.
[[nodiscard]] inline std::vector<size_t> sample_training_points(
    size_t n, size_t count, uint32_t seed
) {
    assert(count <= n);
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), size_t{0});
    std::seed_seq sequence{seed, uint32_t{1}};
    std::mt19937 rng(sequence);
    for (size_t i = 0; i < count; ++i) {
        std::uniform_int_distribution<size_t> pick(i, n - 1);
        std::swap(indices[i], indices[pick(rng)]);
    }
    indices.resize(count);
    std::sort(indices.begin(), indices.end());
    return indices;
}

template <typename Parameters, typename Assignment>
class LloydKMeans : public Parameters {
   public:
    using Parameters::early_stop_threshold;
    using Parameters::final_assignment;
    using Parameters::max_points_per_centroid;
    using Parameters::min_points_per_centroid;
    using Parameters::niter;
    using Parameters::num_threads;
    using Parameters::seed;
    using Parameters::spherical;
    using Parameters::verbose;

    LloydKMeans(size_t d, size_t k, const Parameters& cp) : Parameters(cp), d(d), k(k) {
        validate_parameters();
    }

    void train(size_t n, const float* x) { fit(n, x, true); }

    /**
     * Trains the same centroids as train() but skips its final assignment of all n
     * vectors, which can cost more than every sampled iteration together. Leaves
     * assignments and distances empty and final_obj zero; centroids,
     * iteration_stats and assign() match a train() call.
     */
    void train_centroids(size_t n, const float* x) { fit(n, x, false); }

   private:
    // assign_all selects the final assignment that train() returns.
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
        validate_points(n, x, num_workers, "x");
        if (verbose && min_points_per_centroid > 0 && n / k < min_points_per_centroid) {
            std::cerr << "WARNING " << Assignment::kName << " clustering " << n
                      << " points to " << k << " centroids: please provide at least "
                      << min_points_per_centroid << " training points per centroid\n";
        }

        // Retraining may borrow the previous centroids, including a subrange.
        // Preserve that input through initialization, updates, and final assignment.
        std::vector<float> input_copy;
        const std::less<const float*> before;
        if (!centroids.empty() && !before(x, centroids.data()) &&
            before(x, centroids.data() + centroids.size())) {
            input_copy.assign(x, x + n * d);
            x = input_copy.data();
        }

        // Sample when n > k * max_points_per_centroid, without forming the product.
        const bool sampled =
            max_points_per_centroid != 0 && max_points_per_centroid <= (n - 1) / k;
        const size_t num_samples = sampled ? k * max_points_per_centroid : n;
        std::vector<float> samples;
        const float* training_points = x;
        if (sampled) {
            if (verbose) {
                std::cout << Assignment::kName << " sampling " << num_samples << " of " << n
                          << " training points\n";
            }
            const auto indices = sample_training_points(n, num_samples, seed);
            samples.resize(num_samples * d);
#pragma omp parallel for num_threads(num_workers) schedule(static)
            for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_samples);
                 ++index) {
                const auto sample = static_cast<size_t>(index);
                std::copy_n(x + indices[sample] * d, d, samples.data() + sample * d);
            }
            training_points = samples.data();
        }

        std::optional<Assignment> assigner(
            std::in_place, d, k, training_points, num_samples, num_workers, *this
        );

        centroids.resize(k * d);
        assignments.resize(num_samples);
        distances.resize(num_samples);
        iteration_stats.clear();
        initialize_centroids(training_points, num_samples);
        if (spherical) {
            normalize_centroids();
        }

        std::vector<float> previous_centroids(centroids.size());
        std::vector<size_t> cluster_sizes(k);
        std::vector<double> sums(centroids.size());
        const size_t update_workers = std::min(k, static_cast<size_t>(num_workers));
        std::vector<UpdatePartition> update_partitions(update_workers);
        std::vector<size_t> update_worker_for_cluster(k);
        for (size_t worker = 0; worker < update_workers; ++worker) {
            auto& partition = update_partitions[worker];
            partition.first_cluster = k * worker / update_workers;
            partition.last_cluster = k * (worker + 1) / update_workers;
            for (size_t cluster = partition.first_cluster; cluster < partition.last_cluster;
                 ++cluster) {
                update_worker_for_cluster[cluster] = worker;
            }
            partition.points.reserve(num_samples / update_workers);
        }
        double previous_obj = std::numeric_limits<double>::infinity();

        for (size_t iteration = 0; iteration < niter; ++iteration) {
            assigner->assign(centroids.data(), assignments.data(), distances.data());
            // Assignment has finished borrowing the current centroids. Keep them
            // for the shift calculation and overwrite the other buffer with means.
            previous_centroids.swap(centroids);

            const double obj = std::accumulate(distances.begin(), distances.end(), 0.0);
            update_centroids(
                training_points,
                num_samples,
                sums,
                cluster_sizes,
                update_partitions,
                update_worker_for_cluster
            );
            const auto [nsplit, shift] = consolidate_centroids(
                training_points, num_samples, sums, cluster_sizes, previous_centroids
            );
            iteration_stats.push_back(
                KMeansIterationStats{iteration + 1, obj, shift, nsplit}
            );

            if (verbose) {
                std::cout << Assignment::kName << " iteration " << iteration + 1 << "/"
                          << niter << " | objective=" << obj << " | shift=" << shift
                          << " | splits=" << nsplit << '\n';
            }

            const bool converged = detail::should_terminate_by_objective(
                previous_obj, obj, early_stop_threshold
            );
            previous_obj = obj;
            if (iteration > 0 && converged) {
                break;
            }
        }

        assigner->store_model_state(assignment_state_);
        if (!assign_all) {
            std::vector<PID>().swap(assignments);
            std::vector<float>().swap(distances);
            final_obj = 0.0;
            return;
        }
        // Return assignments for the final centroids, not the centroids from
        // the beginning of the last update step.
        if (sampled) {
            // Release the sample and its encoded form before assigning all points.
            assigner.reset();
            std::vector<float>().swap(samples);
            assignments.resize(n);
            distances.resize(n);
            assign_points(n, x, num_workers, assignments.data(), distances.data());
        } else if (final_assignment == FinalAssignmentMode::Exact) {
            detail::exact_assign(
                x,
                centroids.data(),
                n,
                d,
                k,
                spherical,
                num_workers,
                assignments.data(),
                distances.data(),
                assignments.data()
            );
        } else {
            assigner->assign(centroids.data(), assignments.data(), distances.data(), true);
        }
        final_obj = std::accumulate(distances.begin(), distances.end(), 0.0);
    }

   public:
    /**
     * Assigns n row-major vectors to the current centroids with final_assignment.
     * Each vector's result is independent of the others in the call and of
     * previous labels. Requires a successful train() or train_centroids(); the
     * outputs hold n elements each and must not overlap x or centroids.
     */
    void assign(
        size_t n, const float* x, PID* point_assignments, float* point_distances
    ) const {
        validate_parameters();
        if (centroids.size() != k * d ||
            !Assignment::has_model_state(assignment_state_, d)) {
            throw std::runtime_error(
                std::string(Assignment::kName) + " has not been trained"
            );
        }
        const uint32_t num_workers = resolved_num_workers();
        validate_points(k, centroids.data(), num_workers, "centroids");
        if (n == 0) {
            return;
        }
        if (x == nullptr || point_assignments == nullptr || point_distances == nullptr) {
            throw std::invalid_argument(
                std::string(Assignment::kName) +
                " assign input and output pointers must not be null"
            );
        }
        validate_points(n, x, num_workers, "x");
        assign_points(n, x, num_workers, point_assignments, point_distances);
    }

    size_t d;
    size_t k;
    std::vector<float> centroids;
    std::vector<PID> assignments;
    std::vector<float> distances;
    std::vector<KMeansIterationStats> iteration_stats;
    // Objective of the final returned assignments against the final centroids.
    double final_obj = 0.0;

   private:
    // Encoding center or centroid graph cache for assign(); see the Assignment type.
    typename Assignment::ModelState assignment_state_;

    struct UpdatePartition {
        size_t first_cluster;
        size_t last_cluster;  // Exclusive; also bounds the accumulators we clear.
        std::vector<size_t> points;
    };

    void validate_parameters() const {
        if (d == 0 || k == 0 || niter == 0) {
            throw std::invalid_argument(
                std::string(Assignment::kName) + " d, k, and niter must be positive"
            );
        }
        if (d < 64 || d > rotator_impl::FhtKacRotator::kMaxDim) {
            throw std::invalid_argument(
                std::string(Assignment::kName) +
                " currently requires dimensions in the range [64, 65536]"
            );
        }
        if (num_threads > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(
                std::string(Assignment::kName) + " num_threads must fit in an OpenMP int"
            );
        }
        if (final_assignment != FinalAssignmentMode::Approximate &&
            final_assignment != FinalAssignmentMode::Exact) {
            throw std::invalid_argument(
                std::string(Assignment::kName) + " final_assignment mode is invalid"
            );
        }
        Assignment::validate_parameters(k, *this);
        if (!std::isfinite(early_stop_threshold) || early_stop_threshold < 0.0 ||
            early_stop_threshold > 1.0) {
            throw std::invalid_argument(
                std::string(Assignment::kName) +
                " early_stop_threshold must be in the range [0, 1]"
            );
        }
    }

    // Requires a non-null x. name identifies the rows in error messages.
    void validate_points(
        size_t n, const float* x, uint32_t num_workers, const char* name
    ) const {
        if (n > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / d) {
            throw std::invalid_argument(
                std::string(Assignment::kName) + " input dimensions overflow ptrdiff_t"
            );
        }
        // Leave headroom for products of squared distances in graph pruning,
        // as well as quantized reconstruction and rotated residuals.
        const double max_coordinate = std::sqrt(
            std::sqrt(static_cast<double>(std::numeric_limits<float>::max())) /
            (64.0 * static_cast<double>(d))
        );
        bool in_range = true;
        bool finite = true;
#pragma omp parallel for num_threads(num_workers) reduction(&& : finite, in_range) \
    schedule(static)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(n * d);
             ++index) {
            const size_t i = static_cast<size_t>(index);
            finite = finite && std::isfinite(x[i]);
            in_range = in_range && std::abs(static_cast<double>(x[i])) <= max_coordinate;
        }
        if (!finite) {
            throw std::invalid_argument(
                std::string(Assignment::kName) + " " + name +
                " must contain only finite values"
            );
        }
        if (!in_range) {
            throw std::invalid_argument(
                std::string(Assignment::kName) + " " + name +
                " exceeds the safe float32 coordinate range"
            );
        }
    }

    // Assigns validated points to the current centroids. Uses the state saved by
    // train() but neither Lloyd assignments nor graph search hints, so each
    // point's result is independent of the batch. Requires n > 0 and outputs
    // that do not overlap x or centroids.
    void assign_points(
        size_t n,
        const float* x,
        uint32_t num_workers,
        PID* point_assignments,
        float* point_distances
    ) const {
        if (final_assignment == FinalAssignmentMode::Exact) {
            detail::exact_assign(
                x,
                centroids.data(),
                n,
                d,
                k,
                spherical,
                num_workers,
                point_assignments,
                point_distances
            );
        } else {
            Assignment::assign_with_state(
                d,
                k,
                centroids.data(),
                x,
                n,
                num_workers,
                *this,
                assignment_state_,
                point_assignments,
                point_distances
            );
        }
    }

    void initialize_centroids(const float* x, size_t n) {
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), size_t{0});
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
        for (size_t cluster = 0; cluster < k; ++cluster) {
            std::memcpy(
                centroids.data() + cluster * d, x + indices[cluster] * d, d * sizeof(float)
            );
        }
    }

    void update_centroids(
        const float* x,
        size_t n,
        std::vector<double>& sums,
        std::vector<size_t>& cluster_sizes,
        std::vector<UpdatePartition>& partitions,
        const std::vector<size_t>& worker_for_cluster
    ) {
        for (auto& partition : partitions) {
            partition.points.clear();
        }
        // Keep point order within each cluster, preserving the original sums.
        for (size_t point = 0; point < n; ++point) {
            partitions[worker_for_cluster[assignments[point]]].points.push_back(point);
        }
        const uint32_t num_workers = resolved_num_workers();
        static_cast<void>(num_workers);
#pragma omp parallel for num_threads(num_workers) schedule(static)
        for (std::ptrdiff_t index = 0;
             index < static_cast<std::ptrdiff_t>(partitions.size());
             ++index) {
            const auto& partition = partitions[static_cast<size_t>(index)];
            // The same interval defines point ownership and accumulator clearing,
            // including empty clusters, independently of the physical thread team.
            const size_t first_cluster = partition.first_cluster;
            const size_t last_cluster = partition.last_cluster;
            std::fill_n(
                sums.data() + first_cluster * d, (last_cluster - first_cluster) * d, 0.0
            );
            std::fill_n(
                cluster_sizes.data() + first_cluster,
                last_cluster - first_cluster,
                size_t{0}
            );
            simd::accumulate_cluster_sums(
                x,
                partition.points.data(),
                partition.points.size(),
                d,
                assignments.data(),
                sums.data(),
                cluster_sizes.data()
            );
        }
    }

    std::pair<size_t, double> consolidate_centroids(
        const float* x,
        size_t n,
        const std::vector<double>& sums,
        std::vector<size_t>& cluster_sizes,
        const std::vector<float>& previous_centroids
    ) {
        const uint32_t num_workers = resolved_num_workers();
        // Means, normalization, and displacement share one worker team. Preserve
        // the per-centroid arithmetic; empty rows are measured after refilling.
        const auto cluster_shift = [&](size_t cluster) {
            return euclidean_sqr_f64(
                centroids.data() + cluster * d, previous_centroids.data() + cluster * d, d
            );
        };
        double shift = 0;
#pragma omp parallel for num_threads(num_workers) reduction(+ : shift)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(k); ++index) {
            const size_t cluster = static_cast<size_t>(index);
            const double scale = cluster_sizes[cluster] == 0
                                     ? 0.0
                                     : 1.0 / static_cast<double>(cluster_sizes[cluster]);
            for (size_t dim = 0; dim < d; ++dim) {
                centroids[cluster * d + dim] =
                    static_cast<float>(sums[cluster * d + dim] * scale);
            }
            if (spherical) {
                normalize_centroid(centroids.data() + cluster * d);
            }
            // Empty centroids contribute only after their replacement is known.
            if (cluster_sizes[cluster] != 0) {
                shift += cluster_shift(cluster);
            }
        }

        const size_t nsplit = static_cast<size_t>(
            std::count(cluster_sizes.begin(), cluster_sizes.end(), size_t{0})
        );
        if (nsplit == 0) {
            return {0, shift};
        }

        const auto replace_empty_centroid = [&](size_t cluster, size_t point) {
            std::copy_n(x + point * d, d, centroids.data() + cluster * d);
            if (spherical) {
                normalize_centroid(centroids.data() + cluster * d);
            }
            --cluster_sizes[assignments[point]];
            cluster_sizes[cluster] = 1;
            shift += cluster_shift(cluster);
        };

        // Reuse the distance buffer after recording the pre-update objective.
        // Keep populated means intact and give unused centroids poorly fitted
        // training points. Unlike coordinate perturbations, this also works at zero.
        const auto distance_func =
            spherical ? dot_product_dis<float> : euclidean_sqr<float>;
        const auto less = [&](size_t left, size_t right) {
            return distances[left] == distances[right] ? left > right
                                                       : distances[left] < distances[right];
        };
        if (nsplit == 1) {
            // A single empty centroid needs only one eligible maximum per worker.
            std::vector<size_t> best_points(num_workers, n);
#pragma omp parallel num_threads(num_workers)
            {
                size_t best = n;
#pragma omp for schedule(static)
                for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(n);
                     ++index) {
                    const size_t point = static_cast<size_t>(index);
                    if (cluster_sizes[assignments[point]] <= 1) {
                        continue;
                    }
                    distances[point] = distance_func(
                        x + point * d, centroids.data() + assignments[point] * d, d
                    );
                    if (best == n || less(best, point)) {
                        best = point;
                    }
                }
                best_points[static_cast<size_t>(omp_get_thread_num())] = best;
            }
            size_t point = n;
            for (const size_t candidate : best_points) {
                if (candidate != n && (point == n || less(point, candidate))) {
                    point = candidate;
                }
            }
            assert(point != n);  // n >= k guarantees a donor with at least two points.
            const size_t cluster = static_cast<size_t>(
                std::find(cluster_sizes.begin(), cluster_sizes.end(), size_t{0}) -
                cluster_sizes.begin()
            );
            replace_empty_centroid(cluster, point);
            return {1, shift};
        }

#pragma omp parallel for num_threads(num_workers) schedule(static)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(n); ++index) {
            const size_t point = static_cast<size_t>(index);
            distances[point] =
                distance_func(x + point * d, centroids.data() + assignments[point] * d, d);
        }
        std::vector<size_t> candidates(n);
        std::iota(candidates.begin(), candidates.end(), size_t{0});
        std::make_heap(candidates.begin(), candidates.end(), less);
        for (size_t cluster = 0; cluster < k; ++cluster) {
            if (cluster_sizes[cluster] != 0) {
                continue;
            }
            size_t point = 0;
            do {
                point = candidates.front();
                std::pop_heap(candidates.begin(), candidates.end(), less);
                candidates.pop_back();
            } while (cluster_sizes[assignments[point]] <= 1);
            replace_empty_centroid(cluster, point);
        }
        return {nsplit, shift};
    }

    void normalize_centroid(float* centroid) const {
        const double norm_sqr = l2norm_sqr_f64(centroid, d);
        if (norm_sqr == 0) {
            return;
        }
        const double scale = 1.0 / std::sqrt(norm_sqr);
        for (size_t dim = 0; dim < d; ++dim) {
            centroid[dim] = static_cast<float>(centroid[dim] * scale);
        }
    }

    void normalize_centroids() {
        const uint32_t num_workers = resolved_num_workers();
        static_cast<void>(num_workers);
#pragma omp parallel for num_threads(num_workers)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(k); ++index) {
            const size_t cluster = static_cast<size_t>(index);
            normalize_centroid(centroids.data() + cluster * d);
        }
    }

    [[nodiscard]] uint32_t resolved_num_workers() const {
        return static_cast<uint32_t>(resolve_num_threads(num_threads));
    }
};

}  // namespace detail

}  // namespace rabitqlib::clustering
