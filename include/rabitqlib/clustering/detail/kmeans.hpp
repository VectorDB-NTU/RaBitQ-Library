#pragma once

#include <omp.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/simd/matrix_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/rotator.hpp"
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
    uint32_t num_threads,
    PID* assignments,
    float* distances,
    const PID* previous_assignments = nullptr
) {
    constexpr size_t kMaxPointBlock = 128;
    // Larger row blocks amortize packing on the AVX-512 Eigen backend. Retain
    // smaller blocks on other backends and require at least two blocks per
    // worker to avoid doubling the work of a worker handling a small tail.
    const size_t point_block =
        cpu::has_avx512_core() && k >= 256 && n / num_threads >= 2 * kMaxPointBlock
            ? kMaxPointBlock
            : 64;
    constexpr size_t kCentroidBlock = 256;
    const size_t blocks = 1 + (n - 1) / point_block;
    const auto threads = static_cast<uint32_t>(std::min<size_t>(num_threads, blocks));
    std::vector<double> centroid_norms(k);
    std::vector<std::vector<float>> products(
        threads, std::vector<float>(point_block * std::min(k, kCentroidBlock))
    );
    std::vector<std::exception_ptr> errors(threads);

    const auto exact_distance = [d, spherical](const float* vector, const float* centroid) {
        double distance = 0.0;
        if (spherical) {
            for (size_t dim = 0; dim < d; ++dim) {
                distance -= static_cast<double>(vector[dim]) * centroid[dim];
            }
            distance += 1.0;
        } else {
            for (size_t dim = 0; dim < d; ++dim) {
                const double difference = static_cast<double>(vector[dim]) - centroid[dim];
                distance += difference * difference;
            }
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
#pragma omp parallel num_threads(threads)
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
            double norm = 0.0;
            for (size_t dim = 0; dim < d; ++dim) {
                const double value = centroids[static_cast<size_t>(cluster) * d + dim];
                norm += value * value;
            }
            centroid_norms[static_cast<size_t>(cluster)] = norm;
        }
#pragma omp for schedule(static)
        for (std::ptrdiff_t block = 0; block < static_cast<std::ptrdiff_t>(blocks);
             ++block) {
            if (errors[thread]) {
                continue;
            }
            try {
                const size_t first_point = static_cast<size_t>(block) * point_block;
                const size_t rows = std::min(point_block, n - first_point);
                for (size_t row = 0; row < rows; ++row) {
                    const float* vector = x + (first_point + row) * d;
                    double norm = 0.0;
                    for (size_t dim = 0; dim < d; ++dim) {
                        const double value = vector[dim];
                        norm += value * value;
                    }
                    point_norms[row] = norm;
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
            } catch (...) { errors[thread] = std::current_exception(); }
        }
    }
    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}

template <typename Parameters, typename Assignment>
class LloydKMeans : public Parameters {
   public:
    using Parameters::early_stop_threshold;
    using Parameters::final_assignment;
    using Parameters::min_points_per_centroid;
    using Parameters::niter;
    using Parameters::num_threads;
    using Parameters::seed;
    using Parameters::spherical;
    using Parameters::verbose;

    LloydKMeans(size_t d, size_t k, const Parameters& cp) : Parameters(cp), d(d), k(k) {
        validate_parameters();
    }

    void train(size_t n, const float* x) {
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
        if (n > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / d) {
            throw std::invalid_argument(
                std::string(Assignment::kName) + " input dimensions overflow size_t"
            );
        }
        const uint32_t threads = effective_num_threads();
        static_cast<void>(threads);
        // Leave headroom for products of squared distances in graph pruning,
        // as well as quantized reconstruction and rotated residuals.
        const double max_coordinate = std::sqrt(
            std::sqrt(static_cast<double>(std::numeric_limits<float>::max())) /
            (64.0 * static_cast<double>(d))
        );
        bool in_range = true;
        bool finite = true;
#pragma omp parallel for num_threads(threads) reduction(&& : finite, in_range) schedule(static)
        for (std::ptrdiff_t loop_index = 0; loop_index < static_cast<std::ptrdiff_t>(n * d);
             ++loop_index) {
            const size_t i = static_cast<size_t>(loop_index);
            finite = finite && std::isfinite(x[i]);
            in_range = in_range && std::abs(static_cast<double>(x[i])) <= max_coordinate;
        }
        if (!finite) {
            throw std::invalid_argument(
                std::string(Assignment::kName) + " x must contain only finite values"
            );
        }
        if (!in_range) {
            throw std::invalid_argument(
                std::string(Assignment::kName) +
                " x exceeds the safe float32 coordinate range"
            );
        }
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

        Assignment assigner(d, k, x, n, threads, *this);

        centroids.resize(k * d);
        assignments.resize(n);
        distances.resize(n);
        iteration_stats.clear();
        iteration_stats.reserve(niter);
        initialize_centroids(x, n);
        if (spherical) {
            normalize_centroids();
        }

        std::vector<float> previous_centroids(centroids.size());
        std::vector<size_t> cluster_sizes(k);
        std::vector<double> sums(centroids.size());
        const size_t update_workers = std::min(k, static_cast<size_t>(threads));
        std::vector<std::vector<size_t>> update_point_bins(update_workers);
        std::vector<size_t> update_worker_for_cluster(k);
        for (size_t worker = 0; worker < update_workers; ++worker) {
            const size_t first_cluster = k * worker / update_workers;
            const size_t last_cluster = k * (worker + 1) / update_workers;
            for (size_t cluster = first_cluster; cluster < last_cluster; ++cluster) {
                update_worker_for_cluster[cluster] = worker;
            }
            update_point_bins[worker].reserve(n / update_workers);
        }
        double previous_obj = std::numeric_limits<double>::infinity();

        for (size_t iteration = 0; iteration < niter; ++iteration) {
            assigner.assign(centroids.data(), assignments.data(), distances.data());
            // Assignment has finished borrowing the current centroids. Keep them
            // for the shift calculation and overwrite the other buffer with means.
            previous_centroids.swap(centroids);

            const double obj = std::accumulate(distances.begin(), distances.end(), 0.0);
            std::fill(sums.begin(), sums.end(), 0.0);
            std::fill(cluster_sizes.begin(), cluster_sizes.end(), size_t{0});
            update_centroids(
                x, n, sums, cluster_sizes, update_point_bins, update_worker_for_cluster
            );
            const auto [nsplit, shift] =
                consolidate_centroids(x, n, sums, cluster_sizes, previous_centroids);
            iteration_stats.push_back(KMeansIterationStats{
                iteration + 1, obj, shift, nsplit});

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

        // Return assignments for the final centroids, not the centroids from
        // the beginning of the last update step.
        if (final_assignment == FinalAssignmentMode::Exact) {
            detail::exact_assign(
                x,
                centroids.data(),
                n,
                d,
                k,
                spherical,
                threads,
                assignments.data(),
                distances.data(),
                assignments.data()
            );
        } else {
            assigner.assign(centroids.data(), assignments.data(), distances.data(), true);
        }
        final_obj = std::accumulate(distances.begin(), distances.end(), 0.0);
    }

    size_t d;
    size_t k;
    std::vector<float> centroids;
    std::vector<PID> assignments;
    std::vector<float> distances;
    std::vector<KMeansIterationStats> iteration_stats;
    /// Objective of the final returned assignments against the final centroids.
    double final_obj = 0.0;

   private:
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
        std::vector<std::vector<size_t>>& point_bins,
        const std::vector<size_t>& worker_for_cluster
    ) {
        for (auto& bin : point_bins) {
            bin.clear();
        }
        // Keep point order within each cluster, preserving the original sums.
        for (size_t point = 0; point < n; ++point) {
            point_bins[worker_for_cluster[assignments[point]]].push_back(point);
        }
        const uint32_t threads = effective_num_threads();
        static_cast<void>(threads);
#pragma omp parallel for num_threads(threads) schedule(static)
        for (std::ptrdiff_t loop_worker = 0;
             loop_worker < static_cast<std::ptrdiff_t>(point_bins.size());
             ++loop_worker) {
            const size_t worker = static_cast<size_t>(loop_worker);
            simd::accumulate_cluster_sums(
                x,
                point_bins[worker].data(),
                point_bins[worker].size(),
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
        const uint32_t threads = effective_num_threads();
        static_cast<void>(threads);
        // Means, normalization, and displacement share one worker team. Preserve
        // the per-centroid arithmetic; empty rows are measured after refilling.
        const auto cluster_shift = [&](size_t cluster) {
            double result = 0;
            for (size_t dim = 0; dim < d; ++dim) {
                const size_t i = cluster * d + dim;
                const double difference =
                    static_cast<double>(centroids[i]) - previous_centroids[i];
                result += difference * difference;
            }
            return result;
        };
        double shift = 0;
#pragma omp parallel for num_threads(threads) reduction(+ : shift)
        for (std::ptrdiff_t loop_index = 0; loop_index < static_cast<std::ptrdiff_t>(k);
             ++loop_index) {
            const size_t cluster = static_cast<size_t>(loop_index);
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
            std::vector<size_t> best_points(threads, n);
#pragma omp parallel num_threads(threads)
            {
                size_t best = n;
#pragma omp for schedule(static)
                for (std::ptrdiff_t loop_index = 0;
                     loop_index < static_cast<std::ptrdiff_t>(n);
                     ++loop_index) {
                    const size_t point = static_cast<size_t>(loop_index);
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
            std::copy_n(x + point * d, d, centroids.data() + cluster * d);
            if (spherical) {
                normalize_centroid(centroids.data() + cluster * d);
            }
            --cluster_sizes[assignments[point]];
            cluster_sizes[cluster] = 1;
            shift += cluster_shift(cluster);
            return {1, shift};
        }

#pragma omp parallel for num_threads(threads) schedule(static)
        for (std::ptrdiff_t loop_index = 0; loop_index < static_cast<std::ptrdiff_t>(n);
             ++loop_index) {
            const size_t point = static_cast<size_t>(loop_index);
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
            std::copy_n(x + point * d, d, centroids.data() + cluster * d);
            if (spherical) {
                normalize_centroid(centroids.data() + cluster * d);
            }
            --cluster_sizes[assignments[point]];
            cluster_sizes[cluster] = 1;
            shift += cluster_shift(cluster);
        }
        return {nsplit, shift};
    }

    void normalize_centroid(float* centroid) const {
        double squared_norm = 0;
        for (size_t dim = 0; dim < d; ++dim) {
            squared_norm += static_cast<double>(centroid[dim]) * centroid[dim];
        }
        if (squared_norm == 0) {
            return;
        }
        const double scale = 1.0 / std::sqrt(squared_norm);
        for (size_t dim = 0; dim < d; ++dim) {
            centroid[dim] = static_cast<float>(centroid[dim] * scale);
        }
    }

    void normalize_centroids() {
        const uint32_t threads = effective_num_threads();
        static_cast<void>(threads);
#pragma omp parallel for num_threads(threads)
        for (std::ptrdiff_t loop_index = 0; loop_index < static_cast<std::ptrdiff_t>(k);
             ++loop_index) {
            const size_t cluster = static_cast<size_t>(loop_index);
            normalize_centroid(centroids.data() + cluster * d);
        }
    }

    [[nodiscard]] uint32_t effective_num_threads() const {
        return static_cast<uint32_t>(resolve_num_threads(num_threads));
    }
};

}  // namespace detail

}  // namespace rabitqlib::clustering
