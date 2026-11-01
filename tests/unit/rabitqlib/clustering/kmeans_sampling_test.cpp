#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "rabitqlib/clustering/qgkmeans.hpp"
#include "rabitqlib/clustering/rabitqkmeans.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib {
namespace {

using clustering::FinalAssignmentMode;
using clustering::detail::sample_training_points;

constexpr size_t kDim = 64;

std::vector<float> clustered_points(size_t num_points, uint32_t seed) {
    constexpr size_t kGroups = 5;
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> centers(kGroups * kDim);
    for (auto& value : centers) {
        value = 4.0F * normal(rng);
    }
    std::vector<float> points(num_points * kDim);
    for (size_t point = 0; point < num_points; ++point) {
        for (size_t dim = 0; dim < kDim; ++dim) {
            points[point * kDim + dim] =
                centers[(point % kGroups) * kDim + dim] + normal(rng);
        }
    }
    return points;
}

rabitqkmeans::RaBitQKMeansParameters test_parameters(const rabitqkmeans::RaBitQKMeans*) {
    rabitqkmeans::RaBitQKMeansParameters parameters;
    parameters.niter = 4;
    parameters.num_threads = 1;
    parameters.min_points_per_centroid = 1;
    return parameters;
}

qgkmeans::QGKMeansParameters test_parameters(const qgkmeans::QGKMeans*) {
    qgkmeans::QGKMeansParameters parameters;
    parameters.niter = 4;
    parameters.num_threads = 1;
    parameters.min_points_per_centroid = 1;
    parameters.ef_build = 64;
    parameters.ef_search = 16;
    return parameters;
}

template <typename Clustering>
auto test_parameters() {
    return test_parameters(static_cast<const Clustering*>(nullptr));
}

// QGKMeans requires more centroids than its default graph degree of 32.
template <typename Clustering>
constexpr size_t kClusters = std::is_same_v<Clustering, qgkmeans::QGKMeans> ? 33 : 7;

template <typename Clustering>
void expect_identical_results(const Clustering& actual, const Clustering& expected) {
    EXPECT_EQ(actual.centroids, expected.centroids);
    EXPECT_EQ(actual.assignments, expected.assignments);
    EXPECT_EQ(actual.distances, expected.distances);
    EXPECT_EQ(actual.final_obj, expected.final_obj);
    ASSERT_EQ(actual.iteration_stats.size(), expected.iteration_stats.size());
    for (size_t i = 0; i < actual.iteration_stats.size(); ++i) {
        EXPECT_EQ(actual.iteration_stats[i].obj, expected.iteration_stats[i].obj);
        EXPECT_EQ(actual.iteration_stats[i].shift, expected.iteration_stats[i].shift);
        EXPECT_EQ(actual.iteration_stats[i].nsplit, expected.iteration_stats[i].nsplit);
    }
}

double reference_distance(const float* x, const float* centroid, bool spherical) {
    double distance = 0;
    for (size_t dim = 0; dim < kDim; ++dim) {
        const double left = x[dim];
        const double right = centroid[dim];
        distance += spherical ? -left * right : (left - right) * (left - right);
    }
    return spherical ? 1.0 + distance : distance;
}

// Distances belong to every input point and its returned centroid; Exact mode
// additionally returns the lowest-ID nearest centroid.
template <typename Clustering>
void expect_points_assigned(
    const Clustering& clustering,
    const std::vector<float>& points,
    const std::vector<PID>& assignments,
    const std::vector<float>& distances
) {
    const size_t num_points = points.size() / kDim;
    ASSERT_EQ(assignments.size(), num_points);
    ASSERT_EQ(distances.size(), num_points);
    for (size_t point = 0; point < num_points; ++point) {
        const PID cluster_id = assignments[point];
        ASSERT_LT(cluster_id, clustering.k);
        const float* vector = points.data() + point * kDim;
        const double distance = reference_distance(
            vector, clustering.centroids.data() + cluster_id * kDim, clustering.spherical
        );
        EXPECT_NEAR(distances[point], distance, 1e-5 * std::max(1.0, distance));
        if (clustering.final_assignment != FinalAssignmentMode::Exact) {
            continue;
        }
        double best = std::numeric_limits<double>::infinity();
        PID best_id = 0;
        for (size_t cluster = 0; cluster < clustering.k; ++cluster) {
            const double candidate = reference_distance(
                vector, clustering.centroids.data() + cluster * kDim, clustering.spherical
            );
            if (candidate < best) {
                best = candidate;
                best_id = static_cast<PID>(cluster);
            }
        }
        EXPECT_EQ(cluster_id, best_id) << point;
    }
}

template <typename Clustering>
void expect_all_points_assigned(
    const Clustering& clustering, const std::vector<float>& points, size_t num_points
) {
    ASSERT_EQ(points.size(), num_points * kDim);
    expect_points_assigned(
        clustering, points, clustering.assignments, clustering.distances
    );
    EXPECT_EQ(
        clustering.final_obj,
        std::accumulate(clustering.distances.begin(), clustering.distances.end(), 0.0)
    );
}

template <typename Clustering>
std::pair<std::vector<PID>, std::vector<float>> assign_all(
    const Clustering& clustering, const std::vector<float>& points
) {
    const size_t num_points = points.size() / kDim;
    std::vector<PID> assignments(num_points);
    std::vector<float> distances(num_points);
    clustering.assign(num_points, points.data(), assignments.data(), distances.data());
    return {assignments, distances};
}

template <typename Clustering>
void check_assign_requires_training_and_valid_input(const std::string& name) {
    constexpr size_t kClusterCount = kClusters<Clustering>;
    const auto points = clustered_points(2 * kClusterCount, 61);
    Clustering clustering(kDim, kClusterCount, test_parameters<Clustering>());
    PID assignment = 0;
    float distance = 0;
    // Centroids written without train() lack the state that assignment uses.
    for (const bool with_centroids : {false, true}) {
        if (with_centroids) {
            clustering.centroids.assign(
                points.begin(), points.begin() + kClusterCount * kDim
            );
        }
        try {
            clustering.assign(1, points.data(), &assignment, &distance);
            FAIL() << "Assigned before training";
        } catch (const std::runtime_error& error) {
            EXPECT_EQ(error.what(), name + " has not been trained");
        }
    }

    clustering.train(2 * kClusterCount, points.data());
    EXPECT_NO_THROW(clustering.assign(0, nullptr, nullptr, nullptr));
    const auto expect_invalid = [&](const float* x, const std::string& message) {
        try {
            clustering.assign(1, x, &assignment, &distance);
            FAIL() << "Accepted " << message;
        } catch (const std::invalid_argument& error) {
            EXPECT_EQ(error.what(), name + message);
        }
    };
    expect_invalid(nullptr, " assign input and output pointers must not be null");
    try {
        clustering.assign(1, points.data(), nullptr, &distance);
        FAIL() << "Accepted a null output";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(
            error.what(), name + " assign input and output pointers must not be null"
        );
    }
    std::vector<float> invalid(points.begin(), points.begin() + kDim);
    invalid.back() = std::numeric_limits<float>::infinity();
    expect_invalid(invalid.data(), " x must contain only finite values");
    invalid.back() = 1e8F;
    expect_invalid(invalid.data(), " x exceeds the safe float32 coordinate range");

    // Public centroids are validated as inputs, also before an empty assignment.
    const auto trained = clustering.centroids;
    for (const float value : {std::numeric_limits<float>::quiet_NaN(), 1e8F}) {
        clustering.centroids = trained;
        clustering.centroids[kDim + 1] = value;
        const std::string message =
            std::isnan(value) ? " centroids must contain only finite values"
                              : " centroids exceeds the safe float32 coordinate range";
        for (const size_t count : {size_t{0}, size_t{1}}) {
            assignment = kPidMax;
            try {
                clustering.assign(count, points.data(), &assignment, &distance);
                FAIL() << "Accepted" << message;
            } catch (const std::invalid_argument& error) {
                EXPECT_EQ(error.what(), name + message);
            }
            EXPECT_EQ(assignment, kPidMax);
        }
    }
    clustering.centroids = trained;
    EXPECT_NO_THROW(clustering.assign(1, points.data(), &assignment, &distance));

    // A centroid buffer that no longer matches k is not a trained model.
    clustering.k = kClusterCount - 1;
    EXPECT_THROW(
        clustering.assign(1, points.data(), &assignment, &distance), std::runtime_error
    );
}

// With sampling, train's final pass is assign() itself, including QGKMeans'
// cached graph. Without it, Exact assignment and RaBitQKMeans' scan around the
// stored training center also reproduce train(); graph training uses hints.
template <typename Clustering>
void check_assign_matches_final_training_assignment() {
    constexpr bool kGraph = std::is_same_v<Clustering, qgkmeans::QGKMeans>;
    constexpr size_t kClusterCount = kClusters<Clustering>;
    constexpr size_t kMaxPoints = 2;
    for (const uint32_t threads : {1U, 4U}) {
        for (const uint32_t bits :
             kGraph ? std::vector<uint32_t>{0, 4, 8} : std::vector<uint32_t>{0}) {
            for (const size_t num_points :
                 {kClusterCount * kMaxPoints, 3 * kClusterCount * kMaxPoints + 1}) {
                const auto points = clustered_points(num_points, 67);
                for (const auto mode :
                     {FinalAssignmentMode::Approximate, FinalAssignmentMode::Exact}) {
                    SCOPED_TRACE(
                        ::testing::Message() << threads << "/" << bits << "/" << num_points
                                             << "/" << static_cast<int>(mode)
                    );
                    auto parameters = test_parameters<Clustering>();
                    parameters.num_threads = threads;
                    if constexpr (kGraph) {
                        parameters.quantization_bits = bits;
                    }
                    parameters.final_assignment = mode;
                    parameters.max_points_per_centroid = kMaxPoints;
                    Clustering clustering(kDim, kClusterCount, parameters);
                    clustering.train(num_points, points.data());
                    const bool sampled = num_points > kClusterCount * kMaxPoints;
                    const auto [assignments, distances] = assign_all(clustering, points);
                    if (sampled || mode == FinalAssignmentMode::Exact || !kGraph) {
                        EXPECT_EQ(assignments, clustering.assignments);
                        EXPECT_EQ(distances, clustering.distances);
                    }
                    expect_points_assigned(clustering, points, assignments, distances);

                    const auto unseen = clustered_points(45, 71);
                    const auto [new_assignments, new_distances] =
                        assign_all(clustering, unseen);
                    expect_points_assigned(
                        clustering, unseen, new_assignments, new_distances
                    );
                }
            }
        }
    }
}

// Each vector's label depends only on the model, not on the other vectors in
// a call or on earlier calls.
template <typename Clustering>
void check_assign_is_independent_of_the_batch() {
    constexpr size_t kClusterCount = kClusters<Clustering>;
    const auto points = clustered_points(6 * kClusterCount, 73);
    const auto queries = clustered_points(70, 79);
    for (const auto mode : {FinalAssignmentMode::Approximate, FinalAssignmentMode::Exact}) {
        SCOPED_TRACE(static_cast<int>(mode));
        auto parameters = test_parameters<Clustering>();
        parameters.num_threads = 4;
        parameters.final_assignment = mode;
        Clustering clustering(kDim, kClusterCount, parameters);
        clustering.train(6 * kClusterCount, points.data());
        const auto [assignments, distances] = assign_all(clustering, queries);
        for (const size_t split : {size_t{1}, size_t{33}}) {
            std::vector<PID> split_assignments(queries.size() / kDim);
            std::vector<float> split_distances(split_assignments.size());
            for (size_t first = 0; first < split_assignments.size(); first += split) {
                const size_t count = std::min(split, split_assignments.size() - first);
                clustering.assign(
                    count,
                    queries.data() + first * kDim,
                    split_assignments.data() + first,
                    split_distances.data() + first
                );
            }
            EXPECT_EQ(split_assignments, assignments) << split;
            EXPECT_EQ(split_distances, distances) << split;
        }
    }
}

// The cached graph follows the public centroids and graph settings. A stale
// graph would report distances to the previous centroids.
void check_graph_cache_follows_centroids_and_settings() {
    using qgkmeans::QGKMeans;
    constexpr size_t kClusterCount = kClusters<QGKMeans>;
    const auto points = clustered_points(6 * kClusterCount, 83);
    const auto queries = clustered_points(60, 89);
    for (const uint32_t bits : {0U, 4U}) {
        SCOPED_TRACE(bits);
        auto parameters = test_parameters<QGKMeans>();
        parameters.quantization_bits = bits;
        QGKMeans clustering(kDim, kClusterCount, parameters);
        clustering.train(6 * kClusterCount, points.data());
        QGKMeans fresh = clustering;  // Copied before any graph is cached.
        const auto first = assign_all(clustering, queries);
        EXPECT_EQ(assign_all(clustering, queries), first);
        EXPECT_EQ(assign_all(fresh, queries), first);

        fresh = clustering;
        for (QGKMeans* model : {&clustering, &fresh}) {
            std::swap_ranges(
                model->centroids.begin(),
                model->centroids.begin() + kDim,
                model->centroids.begin() + kDim
            );
            for (size_t cluster = 0; cluster < kClusterCount; ++cluster) {
                model->centroids[cluster * kDim] += 0.5F;
            }
        }
        const auto [moved_assignments, moved_distances] = assign_all(clustering, queries);
        expect_points_assigned(clustering, queries, moved_assignments, moved_distances);
        EXPECT_EQ(
            assign_all(fresh, queries), std::make_pair(moved_assignments, moved_distances)
        );

        clustering.ef_search = fresh.ef_search = 33;
        const auto wider = assign_all(clustering, queries);
        expect_points_assigned(clustering, queries, wider.first, wider.second);
        EXPECT_EQ(assign_all(fresh, queries), wider);
    }
}

template <typename Clustering>
void check_boundary_without_sampling() {
    constexpr size_t kClusterCount = kClusters<Clustering>;
    constexpr size_t kMaxPoints = 2;
    const size_t num_points = kClusterCount * kMaxPoints;
    const auto points = clustered_points(num_points, 11);
    for (const auto mode : {FinalAssignmentMode::Approximate, FinalAssignmentMode::Exact}) {
        SCOPED_TRACE(static_cast<int>(mode));
        auto parameters = test_parameters<Clustering>();
        parameters.final_assignment = mode;
        parameters.max_points_per_centroid = kMaxPoints;
        Clustering at_limit(kDim, kClusterCount, parameters);
        at_limit.train(num_points, points.data());
        parameters.max_points_per_centroid = 0;
        Clustering disabled(kDim, kClusterCount, parameters);
        disabled.train(num_points, points.data());
        expect_identical_results(at_limit, disabled);
        expect_all_points_assigned(at_limit, points, num_points);
    }
}

// Verbose training reports sampling only when n exceeds k * max_points_per_centroid.
template <typename Clustering>
void check_verbose_sampling_threshold(const std::string& name) {
    constexpr size_t kClusterCount = kClusters<Clustering>;
    constexpr size_t kMaxPoints = 2;
    constexpr size_t kLimit = kClusterCount * kMaxPoints;
    const auto points = clustered_points(kLimit + 1, 13);
    auto parameters = test_parameters<Clustering>();
    parameters.niter = 1;
    parameters.verbose = true;
    parameters.max_points_per_centroid = kMaxPoints;
    for (const size_t num_points : {kLimit, kLimit + 1}) {
        SCOPED_TRACE(num_points);
        Clustering clustering(kDim, kClusterCount, parameters);
        testing::internal::CaptureStdout();
        clustering.train(num_points, points.data());
        const std::string output = testing::internal::GetCapturedStdout();
        const std::string message = name + " sampling " + std::to_string(kLimit) + " of " +
                                    std::to_string(num_points) + " training points\n";
        EXPECT_EQ(output.find(message) != std::string::npos, num_points > kLimit) << output;
        EXPECT_EQ(output.find("sampling") != std::string::npos, num_points > kLimit)
            << output;
    }
}

// Initialization and Lloyd iterations see only the sample, so they match
// unsampled training on the gathered sample; the final assignment covers all points.
template <typename Clustering>
void check_sampled_training() {
    constexpr size_t kClusterCount = kClusters<Clustering>;
    constexpr size_t kMaxPoints = 2;
    constexpr size_t kNumSamples = kClusterCount * kMaxPoints;
    for (const size_t num_points : {kNumSamples + 1, 3 * kNumSamples + 5}) {
        const auto points = clustered_points(num_points, 23);
        for (const bool spherical : {false, true}) {
            for (const auto mode :
                 {FinalAssignmentMode::Approximate, FinalAssignmentMode::Exact}) {
                SCOPED_TRACE(
                    ::testing::Message()
                    << num_points << "/" << spherical << "/" << static_cast<int>(mode)
                );
                auto parameters = test_parameters<Clustering>();
                parameters.spherical = spherical;
                parameters.final_assignment = mode;
                parameters.max_points_per_centroid = kMaxPoints;
                Clustering sampled(kDim, kClusterCount, parameters);
                sampled.train(num_points, points.data());

                const auto indices =
                    sample_training_points(num_points, kNumSamples, parameters.seed);
                std::vector<float> samples;
                for (const size_t index : indices) {
                    samples.insert(
                        samples.end(),
                        points.begin() + static_cast<std::ptrdiff_t>(index * kDim),
                        points.begin() + static_cast<std::ptrdiff_t>((index + 1) * kDim)
                    );
                }
                parameters.max_points_per_centroid = 0;
                Clustering reference(kDim, kClusterCount, parameters);
                reference.train(kNumSamples, samples.data());

                EXPECT_EQ(sampled.centroids, reference.centroids);
                ASSERT_EQ(sampled.iteration_stats.size(), reference.iteration_stats.size());
                for (size_t i = 0; i < sampled.iteration_stats.size(); ++i) {
                    EXPECT_EQ(
                        sampled.iteration_stats[i].obj, reference.iteration_stats[i].obj
                    );
                    EXPECT_EQ(
                        sampled.iteration_stats[i].nsplit,
                        reference.iteration_stats[i].nsplit
                    );
                }
                expect_all_points_assigned(sampled, points, num_points);
            }
        }
    }
}

template <typename Clustering>
void check_reproducible_sampling() {
    constexpr size_t kClusterCount = kClusters<Clustering>;
    const size_t num_points = 6 * kClusterCount + 3;
    const auto points = clustered_points(num_points, 37);
    for (const auto mode : {FinalAssignmentMode::Approximate, FinalAssignmentMode::Exact}) {
        SCOPED_TRACE(static_cast<int>(mode));
        auto parameters = test_parameters<Clustering>();
        parameters.final_assignment = mode;
        parameters.max_points_per_centroid = 2;
        Clustering first(kDim, kClusterCount, parameters);
        first.train(num_points, points.data());
        Clustering second(kDim, kClusterCount, parameters);
        second.train(num_points, points.data());
        expect_identical_results(first, second);

        parameters.seed += 1;
        Clustering reseeded(kDim, kClusterCount, parameters);
        reseeded.train(num_points, points.data());
        EXPECT_NE(reseeded.centroids, first.centroids);
    }
}

// Validation covers every input point, not only those selected for training.
template <typename Clustering>
void check_rejects_nonfinite_values_outside_sample(const std::string& name) {
    constexpr size_t kClusterCount = kClusters<Clustering>;
    const size_t num_points = 3 * kClusterCount;
    auto parameters = test_parameters<Clustering>();
    parameters.max_points_per_centroid = 1;
    const auto points = clustered_points(num_points, 41);
    Clustering clustering(kDim, kClusterCount, parameters);
    clustering.train(num_points, points.data());
    const auto expected = clustering;

    const auto indices = sample_training_points(num_points, kClusterCount, parameters.seed);
    size_t outside = 0;
    while (std::binary_search(indices.begin(), indices.end(), outside)) {
        ++outside;
    }
    auto invalid = points;
    invalid[outside * kDim] = std::numeric_limits<float>::quiet_NaN();
    try {
        clustering.train(num_points, invalid.data());
        FAIL() << "Accepted nonfinite input outside the sample";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(error.what(), name + " x must contain only finite values");
    }
    expect_identical_results(clustering, expected);
}

// Retraining on a subrange of the previous centroids samples from an owned copy.
template <typename Clustering>
void check_sampled_retraining_on_own_centroids() {
    constexpr size_t kClusterCount = kClusters<Clustering>;
    constexpr size_t kInitialClusters = 2 * kClusterCount + 4;
    const auto points = clustered_points(3 * kInitialClusters, 53);
    auto parameters = test_parameters<Clustering>();
    Clustering clustering(kDim, kInitialClusters, parameters);
    clustering.train(3 * kInitialClusters, points.data());
    const auto previous = clustering.centroids;
    const size_t count = kInitialClusters - 1;

    clustering.max_points_per_centroid = 1;
    parameters.max_points_per_centroid = 1;
    Clustering reference(kDim, kClusterCount, parameters);
    reference.train(count, previous.data() + kDim);
    clustering.k = kClusterCount;
    clustering.train(count, clustering.centroids.data() + kDim);
    expect_identical_results(clustering, reference);
    ASSERT_EQ(clustering.assignments.size(), count);
}

TEST(KMeansSamplingTest, IsDisabledByDefault) {
    EXPECT_EQ(rabitqkmeans::RaBitQKMeansParameters{}.max_points_per_centroid, 0U);
    EXPECT_EQ(qgkmeans::QGKMeansParameters{}.max_points_per_centroid, 0U);
}

TEST(KMeansSamplingTest, SampleIndicesAreSortedDistinctAndSeeded) {
    const auto indices = sample_training_points(1000, 100, 42);
    ASSERT_EQ(indices.size(), 100U);
    EXPECT_TRUE(std::is_sorted(indices.begin(), indices.end()));
    EXPECT_EQ(std::adjacent_find(indices.begin(), indices.end()), indices.end());
    EXPECT_LT(indices.back(), 1000U);
    EXPECT_EQ(sample_training_points(1000, 100, 42), indices);
    EXPECT_NE(sample_training_points(1000, 100, 43), indices);

    std::vector<size_t> all(10);
    std::iota(all.begin(), all.end(), size_t{0});
    EXPECT_EQ(sample_training_points(10, 10, 42), all);
}

TEST(KMeansSamplingTest, RaBitQBoundaryWithoutSamplingMatchesDisabledSampling) {
    check_boundary_without_sampling<rabitqkmeans::RaBitQKMeans>();
}

TEST(KMeansSamplingTest, QGBoundaryWithoutSamplingMatchesDisabledSampling) {
    check_boundary_without_sampling<qgkmeans::QGKMeans>();
}

TEST(KMeansSamplingTest, RaBitQSamplesOnlyAboveTheLimit) {
    check_verbose_sampling_threshold<rabitqkmeans::RaBitQKMeans>("RaBitQKMeans");
}

TEST(KMeansSamplingTest, QGSamplesOnlyAboveTheLimit) {
    check_verbose_sampling_threshold<qgkmeans::QGKMeans>("QGKMeans");
}

TEST(KMeansSamplingTest, RaBitQTrainsOnSampleAndAssignsAllPoints) {
    check_sampled_training<rabitqkmeans::RaBitQKMeans>();
}

TEST(KMeansSamplingTest, QGTrainsOnSampleAndAssignsAllPoints) {
    check_sampled_training<qgkmeans::QGKMeans>();
}

TEST(KMeansSamplingTest, RaBitQSamplingIsReproducibleForASeed) {
    check_reproducible_sampling<rabitqkmeans::RaBitQKMeans>();
}

TEST(KMeansSamplingTest, QGSamplingIsReproducibleForASeed) {
    check_reproducible_sampling<qgkmeans::QGKMeans>();
}

TEST(KMeansSamplingTest, RaBitQRejectsNonfiniteValuesOutsideTheSample) {
    check_rejects_nonfinite_values_outside_sample<rabitqkmeans::RaBitQKMeans>(
        "RaBitQKMeans"
    );
}

TEST(KMeansSamplingTest, QGRejectsNonfiniteValuesOutsideTheSample) {
    check_rejects_nonfinite_values_outside_sample<qgkmeans::QGKMeans>("QGKMeans");
}

TEST(KMeansAssignTest, RaBitQRequiresTrainingAndValidInput) {
    check_assign_requires_training_and_valid_input<rabitqkmeans::RaBitQKMeans>(
        "RaBitQKMeans"
    );
}

TEST(KMeansAssignTest, QGRequiresTrainingAndValidInput) {
    check_assign_requires_training_and_valid_input<qgkmeans::QGKMeans>("QGKMeans");
}

TEST(KMeansAssignTest, RaBitQMatchesFinalTrainingAssignment) {
    check_assign_matches_final_training_assignment<rabitqkmeans::RaBitQKMeans>();
}

TEST(KMeansAssignTest, QGMatchesFinalTrainingAssignment) {
    check_assign_matches_final_training_assignment<qgkmeans::QGKMeans>();
}

TEST(KMeansAssignTest, RaBitQIsIndependentOfTheBatch) {
    check_assign_is_independent_of_the_batch<rabitqkmeans::RaBitQKMeans>();
}

TEST(KMeansAssignTest, QGIsIndependentOfTheBatch) {
    check_assign_is_independent_of_the_batch<qgkmeans::QGKMeans>();
}

TEST(KMeansAssignTest, QGGraphCacheFollowsCentroidsAndSettings) {
    check_graph_cache_follows_centroids_and_settings();
}

TEST(KMeansSamplingTest, RaBitQSampledRetrainingOnOwnCentroids) {
    check_sampled_retraining_on_own_centroids<rabitqkmeans::RaBitQKMeans>();
}

TEST(KMeansSamplingTest, QGSampledRetrainingOnOwnCentroids) {
    check_sampled_retraining_on_own_centroids<qgkmeans::QGKMeans>();
}

}  // namespace
}  // namespace rabitqlib
