#include "rabitqlib/clustering/qgemeans.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/index/types.hpp"

namespace rabitqlib::qgemeans {
namespace {

constexpr size_t kDim = 64;
constexpr size_t kClusters = 40;
constexpr size_t kPointsPerCluster = 30;
constexpr size_t kCount = kClusters * kPointsPerCluster;

// Well-separated Gaussian clusters around random centers.
std::vector<float> clustered_data(
    size_t clusters = kClusters, size_t per_cluster = kPointsPerCluster, uint32_t seed = 7
) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> center(0.0F, 4.0F);
    std::normal_distribution<float> noise(0.0F, 0.1F);
    std::vector<float> centers(clusters * kDim);
    for (auto& value : centers) {
        value = center(rng);
    }
    std::vector<float> data(clusters * per_cluster * kDim);
    for (size_t point = 0; point < clusters * per_cluster; ++point) {
        const size_t cluster = point % clusters;
        for (size_t dim = 0; dim < kDim; ++dim) {
            data[point * kDim + dim] = centers[cluster * kDim + dim] + noise(rng);
        }
    }
    return data;
}

QGEMeansParameters small_parameters(uint32_t threads = 2) {
    QGEMeansParameters parameters;
    parameters.epochs = 4;
    parameters.ef_build = 40;
    parameters.ef_search = 16;
    parameters.num_threads = threads;
    return parameters;
}

double squared_distance(const float* a, const float* b) {
    double sum = 0.0;
    for (size_t dim = 0; dim < kDim; ++dim) {
        const double difference = static_cast<double>(a[dim]) - b[dim];
        sum += difference * difference;
    }
    return sum;
}

template <typename Function>
void expect_invalid_argument(Function&& function, const std::string& message) {
    try {
        function();
        FAIL() << "expected std::invalid_argument: " << message;
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()), message);
    }
}

TEST(QGEMeansTest, UsesDocumentedDefaults) {
    const QGEMeansParameters parameters;
    EXPECT_EQ(parameters.epochs, 8U);
    EXPECT_EQ(parameters.batch_size, 0U);
    EXPECT_FALSE(parameters.verbose);
    EXPECT_EQ(parameters.seed, 42U);
    EXPECT_EQ(parameters.final_assignment, FinalAssignmentMode::Approximate);
    EXPECT_EQ(parameters.graph_degree, 32U);
    EXPECT_EQ(parameters.ef_build, 240U);
    EXPECT_EQ(parameters.ef_search, 16U);
    EXPECT_EQ(parameters.graph_build_iterations, 1U);
    EXPECT_EQ(parameters.num_threads, 0U);
}

TEST(QGEMeansRecipeTest, FollowsBatchSizeAndStepLaws) {
    // GIST1M with k = 4096 over 8 epochs: the law's batch of 2^17 vectors.
    const auto gist = detail::resolve_recipe(1000000, 4096, 8, 0);
    EXPECT_EQ(gist.batch_size, 131072U);
    EXPECT_EQ(gist.steps_per_epoch, 8U);
    EXPECT_EQ(gist.total_steps, 64U);
    const double expected_alpha = 0.8244 * std::pow(4096.0, -0.4617) *
                                  std::pow(131072.0, 0.9003) * std::pow(1e6, -0.4498) *
                                  std::pow(8.0, -0.1057);
    EXPECT_DOUBLE_EQ(gist.alpha0, std::min(std::max(expected_alpha, 1e-4), 1.0));

    // An explicit batch is kept, and capped at n.
    EXPECT_EQ(detail::resolve_recipe(1000, 40, 2, 64).batch_size, 64U);
    const auto capped = detail::resolve_recipe(1000, 40, 2, 5000);
    EXPECT_EQ(capped.batch_size, 1000U);
    EXPECT_EQ(capped.steps_per_epoch, 1U);
    EXPECT_EQ(capped.total_steps, 2U);

    // The law never exceeds half the vectors, rounded to a power of two, and the
    // step size stays in [1e-4, 1].
    for (size_t n : {33U, 100U, 1000U, 100000U}) {
        const auto recipe = detail::resolve_recipe(n, 33, 8, 0);
        EXPECT_GE(recipe.batch_size, 1U);
        EXPECT_LE(recipe.batch_size, n);
        EXPECT_EQ(recipe.batch_size & (recipe.batch_size - 1), 0U);
        EXPECT_GE(recipe.alpha0, 1e-4);
        EXPECT_LE(recipe.alpha0, 1.0);
        EXPECT_EQ(recipe.steps_per_epoch, (n + recipe.batch_size - 1) / recipe.batch_size);
    }
}

TEST(QGEMeansRecipeTest, RoundsToTheNearestPowerOfTwo) {
    EXPECT_EQ(detail::round_power_of_two(0.5), 1U);
    EXPECT_EQ(detail::round_power_of_two(1.0), 1U);
    EXPECT_EQ(detail::round_power_of_two(2.8), 2U);
    EXPECT_EQ(detail::round_power_of_two(2.9), 4U);
    EXPECT_EQ(detail::round_power_of_two(1024.0), 1024U);
    EXPECT_EQ(detail::round_power_of_two(1500.0), 2048U);
}

TEST(QGEMeansRecipeTest, CosineScheduleDecaysToATenthOfTheInitialStep) {
    EXPECT_DOUBLE_EQ(detail::scheduled_alpha(0.5, 0, 10), 0.5);
    EXPECT_DOUBLE_EQ(detail::scheduled_alpha(0.5, 10, 10), 0.05);
    EXPECT_DOUBLE_EQ(detail::scheduled_alpha(0.5, 5, 10), 0.5 * (0.1 + (0.9 * 0.5)));
    // Steps past the end stay at the floor.
    EXPECT_DOUBLE_EQ(detail::scheduled_alpha(0.5, 20, 10), 0.05);
}

TEST(QGEMeansTest, RejectsInvalidParametersWithDescriptiveMessages) {
    const auto with = [](auto change) {
        QGEMeansParameters parameters = small_parameters();
        change(parameters);
        return parameters;
    };
    expect_invalid_argument(
        [&] { QGEMeans(kDim, kClusters, with([](auto& p) { p.epochs = 0; })); },
        "QGEMeans d, k, and epochs must be positive"
    );
    expect_invalid_argument(
        [&] { QGEMeans(kDim, 0, small_parameters()); },
        "QGEMeans d, k, and epochs must be positive"
    );
    expect_invalid_argument(
        [&] { QGEMeans(63, kClusters, small_parameters()); },
        "QGEMeans currently requires dimensions in the range [64, 65536]"
    );
    expect_invalid_argument(
        [&] { QGEMeans(65537, kClusters, small_parameters()); },
        "QGEMeans currently requires dimensions in the range [64, 65536]"
    );
    expect_invalid_argument(
        [&] {
            QGEMeans(kDim, kClusters, with([](auto& p) {
                         p.num_threads =
                             static_cast<uint32_t>(std::numeric_limits<int>::max()) + 1U;
                     }));
        },
        "QGEMeans num_threads must fit in an OpenMP int"
    );
    expect_invalid_argument(
        [&] {
            QGEMeans(kDim, kClusters, with([](auto& p) {
                         p.final_assignment = static_cast<FinalAssignmentMode>(7);
                     }));
        },
        "QGEMeans final_assignment mode is invalid"
    );
    for (uint32_t degree : {0U, 48U}) {
        expect_invalid_argument(
            [&] {
                QGEMeans(kDim, kClusters, with([degree](auto& p) {
                             p.graph_degree = degree;
                         }));
            },
            "QGEMeans graph degree must be a positive multiple of 32"
        );
    }
    expect_invalid_argument(
        [&] { QGEMeans(kDim, 32, small_parameters()); },
        "QGEMeans requires more centroids than the graph degree"
    );
    expect_invalid_argument(
        [&] { QGEMeans(kDim, kClusters, with([](auto& p) { p.ef_build = 0; })); },
        "QGEMeans ef_build and ef_search must be positive"
    );
    expect_invalid_argument(
        [&] { QGEMeans(kDim, kClusters, with([](auto& p) { p.ef_search = 0; })); },
        "QGEMeans ef_build and ef_search must be positive"
    );
    expect_invalid_argument(
        [&] {
            QGEMeans(kDim, kClusters, with([](auto& p) { p.graph_build_iterations = 0; }));
        },
        "QGEMeans graph_build_iterations must be positive"
    );

    // Parameters changed after construction are checked again by train.
    QGEMeans model(kDim, kClusters, small_parameters());
    model.epochs = 0;
    const auto data = clustered_data();
    expect_invalid_argument(
        [&] { model.train(kCount, data.data()); },
        "QGEMeans d, k, and epochs must be positive"
    );
}

TEST(QGEMeansTest, RejectsInvalidInputAndKeepsPreviousResults) {
    auto data = clustered_data();
    QGEMeans model(kDim, kClusters, small_parameters());
    expect_invalid_argument(
        [&] { model.train(kCount, nullptr); }, "QGEMeans x must not be null"
    );
    expect_invalid_argument(
        [&] { model.train(kClusters - 1, data.data()); }, "QGEMeans requires n >= k"
    );

    model.train(kCount, data.data());
    const auto centroids = model.centroids;
    const auto assignments = model.assignments;
    const double objective = model.final_obj;

    auto bad = data;
    bad[5] = std::numeric_limits<float>::quiet_NaN();
    expect_invalid_argument(
        [&] { model.train(kCount, bad.data()); },
        "QGEMeans x must contain only finite values"
    );
    bad[5] = 1e30F;
    expect_invalid_argument(
        [&] { model.train(kCount, bad.data()); },
        "QGEMeans x exceeds the safe float32 coordinate range"
    );
    EXPECT_EQ(model.centroids, centroids);
    EXPECT_EQ(model.assignments, assignments);
    EXPECT_EQ(model.final_obj, objective);
}

TEST(QGEMeansTest, AssignRequiresTrainingAndValidPointers) {
    const auto data = clustered_data();
    QGEMeans model(kDim, kClusters, small_parameters());
    std::vector<PID> ids(kCount);
    std::vector<float> distances(kCount);
    try {
        model.assign(kCount, data.data(), ids.data(), distances.data());
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), "QGEMeans has not been trained");
    }

    model.train(kCount, data.data());
    EXPECT_NO_THROW(model.assign(0, nullptr, nullptr, nullptr));
    expect_invalid_argument(
        [&] { model.assign(kCount, data.data(), nullptr, distances.data()); },
        "QGEMeans assign input and output pointers must not be null"
    );
    auto bad = data;
    bad[0] = std::numeric_limits<float>::infinity();
    expect_invalid_argument(
        [&] { model.assign(kCount, bad.data(), ids.data(), distances.data()); },
        "QGEMeans x must contain only finite values"
    );
}

TEST(QGEMeansTest, RecordsOneStatisticPerEpoch) {
    const auto data = clustered_data();
    QGEMeans model(kDim, kClusters, small_parameters());
    model.train(kCount, data.data());
    ASSERT_EQ(model.iteration_stats.size(), model.epochs);
    for (size_t epoch = 0; epoch < model.epochs; ++epoch) {
        const auto& stats = model.iteration_stats[epoch];
        EXPECT_EQ(stats.iteration, epoch + 1);
        EXPECT_GT(stats.obj, 0.0);
        EXPECT_GE(stats.shift, 0.0);
    }
    ASSERT_EQ(model.centroids.size(), kClusters * kDim);
    ASSERT_EQ(model.assignments.size(), kCount);
    ASSERT_EQ(model.distances.size(), kCount);
}

TEST(QGEMeansTest, ReturnsExactDistancesForItsAssignments) {
    const auto data = clustered_data();
    QGEMeans model(kDim, kClusters, small_parameters());
    model.train(kCount, data.data());
    double objective = 0.0;
    for (size_t point = 0; point < kCount; ++point) {
        const PID cluster = model.assignments[point];
        ASSERT_LT(cluster, kClusters);
        const double expected = squared_distance(
            data.data() + point * kDim, model.centroids.data() + cluster * kDim
        );
        EXPECT_NEAR(model.distances[point], expected, 1e-4 * (1.0 + expected));
        objective += model.distances[point];
    }
    EXPECT_NEAR(model.final_obj, objective, 1e-6 * objective);
}

TEST(QGEMeansTest, ExactFinalAssignmentReturnsTheNearestCentroid) {
    const auto data = clustered_data();
    QGEMeansParameters parameters = small_parameters();
    parameters.final_assignment = FinalAssignmentMode::Exact;
    QGEMeans model(kDim, kClusters, parameters);
    model.train(kCount, data.data());
    for (size_t point = 0; point < kCount; ++point) {
        const float* vector = data.data() + point * kDim;
        double best = std::numeric_limits<double>::infinity();
        PID best_cluster = 0;
        for (size_t cluster = 0; cluster < kClusters; ++cluster) {
            const double distance =
                squared_distance(vector, model.centroids.data() + cluster * kDim);
            if (distance < best) {
                best = distance;
                best_cluster = static_cast<PID>(cluster);
            }
        }
        EXPECT_EQ(model.assignments[point], best_cluster) << "point " << point;
    }
}

TEST(QGEMeansTest, ApproximateAssignmentAgreesWithExactOnSeparatedClusters) {
    const auto data = clustered_data();
    QGEMeansParameters parameters = small_parameters();
    QGEMeans approximate(kDim, kClusters, parameters);
    approximate.train(kCount, data.data());
    parameters.final_assignment = FinalAssignmentMode::Exact;
    QGEMeans exact(kDim, kClusters, parameters);
    exact.train(kCount, data.data());
    // The final assignment does not change the trained centroids.
    EXPECT_EQ(approximate.centroids, exact.centroids);
    EXPECT_EQ(approximate.iteration_stats.size(), exact.iteration_stats.size());
    size_t agree = 0;
    for (size_t point = 0; point < kCount; ++point) {
        agree += approximate.assignments[point] == exact.assignments[point] ? 1 : 0;
    }
    EXPECT_GE(agree, kCount * 99 / 100);
    EXPECT_GE(approximate.final_obj, exact.final_obj * (1.0 - 1e-6));
}

TEST(QGEMeansTest, IsDeterministicAcrossRunsAndThreadCounts) {
    const auto data = clustered_data();
    QGEMeans single(kDim, kClusters, small_parameters(1));
    single.train(kCount, data.data());
    for (uint32_t threads : {1U, 4U}) {
        QGEMeans model(kDim, kClusters, small_parameters(threads));
        model.train(kCount, data.data());
        EXPECT_EQ(model.centroids, single.centroids) << threads << " threads";
        EXPECT_EQ(model.assignments, single.assignments) << threads << " threads";
        EXPECT_EQ(model.distances, single.distances) << threads << " threads";
        ASSERT_EQ(model.iteration_stats.size(), single.iteration_stats.size());
        for (size_t epoch = 0; epoch < model.iteration_stats.size(); ++epoch) {
            EXPECT_EQ(model.iteration_stats[epoch].obj, single.iteration_stats[epoch].obj);
            EXPECT_EQ(
                model.iteration_stats[epoch].shift, single.iteration_stats[epoch].shift
            );
            EXPECT_EQ(
                model.iteration_stats[epoch].nsplit, single.iteration_stats[epoch].nsplit
            );
        }
    }
    // A different seed trains different centroids.
    QGEMeansParameters reseeded = small_parameters(1);
    reseeded.seed = 43;
    QGEMeans other(kDim, kClusters, reseeded);
    other.train(kCount, data.data());
    EXPECT_NE(other.centroids, single.centroids);
}

TEST(QGEMeansTest, TrainCentroidsMatchesTrainWithoutFinalAssignment) {
    const auto data = clustered_data();
    QGEMeans full(kDim, kClusters, small_parameters());
    full.train(kCount, data.data());
    QGEMeans centroids_only(kDim, kClusters, small_parameters());
    centroids_only.train_centroids(kCount, data.data());
    EXPECT_EQ(centroids_only.centroids, full.centroids);
    EXPECT_TRUE(centroids_only.assignments.empty());
    EXPECT_TRUE(centroids_only.distances.empty());
    EXPECT_EQ(centroids_only.final_obj, 0.0);

    std::vector<PID> ids(kCount);
    std::vector<float> distances(kCount);
    centroids_only.assign(kCount, data.data(), ids.data(), distances.data());
    EXPECT_EQ(ids, full.assignments);
    EXPECT_EQ(distances, full.distances);
}

TEST(QGEMeansTest, AssignReproducesTrainingAssignmentsInBothModes) {
    const auto data = clustered_data();
    for (auto mode : {FinalAssignmentMode::Approximate, FinalAssignmentMode::Exact}) {
        QGEMeansParameters parameters = small_parameters();
        parameters.final_assignment = mode;
        QGEMeans model(kDim, kClusters, parameters);
        model.train(kCount, data.data());
        std::vector<PID> ids(kCount);
        std::vector<float> distances(kCount);
        model.assign(kCount, data.data(), ids.data(), distances.data());
        EXPECT_EQ(ids, model.assignments);
        EXPECT_EQ(distances, model.distances);
        // Repeated calls reuse the cached centroid graph and agree.
        std::vector<PID> again(kCount);
        model.assign(kCount, data.data(), again.data(), distances.data());
        EXPECT_EQ(again, ids);
        // Each point's result is independent of the others in the call.
        PID single = 0;
        float single_distance = 0;
        model.assign(1, data.data() + 17 * kDim, &single, &single_distance);
        EXPECT_EQ(single, ids[17]);
    }
}

TEST(QGEMeansTest, ExplicitBatchSizesIncludingOneVectorPerBatch) {
    const auto data = clustered_data(kClusters, 3);
    const size_t count = kClusters * 3;
    for (size_t batch : {1U, 7U, 1000000U}) {
        QGEMeansParameters parameters = small_parameters();
        parameters.epochs = 2;
        parameters.batch_size = batch;
        QGEMeans model(kDim, kClusters, parameters);
        model.train(count, data.data());
        EXPECT_EQ(model.iteration_stats.size(), 2U);
        EXPECT_EQ(model.assignments.size(), count);
        for (float value : model.centroids) {
            ASSERT_TRUE(std::isfinite(value));
        }
    }
}

TEST(QGEMeansTest, TrainsWithAsManyCentroidsAsVectors) {
    const auto data = clustered_data(kClusters, 1);
    QGEMeans model(kDim, kClusters, small_parameters());
    model.train(kClusters, data.data());
    EXPECT_EQ(model.assignments.size(), kClusters);
    for (float value : model.centroids) {
        ASSERT_TRUE(std::isfinite(value));
    }
}

TEST(QGEMeansTest, ConstantDataKeepsFiniteCentroidsAndZeroObjective) {
    // Every centroid starts at the same point; revival splits zero donors.
    const std::vector<float> data(kCount * kDim, 0.0F);
    QGEMeans model(kDim, kClusters, small_parameters());
    model.train(kCount, data.data());
    for (float value : model.centroids) {
        ASSERT_TRUE(std::isfinite(value));
    }
    EXPECT_NEAR(model.final_obj, 0.0, 1e-20);
}

TEST(QGEMeansTest, RetrainsOnItsOwnCentroidsLikeAnIndependentCopy) {
    const auto data = clustered_data();
    QGEMeans model(kDim, kClusters, small_parameters());
    model.train(kCount, data.data());
    const std::vector<float> copy = model.centroids;

    QGEMeans reference(kDim, kClusters, small_parameters());
    reference.train(kClusters, copy.data());
    // Training reads model.centroids while it trains new ones.
    model.train(kClusters, model.centroids.data());
    EXPECT_EQ(model.centroids, reference.centroids);
    EXPECT_EQ(model.assignments, reference.assignments);
}

TEST(QGEMeansTest, ReducesTheObjectiveOnSeparatedClusters) {
    const auto data = clustered_data();
    QGEMeansParameters parameters = small_parameters();
    parameters.epochs = 8;
    QGEMeans model(kDim, kClusters, parameters);
    model.train(kCount, data.data());
    EXPECT_LT(model.iteration_stats.back().obj, model.iteration_stats.front().obj);
    // Random initial centroids give a much larger objective than the trained ones.
    double initial = 0.0;
    for (size_t point = 0; point < kCount; ++point) {
        initial += squared_distance(data.data() + point * kDim, data.data());
    }
    EXPECT_LT(model.final_obj, initial / 10.0);
}

}  // namespace
}  // namespace rabitqlib::qgemeans
