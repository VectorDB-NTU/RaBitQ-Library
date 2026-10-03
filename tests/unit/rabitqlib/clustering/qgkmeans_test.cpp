#include "rabitqlib/clustering/qgkmeans.hpp"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::qgkmeans {
namespace {

TEST(QGKMeansTest, UsesTwentyFiveIterationsByDefault) {
    EXPECT_EQ(QGKMeansParameters{}.niter, 25U);
}

TEST(QGKMeansTest, UsesRawGraphStorageByDefaultAndRejectsUnsupportedBits) {
    EXPECT_EQ(QGKMeansParameters{}.quantization_bits, 0U);
    for (const uint32_t bits : {1U, 7U, 9U, 32U}) {
        QGKMeansParameters parameters;
        parameters.quantization_bits = bits;
        try {
            QGKMeans clustering(65, 33, parameters);
            FAIL() << "Accepted unsupported quantization bits " << bits;
        } catch (const std::invalid_argument& error) {
            EXPECT_STREQ(error.what(), "QGKMeans quantization_bits must be 0, 4, or 8");
        }
    }
}

TEST(QGAssignerTest, QuantizedSearchReturnsExactDistancesToOriginalCentroids) {
    constexpr size_t kDimension = 65;  // Exercise the padded quantized domain.
    constexpr size_t kClusters = 33;
    constexpr size_t kQueries = 20;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> centroids(kClusters * kDimension);
    std::vector<float> queries(kQueries * kDimension);
    std::generate(centroids.begin(), centroids.end(), [&] { return distribution(rng); });
    std::generate(queries.begin(), queries.end(), [&] { return distribution(rng); });

    for (const uint32_t bits : {4U, 8U}) {
        SCOPED_TRACE(bits);
        for (const MetricType metric : {METRIC_L2, METRIC_IP}) {
            SCOPED_TRACE(static_cast<int>(metric));
            QGAssigner assigner(kDimension, kClusters, 32, 33, 33, 2, 1, 42, metric, bits);
            std::vector<PID> labels(kQueries);
            std::vector<float> distances(kQueries);
            assigner.assign(
                centroids.data(), queries.data(), kQueries, labels.data(), distances.data()
            );

            symqg::QuantizedGraph<float> graph(
                kClusters, kDimension, 32, metric, RotatorType::FhtKacRotator, bits, 42
            );
            symqg::QGBuilder builder(graph, 33, centroids.data(), 1, 42);
            builder.build(2);
            graph.set_ef(33);
            bool saw_estimate_difference = false;
            for (size_t i = 0; i < kQueries; ++i) {
                PID expected_label = kPidMax;
                float estimated_distance = 0;
                graph.search(
                    queries.data() + i * kDimension, 1, &expected_label, &estimated_distance
                );
                ASSERT_EQ(labels[i], expected_label);
                double exact_distance = metric == METRIC_L2 ? 0.0 : 1.0;
                for (size_t dim = 0; dim < kDimension; ++dim) {
                    const double x = queries[i * kDimension + dim];
                    const double c = centroids[labels[i] * kDimension + dim];
                    exact_distance += metric == METRIC_L2 ? (x - c) * (x - c) : -x * c;
                }
                EXPECT_NEAR(distances[i], exact_distance, 1e-5);
                saw_estimate_difference |=
                    std::abs(estimated_distance - exact_distance) > 1e-5;
            }
            EXPECT_TRUE(saw_estimate_difference);
        }
    }
}

TEST(QGAssignerTest, GroupedQueriesPreserveIndependentSearchHintsAcrossRebuilds) {
    constexpr size_t kDimension = 65;
    constexpr size_t kClusters = 65;
    constexpr size_t kQueries = 129;
    constexpr uint32_t kEf = 8;
    std::mt19937 rng(823);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> initial(kClusters * kDimension), queries(kQueries * kDimension);
    std::generate(initial.begin(), initial.end(), [&] { return distribution(rng); });
    std::generate(queries.begin(), queries.end(), [&] { return distribution(rng); });
    for (const uint32_t bits : {0U, 4U, 8U}) {
        for (const MetricType metric : {METRIC_L2, METRIC_IP}) {
            for (const uint32_t threads : {1U, 4U}) {
                SCOPED_TRACE(
                    ::testing::Message() << bits << "/" << metric << "/" << threads
                );
                auto centroids = initial;
                QGAssigner assigner(
                    kDimension, kClusters, 32, 64, kEf, 1, threads, 42, metric, bits
                );
                symqg::QuantizedGraph<float> graph(
                    kClusters, kDimension, 32, metric, RotatorType::FhtKacRotator, bits, 42
                );
                symqg::QGBuilder builder(
                    graph,
                    64,
                    centroids.data(),
                    threads,
                    symqg::QGInitialization::PiPNN,
                    42,
                    true
                );
                graph.set_ef(kEf);
                std::vector<float> rotated(graph.padded_dim()), estimates(32),
                    lut(graph.padded_dim() * 4);
                BatchQuery<float> batch;
                buffer::SearchBuffer<float> search_pool(kEf), result_pool(1);
                VisitedSet visited;
                visited.initialize(kClusters, std::min(size_t{kEf} * kEf, kClusters / 10));
                std::vector<PID> previous;
                for (const size_t count :
                     {kQueries, kQueries, kQueries - 2, kQueries - 2}) {
                    for (float& value : centroids) {
                        value += 0.01F;
                    }
                    builder.reset(centroids.data());
                    builder.build();
                    std::vector<PID> actual(count), expected(count);
                    std::vector<float> actual_distances(count), expected_distances(count);
                    assigner.assign(
                        centroids.data(),
                        queries.data(),
                        count,
                        actual.data(),
                        actual_distances.data()
                    );
                    const auto distance =
                        metric == METRIC_IP ? dot_product_dis<float> : euclidean_sqr<float>;
                    for (size_t point = 0; point < count; ++point) {
                        const PID hint =
                            previous.size() == count ? previous[point] : kPidMax;
                        graph.search_with_scratch(
                            queries.data() + point * kDimension,
                            1,
                            &expected[point],
                            &expected_distances[point],
                            rotated.data(),
                            estimates.data(),
                            lut.data(),
                            batch,
                            search_pool,
                            result_pool,
                            visited,
                            hint
                        );
                        if (bits != 0) {
                            expected_distances[point] = distance(
                                queries.data() + point * kDimension,
                                centroids.data() + expected[point] * kDimension,
                                kDimension
                            );
                        }
                    }
                    EXPECT_EQ(actual, expected);
                    EXPECT_EQ(actual_distances, expected_distances);
                    previous = expected;
                }
            }
        }
    }
}

TEST(QGKMeansTest, UsesSymphonyQGForFinalAssignmentByDefault) {
    EXPECT_EQ(QGKMeansParameters{}.final_assignment, FinalAssignmentMode::SymphonyQG);
}

TEST(QGKMeansTest, UsesFaissTrainingSizeWarningThresholdByDefault) {
    EXPECT_EQ(QGKMeansParameters{}.min_points_per_centroid, 39U);
}

QGKMeansParameters exact_search_parameters() {
    QGKMeansParameters parameters;
    parameters.niter = 4;
    parameters.graph_degree = 32;
    parameters.ef_build = 33;
    parameters.ef_search = 33;
    parameters.graph_build_iterations = 2;
    parameters.num_threads = 1;
    parameters.min_points_per_centroid = 1;
    return parameters;
}

QGKMeans train_qgkmeans(
    size_t d, size_t k, const QGKMeansParameters& parameters, const float* x, size_t n
) {
    QGKMeans clustering(d, k, parameters);
    clustering.train(n, x);
    return clustering;
}

std::vector<float> separated_points(size_t num_points, size_t dimension) {
    std::vector<float> data(num_points * dimension);
    for (size_t point = 0; point < num_points; ++point) {
        for (size_t dim = 0; dim < dimension; ++dim) {
            data[point * dimension + dim] =
                static_cast<float>(point * 10) + static_cast<float>(dim) / 64.0F;
        }
    }
    return data;
}

TEST(QGAssignerTest, ReusesVisitedStateAcrossQueriesAndAssignerSizes) {
    constexpr size_t kDimension = 64;
    constexpr size_t kQueries = 256;
    for (const uint32_t threads : {1U, 4U}) {
        SCOPED_TRACE(threads);
        for (const size_t clusters : {33U, 65U, 33U}) {
            SCOPED_TRACE(clusters);
            const auto centroids = separated_points(clusters, kDimension);
            std::vector<float> queries(kQueries * kDimension);
            std::vector<PID> labels(kQueries);
            std::vector<float> distances(kQueries);
            QGAssigner assigner(kDimension, clusters, 32, 128, 128, 2, threads, 42);
            for (size_t round = 0; round < 2; ++round) {
                for (size_t point = 0; point < kQueries; ++point) {
                    const size_t cluster = (point + round) % clusters;
                    std::copy_n(
                        centroids.data() + cluster * kDimension,
                        kDimension,
                        queries.data() + point * kDimension
                    );
                }
                assigner.assign(
                    centroids.data(),
                    queries.data(),
                    kQueries,
                    labels.data(),
                    distances.data()
                );
                for (size_t point = 0; point < kQueries; ++point) {
                    EXPECT_EQ(labels[point], (point + round) % clusters);
                    EXPECT_EQ(distances[point], 0.0F);
                }
            }
        }
    }
}

TEST(QGKMeansTest, WarnsWhenThereAreTooFewPointsPerCentroid) {
    constexpr size_t kClusters = 33;
    constexpr size_t kDimension = 64;
    const std::vector<float> data = separated_points(kClusters, kDimension);
    QGKMeansParameters parameters = exact_search_parameters();
    parameters.niter = 1;
    parameters.verbose = true;
    parameters.min_points_per_centroid = 39;

    testing::internal::CaptureStderr();
    QGKMeans clustering(kDimension, kClusters, parameters);
    clustering.train(kClusters, data.data());
    const std::string warning = testing::internal::GetCapturedStderr();

    EXPECT_NE(
        warning.find("please provide at least 39 training points per centroid"),
        std::string::npos
    );
}

TEST(QGKMeansTest, UsesFaissEarlyStopThresholdByDefault) {
    constexpr size_t kClusters = 33;
    constexpr size_t kDimension = 64;
    const std::vector<float> data = separated_points(kClusters, kDimension);
    QGKMeansParameters parameters = exact_search_parameters();

    EXPECT_DOUBLE_EQ(parameters.early_stop_threshold, 0.0);
}

TEST(QGKMeansTest, StopsAtFaissEarlyStopThreshold) {
    constexpr size_t kClusters = 33;
    constexpr size_t kDimension = 64;
    const std::vector<float> data = separated_points(kClusters * 2, kDimension);
    QGKMeansParameters parameters = exact_search_parameters();
    parameters.early_stop_threshold = 1.0;

    const QGKMeans result =
        train_qgkmeans(kDimension, kClusters, parameters, data.data(), kClusters * 2);

    ASSERT_EQ(result.iteration_stats.size(), 2U);
}

TEST(QGKMeansTest, ExactFinalAssignmentReturnsNearestCentroidForEveryPoint) {
    constexpr size_t kPoints = 100;
    constexpr size_t kClusters = 33;
    constexpr size_t kDimension = 64;
    std::vector<float> data(kPoints * kDimension);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> distribution(-10.0F, 10.0F);
    std::generate(data.begin(), data.end(), [&] { return distribution(rng); });

    QGKMeansParameters parameters = exact_search_parameters();
    parameters.niter = 2;
    parameters.final_assignment = FinalAssignmentMode::Exact;
    const QGKMeans result =
        train_qgkmeans(kDimension, kClusters, parameters, data.data(), kPoints);

    EXPECT_DOUBLE_EQ(
        result.final_obj,
        std::accumulate(result.distances.begin(), result.distances.end(), 0.0)
    );
    for (size_t point = 0; point < kPoints; ++point) {
        PID expected = 0;
        double expected_distance = std::numeric_limits<double>::infinity();
        for (size_t cluster = 0; cluster < kClusters; ++cluster) {
            double distance = 0.0;
            for (size_t dim = 0; dim < kDimension; ++dim) {
                const double difference =
                    static_cast<double>(data[point * kDimension + dim]) -
                    result.centroids[cluster * kDimension + dim];
                distance += difference * difference;
            }
            if (distance < expected_distance) {
                expected = static_cast<PID>(cluster);
                expected_distance = distance;
            }
        }
        EXPECT_EQ(result.assignments[point], expected);
        EXPECT_FLOAT_EQ(result.distances[point], static_cast<float>(expected_distance));
    }
}

void expect_exact_assignment_matches_scalar(
    const std::vector<float>& data,
    const std::vector<float>& centroids,
    size_t dimension,
    bool spherical
) {
    const size_t points = data.size() / dimension;
    const size_t clusters = centroids.size() / dimension;
    std::vector<PID> expected_labels(points);
    std::vector<float> expected_distances(points);
    for (size_t point = 0; point < points; ++point) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t cluster = 0; cluster < clusters; ++cluster) {
            double distance = 0.0;
            for (size_t dim = 0; dim < dimension; ++dim) {
                const double x = data[point * dimension + dim];
                const double c = centroids[cluster * dimension + dim];
                if (spherical) {
                    distance -= x * c;
                } else {
                    const double difference = x - c;
                    distance += difference * difference;
                }
            }
            if (spherical) {
                distance += 1.0;
            }
            if (distance < best) {
                best = distance;
                expected_labels[point] = static_cast<PID>(cluster);
            }
        }
        expected_distances[point] = static_cast<float>(best);
    }
    for (const uint32_t threads : {1U, 4U}) {
        SCOPED_TRACE(threads);
        // Exercise no hint, separate hints, and in-place hints as used by Lloyd.
        // The last centroid deliberately loses ties to earlier centroid IDs.
        const std::vector<PID> previous(points, static_cast<PID>(clusters - 1));
        for (const int hint_mode : {0, 1, 2}) {
            SCOPED_TRACE(hint_mode);
            std::vector<PID> labels =
                hint_mode == 2 ? previous : std::vector<PID>(points, kPidMax);
            std::vector<float> distances(points, std::numeric_limits<float>::quiet_NaN());
            const PID* hints = hint_mode == 0
                                   ? nullptr
                                   : (hint_mode == 1 ? previous.data() : labels.data());
            detail::exact_assign(
                data.data(),
                centroids.data(),
                points,
                dimension,
                clusters,
                spherical,
                threads,
                labels.data(),
                distances.data(),
                hints
            );
            EXPECT_EQ(labels, expected_labels);
            EXPECT_EQ(distances, expected_distances);
        }
    }
}

TEST(QGKMeansTest, ExactAssignmentMatchesScalarAcrossMatrixTileTails) {
    std::mt19937 rng(741);
    std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
    for (const size_t size : {65U, 129U, 257U}) {
        SCOPED_TRACE(size);
        std::vector<float> data(size * size);
        std::vector<float> centroids(257 * size);
        std::generate(data.begin(), data.end(), [&] { return distribution(rng); });
        std::generate(centroids.begin(), centroids.end(), [&] {
            return distribution(rng);
        });
        for (const bool spherical : {false, true}) {
            SCOPED_TRACE(spherical);
            expect_exact_assignment_matches_scalar(data, centroids, size, spherical);
        }
    }
}

TEST(QGKMeansTest, ExactAssignmentMatchesScalarAtMaximumDimension) {
    constexpr size_t kDimension = 65536;
    std::mt19937 rng(963);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> data(3 * kDimension);
    std::vector<float> centroids(3 * kDimension);
    std::generate(data.begin(), data.end(), [&] { return distribution(rng); });
    std::generate(centroids.begin(), centroids.end(), [&] { return distribution(rng); });
    for (const bool spherical : {false, true}) {
        SCOPED_TRACE(spherical);
        expect_exact_assignment_matches_scalar(data, centroids, kDimension, spherical);
    }
}

TEST(QGKMeansTest, ExactAssignmentMatchesScalarWithManyPointBlocks) {
    constexpr size_t kDimension = 17;
    std::mt19937 rng(385);
    std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
    std::vector<float> data(4097 * kDimension);
    std::vector<float> centroids(257 * kDimension);
    std::generate(data.begin(), data.end(), [&] { return distribution(rng); });
    std::generate(centroids.begin(), centroids.end(), [&] { return distribution(rng); });
    for (const bool spherical : {false, true}) {
        SCOPED_TRACE(spherical);
        expect_exact_assignment_matches_scalar(data, centroids, kDimension, spherical);
    }
}

TEST(QGKMeansTest, ExactAssignmentPreservesCallerOpenMPSettings) {
    struct SavedOpenMPSettings {
        int threads = omp_get_max_threads();
#if _OPENMP >= 200805
        int levels = omp_get_max_active_levels();
#else
        int nested = omp_get_nested();
#endif
        ~SavedOpenMPSettings() {
            omp_set_num_threads(threads);
#if _OPENMP >= 200805
            omp_set_max_active_levels(levels);
#else
            omp_set_nested(nested);
#endif
        }
    };
    const SavedOpenMPSettings saved;
    omp_set_num_threads(3);
#if _OPENMP >= 200805
    omp_set_max_active_levels(2);
#else
    omp_set_nested(1);
#endif
    constexpr size_t kDimension = 65;
    const std::vector<float> centroids(3 * kDimension, 1.0F);
    // Cover a single worker, a partial row block, and multiple workers.
    for (const size_t points : {3U, 65U, 513U}) {
        SCOPED_TRACE(points);
        const std::vector<float> data(points * kDimension, 0.0F);
        std::vector<PID> labels(points);
        std::vector<float> distances(points);
        for (const uint32_t threads : {1U, 4U}) {
            SCOPED_TRACE(threads);
            detail::exact_assign(
                data.data(),
                centroids.data(),
                points,
                kDimension,
                3,
                false,
                threads,
                labels.data(),
                distances.data()
            );
            EXPECT_EQ(omp_get_max_threads(), 3);
#if _OPENMP >= 200805
            EXPECT_EQ(omp_get_max_active_levels(), 2);
#else
            EXPECT_NE(omp_get_nested(), 0);
#endif
        }
    }
}

TEST(QGKMeansTest, ExactAssignmentPreservesLowestIndexForTiesAcrossTiles) {
    constexpr size_t kDimension = 65;
    std::vector<float> data(65 * kDimension, 0.0F);
    std::vector<float> centroids(257 * kDimension, 4.0F);
    // Centroids 3 and 256 tie, while preceding centroids are worse for every row.
    std::fill_n(centroids.data() + 3 * kDimension, kDimension, 0.0F);
    std::fill_n(centroids.data() + 256 * kDimension, kDimension, 0.0F);
    centroids[3 * kDimension] = -1.0F;
    centroids[256 * kDimension] = 1.0F;
    expect_exact_assignment_matches_scalar(data, centroids, kDimension, false);
    std::fill(data.begin(), data.end(), 1.0F);
    std::fill(centroids.begin(), centroids.end(), 0.0F);
    centroids[3 * kDimension] = 1.0F;
    centroids[256 * kDimension] = 1.0F;
    expect_exact_assignment_matches_scalar(data, centroids, kDimension, true);
}

TEST(QGKMeansTest, ExactAssignmentRefinesLargeOffsetL2Distances) {
    constexpr size_t kDimension = 129;
    std::mt19937 rng(51);
    std::uniform_real_distribution<float> distribution(-0.125F, 0.125F);
    std::vector<float> data(65 * kDimension);
    std::vector<float> centroids(257 * kDimension);
    auto shifted = [&] { return 100000.0F + distribution(rng); };
    std::generate(data.begin(), data.end(), shifted);
    std::generate(centroids.begin(), centroids.end(), shifted);
    // Include a zero-distance winner at the tail of the centroid matrix.
    std::copy_n(data.data(), kDimension, centroids.data() + 256 * kDimension);
    expect_exact_assignment_matches_scalar(data, centroids, kDimension, false);
}

TEST(QGKMeansTest, ExactAssignmentResolvesFloatNearTiesAndTinyDistances) {
    constexpr size_t kDimension = 65;
    std::vector<float> data(65 * kDimension, 0.0F);
    std::vector<float> centroids(257 * kDimension, 2.0F);
    std::fill_n(centroids.data(), kDimension, 0.0F);
    std::fill_n(centroids.data() + 256 * kDimension, kDimension, 0.0F);
    // Both squared distances round to 1 in float, but centroid 256 is nearer.
    centroids[0] = 1.0F;
    centroids[1] = 0.0001F;
    centroids[256 * kDimension] = 1.0F;
    expect_exact_assignment_matches_scalar(data, centroids, kDimension, false);
    for (float& value : centroids) {
        value *= 1e-22F;
    }
    expect_exact_assignment_matches_scalar(data, centroids, kDimension, false);
    std::fill(centroids.begin(), centroids.end(), 0.0F);
    for (const bool spherical : {false, true}) {
        SCOPED_TRACE(spherical);
        expect_exact_assignment_matches_scalar(data, centroids, kDimension, spherical);
    }
}

TEST(QGKMeansTest, ExactAssignmentRefinesInnerProductCancellation) {
    constexpr size_t kDimension = 129;
    std::vector<float> data(65 * kDimension, 100000.0F);
    std::vector<float> centroids(257 * kDimension);
    for (size_t point = 0; point < 65; ++point) {
        data[point * kDimension + kDimension - 1] = 1.0F;
    }
    for (size_t cluster = 0; cluster < 257; ++cluster) {
        for (size_t dim = 0; dim + 1 < kDimension; ++dim) {
            centroids[cluster * kDimension + dim] = dim % 2 == 0 ? 100000.0F : -100000.0F;
        }
        centroids[cluster * kDimension + kDimension - 1] =
            1.0F + static_cast<float>(cluster) / 4096.0F;
    }
    expect_exact_assignment_matches_scalar(data, centroids, kDimension, true);
}

TEST(QGKMeansTest, SphericalModeNormalizesCentroidsAndUsesInnerProduct) {
    constexpr size_t kPoints = 100;
    constexpr size_t kClusters = 33;
    constexpr size_t kDimension = 64;
    std::vector<float> data(kPoints * kDimension);
    std::mt19937 rng(456);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::generate(data.begin(), data.end(), [&] { return distribution(rng); });
    for (size_t point = 0; point < kPoints; ++point) {
        double squared_norm = 0.0;
        for (size_t dim = 0; dim < kDimension; ++dim) {
            squared_norm += static_cast<double>(data[point * kDimension + dim]) *
                            data[point * kDimension + dim];
        }
        const float scale = 1.0F / std::sqrt(static_cast<float>(squared_norm));
        for (size_t dim = 0; dim < kDimension; ++dim) {
            data[point * kDimension + dim] *= scale;
        }
    }

    QGKMeansParameters parameters = exact_search_parameters();
    parameters.niter = 2;
    parameters.spherical = true;
    parameters.final_assignment = FinalAssignmentMode::Exact;
    const QGKMeans result =
        train_qgkmeans(kDimension, kClusters, parameters, data.data(), kPoints);

    for (size_t cluster = 0; cluster < kClusters; ++cluster) {
        double squared_norm = 0.0;
        for (size_t dim = 0; dim < kDimension; ++dim) {
            squared_norm +=
                static_cast<double>(result.centroids[cluster * kDimension + dim]) *
                result.centroids[cluster * kDimension + dim];
        }
        EXPECT_NEAR(squared_norm, 1.0, 1e-5);
    }

    EXPECT_DOUBLE_EQ(
        result.final_obj,
        std::accumulate(result.distances.begin(), result.distances.end(), 0.0)
    );
    for (size_t point = 0; point < kPoints; ++point) {
        PID expected = 0;
        double expected_distance = std::numeric_limits<double>::infinity();
        for (size_t cluster = 0; cluster < kClusters; ++cluster) {
            double distance = 1.0;
            for (size_t dim = 0; dim < kDimension; ++dim) {
                distance -= static_cast<double>(data[point * kDimension + dim]) *
                            result.centroids[cluster * kDimension + dim];
            }
            if (distance < expected_distance) {
                expected = static_cast<PID>(cluster);
                expected_distance = distance;
            }
        }
        EXPECT_EQ(result.assignments[point], expected);
        EXPECT_NEAR(result.distances[point], expected_distance, 1e-6);
    }
}

TEST(QGKMeansTest, ObjectiveTerminationMatchesFaissAbsoluteRelativeChange) {
    EXPECT_TRUE(detail::should_terminate_by_objective(100.0, 101.0, 0.01));
    EXPECT_TRUE(detail::should_terminate_by_objective(100.0, 99.0, 0.01));
    EXPECT_FALSE(detail::should_terminate_by_objective(100.0, 98.0, 0.01));
    EXPECT_TRUE(detail::should_terminate_by_objective(0.0, 0.0, 0.0));
    EXPECT_FALSE(detail::should_terminate_by_objective(0.0, 1.0, 1.0));
}

TEST(QGKMeansTest, TrainsAndReturnsAssignmentsForFinalCentroids) {
    constexpr size_t kClusters = 33;
    constexpr size_t kDimension = 64;
    std::vector<float> data(kClusters * kDimension);
    for (size_t point = 0; point < kClusters; ++point) {
        for (size_t dim = 0; dim < kDimension; ++dim) {
            data[point * kDimension + dim] =
                static_cast<float>(point * 10) + static_cast<float>(dim) / 64.0F;
        }
    }

    QGKMeansParameters parameters = exact_search_parameters();
    parameters.niter = 1;
    parameters.num_threads = 2;

    const QGKMeans result =
        train_qgkmeans(kDimension, kClusters, parameters, data.data(), kClusters);

    ASSERT_EQ(result.centroids.size(), kClusters * kDimension);
    ASSERT_EQ(result.assignments.size(), kClusters);
    ASSERT_EQ(result.distances.size(), kClusters);
    ASSERT_EQ(result.iteration_stats.size(), 1U);
    for (size_t point = 0; point < kClusters; ++point) {
        EXPECT_LT(result.assignments[point], kClusters);
        EXPECT_TRUE(std::isfinite(result.distances[point]));
        EXPECT_FLOAT_EQ(result.distances[point], 0.0F);
        const size_t cluster = result.assignments[point];
        for (size_t dim = 0; dim < kDimension; ++dim) {
            EXPECT_FLOAT_EQ(
                result.centroids[cluster * kDimension + dim], data[point * kDimension + dim]
            );
        }
    }
}

TEST(QGKMeansTest, RetrainsOnOwnedCentroidsLikeAnIndependentCopy) {
    constexpr size_t kDimension = 65, kInitialClusters = 35, kClusters = 33;
    const auto data = separated_points(kInitialClusters, kDimension);
    for (const uint32_t bits : {0U, 4U, 8U}) {
        for (const auto mode :
             {FinalAssignmentMode::SymphonyQG, FinalAssignmentMode::Exact}) {
            for (const size_t offset : {0U, 1U}) {
                SCOPED_TRACE(
                    ::testing::Message()
                    << bits << "/" << static_cast<int>(mode) << "/" << offset
                );
                auto parameters = exact_search_parameters();
                parameters.niter = 2;
                parameters.quantization_bits = bits;
                parameters.final_assignment = mode;
                QGKMeans clustering(kDimension, kInitialClusters, parameters);
                clustering.train(kInitialClusters, data.data());
                const auto original = clustering.centroids;
                const float* input = original.data() + offset * kDimension;
                const size_t count = kInitialClusters - offset;

                QGKMeans reference(kDimension, kClusters, parameters);
                reference.train(count, input);
                // Shrink the output while training on its former contents, including
                // a subrange that does not start at centroids.data().
                clustering.k = kClusters;
                clustering.train(count, clustering.centroids.data() + offset * kDimension);

                EXPECT_EQ(clustering.centroids, reference.centroids);
                EXPECT_EQ(clustering.assignments, reference.assignments);
                EXPECT_EQ(clustering.distances, reference.distances);
                EXPECT_DOUBLE_EQ(clustering.final_obj, reference.final_obj);
                double objective = 0;
                for (size_t point = 0; point < count; ++point) {
                    ASSERT_LT(clustering.assignments[point], kClusters);
                    double distance = 0;
                    for (size_t dim = 0; dim < kDimension; ++dim) {
                        const double difference =
                            static_cast<double>(input[point * kDimension + dim]) -
                            clustering.centroids
                                [clustering.assignments[point] * kDimension + dim];
                        distance += difference * difference;
                    }
                    EXPECT_NEAR(
                        clustering.distances[point],
                        distance,
                        1e-5 * std::max(1.0, distance)
                    );
                    objective += distance;
                }
                EXPECT_NEAR(
                    clustering.final_obj, objective, 1e-5 * std::max(1.0, objective)
                );
            }
        }
    }
}

TEST(QGKMeansTest, EmptyClusterRecoveryDoesNotPerturbConstantData) {
    constexpr size_t kPoints = 100, kDimension = 64;
    auto parameters = exact_search_parameters();
    parameters.niter = 10;
    for (bool spherical : {false, true}) {
        SCOPED_TRACE(spherical);
        parameters.spherical = spherical;
        std::vector<float> data(kPoints * kDimension, 0.0F);
        for (size_t point = 0; point < kPoints; ++point) {
            data[point * kDimension] = 1.0F;
        }
        const auto clustering =
            train_qgkmeans(kDimension, 33, parameters, data.data(), kPoints);
        EXPECT_LT(clustering.iteration_stats.size(), parameters.niter);
        EXPECT_EQ(clustering.final_obj, 0.0);
        for (size_t cluster = 0; cluster < 33; ++cluster) {
            for (size_t dim = 0; dim < kDimension; ++dim) {
                EXPECT_EQ(clustering.centroids[cluster * kDimension + dim], data[dim]);
            }
        }
    }
}

TEST(QGKMeansTest, EmptyClusterRecoveryPreservesMeansAndFarthestSeedSelection) {
    constexpr size_t kDimension = 64, kClusters = 33, kPoints = kClusters + 8;
    const auto normalize = [=](float* row) {
        double squared_norm = 0;
        for (size_t dim = 0; dim < kDimension; ++dim) {
            squared_norm += static_cast<double>(row[dim]) * row[dim];
        }
        if (squared_norm != 0) {
            const double scale = 1.0 / std::sqrt(squared_norm);
            for (size_t dim = 0; dim < kDimension; ++dim) {
                row[dim] = static_cast<float>(row[dim] * scale);
            }
        }
    };
    std::vector<size_t> indices(kPoints);
    std::iota(indices.begin(), indices.end(), size_t{0});
    std::mt19937 initial_rng(42);
    std::shuffle(indices.begin(), indices.end(), initial_rng);

    for (bool spherical : {false, true}) {
        for (size_t empty_clusters : {1U, 3U}) {
            SCOPED_TRACE(::testing::Message() << spherical << "/" << empty_clusters);
            // Force duplicate initial seeds while all other initial centroids are
            // distinct basis vectors. Only the duplicate seeds have extra points.
            std::vector<float> data(kPoints * kDimension);
            for (size_t cluster = 0; cluster < kClusters; ++cluster) {
                const size_t axis = cluster <= empty_clusters ? 0 : cluster;
                data[indices[cluster] * kDimension + axis] = 1.0F;
            }
            for (size_t extra = 0; extra < 8; ++extra) {
                float* row = data.data() + indices[kClusters + extra] * kDimension;
                row[0] = 1.0F;
                row[kClusters] = static_cast<float>(4 - extra / 2) / 8.0F *
                                 (extra % 2 == 0 ? 1.0F : -1.0F);
                if (spherical) {
                    normalize(row);
                }
            }
            std::vector<float> initial(kClusters * kDimension);
            for (size_t cluster = 0; cluster < kClusters; ++cluster) {
                std::copy_n(
                    data.data() + indices[cluster] * kDimension,
                    kDimension,
                    initial.data() + cluster * kDimension
                );
            }
            QGAssigner reference(
                kDimension,
                kClusters,
                32,
                33,
                33,
                2,
                1,
                42,
                spherical ? METRIC_IP : METRIC_L2
            );
            std::vector<PID> labels(kPoints);
            std::vector<float> distances(kPoints);
            reference.assign(
                initial.data(), data.data(), kPoints, labels.data(), distances.data()
            );

            // Derive occupied means from the public initial assignment. Empty
            // centroids receive selected rows without changing these donor means.
            std::vector<size_t> counts(kClusters);
            std::vector<double> sums(kClusters * kDimension);
            for (size_t point = 0; point < kPoints; ++point) {
                ++counts[labels[point]];
                for (size_t dim = 0; dim < kDimension; ++dim) {
                    sums[labels[point] * kDimension + dim] +=
                        data[point * kDimension + dim];
                }
            }
            ASSERT_EQ(std::count(counts.begin(), counts.end(), size_t{0}), empty_clusters);
            std::vector<float> means(kClusters * kDimension);
            for (size_t cluster = 0; cluster < kClusters; ++cluster) {
                if (counts[cluster] != 0) {
                    for (size_t dim = 0; dim < kDimension; ++dim) {
                        means[cluster * kDimension + dim] = static_cast<float>(
                            sums[cluster * kDimension + dim] /
                            static_cast<double>(counts[cluster])
                        );
                    }
                    if (spherical) {
                        normalize(means.data() + cluster * kDimension);
                    }
                }
            }
            const auto distance = spherical ? dot_product_dis<float> : euclidean_sqr<float>;
            for (size_t point = 0; point < kPoints; ++point) {
                distances[point] = distance(
                    data.data() + point * kDimension,
                    means.data() + labels[point] * kDimension,
                    kDimension
                );
            }
            std::vector<size_t> candidates(kPoints);
            std::iota(candidates.begin(), candidates.end(), size_t{0});
            std::sort(candidates.begin(), candidates.end(), [&](size_t left, size_t right) {
                return distances[left] == distances[right]
                           ? left < right
                           : distances[left] > distances[right];
            });
            // The symmetric farthest pair also exercises deterministic ID tie-breaking.
            ASSERT_EQ(distances[candidates[0]], distances[candidates[1]]);
            size_t next_candidate = 0;
            for (size_t cluster = 0; cluster < kClusters; ++cluster) {
                if (counts[cluster] != 0) {
                    continue;
                }
                while (counts[labels[candidates[next_candidate]]] <= 1) {
                    ++next_candidate;
                    ASSERT_LT(next_candidate, candidates.size());
                }
                const size_t point = candidates[next_candidate++];
                std::copy_n(
                    data.data() + point * kDimension,
                    kDimension,
                    means.data() + cluster * kDimension
                );
                if (spherical) {
                    normalize(means.data() + cluster * kDimension);
                }
                --counts[labels[point]];
                counts[cluster] = 1;
            }
            ASSERT_EQ(std::count(counts.begin(), counts.end(), size_t{0}), 0);
            for (uint32_t threads : {1U, 4U}) {
                SCOPED_TRACE(threads);
                auto parameters = exact_search_parameters();
                parameters.niter = 1;
                parameters.spherical = spherical;
                parameters.num_threads = threads;
                parameters.final_assignment = FinalAssignmentMode::Exact;
                const auto clustering =
                    train_qgkmeans(kDimension, kClusters, parameters, data.data(), kPoints);
                ASSERT_EQ(clustering.iteration_stats.front().nsplit, empty_clusters);
                // Parallel graph construction may choose a different duplicate
                // seed label, but occupied means and selected seeds must match.
                std::vector<std::vector<float>> expected_rows, actual_rows;
                for (size_t cluster = 0; cluster < kClusters; ++cluster) {
                    const size_t offset = cluster * kDimension;
                    expected_rows.emplace_back(
                        means.data() + offset, means.data() + offset + kDimension
                    );
                    actual_rows.emplace_back(
                        clustering.centroids.data() + offset,
                        clustering.centroids.data() + offset + kDimension
                    );
                }
                std::sort(expected_rows.begin(), expected_rows.end());
                std::sort(actual_rows.begin(), actual_rows.end());
                EXPECT_EQ(actual_rows, expected_rows);
                std::fill(counts.begin(), counts.end(), size_t{0});
                for (PID label : clustering.assignments) {
                    ++counts[label];
                }
                EXPECT_EQ(std::count(counts.begin(), counts.end(), size_t{0}), 0);
            }
        }
    }
}

TEST(QGKMeansTest, RejectsNonfiniteTrainingDataWithoutReplacingResults) {
    constexpr size_t kDimension = 64;
    const auto data = separated_points(66, kDimension);
    auto parameters = exact_search_parameters();
    parameters.niter = 1;
    QGKMeans clustering(kDimension, 33, parameters);
    clustering.train(66, data.data());
    const auto expected_centroids = clustering.centroids;
    const auto expected_assignments = clustering.assignments;
    for (float invalid :
         {std::numeric_limits<float>::quiet_NaN(),
          std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity()}) {
        auto invalid_data = data;
        invalid_data.back() = invalid;
        try {
            clustering.train(66, invalid_data.data());
            FAIL() << "Accepted nonfinite input";
        } catch (const std::invalid_argument& error) {
            EXPECT_STREQ(error.what(), "QGKMeans x must contain only finite values");
        }
        EXPECT_EQ(clustering.centroids, expected_centroids);
        EXPECT_EQ(clustering.assignments, expected_assignments);
    }
}

TEST(QGKMeansTest, AccumulatesShiftedClusterMeansAccurately) {
    constexpr size_t kDimension = 64, kClusters = 33, kPoints = 10000;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> distribution(-1, 1);
    std::vector<float> data(kPoints * kDimension);
    for (auto& value : data) {
        value = 10000.0F + distribution(rng);
    }
    // Replay the public seeded initialization to obtain the first assignment;
    // compute its expected means independently in double precision.
    std::vector<size_t> indices(kPoints);
    std::iota(indices.begin(), indices.end(), size_t{0});
    std::mt19937 initial_rng(42);
    std::shuffle(indices.begin(), indices.end(), initial_rng);
    std::vector<float> initial_centroids(kClusters * kDimension);
    for (size_t cluster = 0; cluster < kClusters; ++cluster) {
        std::copy_n(
            data.data() + indices[cluster] * kDimension,
            kDimension,
            initial_centroids.data() + cluster * kDimension
        );
    }
    QGAssigner assigner(kDimension, kClusters, 32, 33, 33, 2, 1, 42);
    std::vector<PID> labels(kPoints);
    std::vector<float> distances(kPoints);
    assigner.assign(
        initial_centroids.data(), data.data(), kPoints, labels.data(), distances.data()
    );
    std::vector<double> sums(kClusters * kDimension);
    std::vector<size_t> counts(kClusters);
    for (size_t point = 0; point < kPoints; ++point) {
        ++counts[labels[point]];
        for (size_t dim = 0; dim < kDimension; ++dim) {
            sums[labels[point] * kDimension + dim] += data[point * kDimension + dim];
        }
    }
    auto parameters = exact_search_parameters();
    parameters.niter = 1;
    const auto clustering =
        train_qgkmeans(kDimension, kClusters, parameters, data.data(), kPoints);
    EXPECT_EQ(
        clustering.iteration_stats.front().nsplit,
        std::count(counts.begin(), counts.end(), size_t{0})
    );
    for (size_t cluster = 0; cluster < kClusters; ++cluster) {
        if (counts[cluster] == 0) {
            continue;
        }
        for (size_t dim = 0; dim < kDimension; ++dim) {
            EXPECT_FLOAT_EQ(
                clustering.centroids[cluster * kDimension + dim],
                static_cast<float>(sums[cluster * kDimension + dim] / counts[cluster])
            );
        }
    }
}

TEST(QGKMeansTest, FinalAssignmentsRetainACloserPreviousCluster) {
    constexpr size_t kDimension = 65, kClusters = 256, kPoints = 2048;
    const auto normalize = [=](float* row) {
        double norm = 0;
        for (size_t dim = 0; dim < kDimension; ++dim) {
            norm += static_cast<double>(row[dim]) * row[dim];
        }
        const double scale = 1.0 / std::sqrt(norm);
        for (size_t dim = 0; dim < kDimension; ++dim) {
            row[dim] = static_cast<float>(row[dim] * scale);
        }
    };
    for (bool spherical : {false, true}) {
        std::mt19937 rng(123);
        std::normal_distribution<float> normal;
        std::vector<float> data(kPoints * kDimension);
        std::generate(data.begin(), data.end(), [&] { return normal(rng); });
        if (spherical) {
            for (size_t point = 0; point < kPoints; ++point) {
                normalize(data.data() + point * kDimension);
            }
        }
        std::vector<size_t> indices(kPoints);
        std::iota(indices.begin(), indices.end(), size_t{0});
        std::mt19937 init_rng(42);
        std::shuffle(indices.begin(), indices.end(), init_rng);
        std::vector<float> initial(kClusters * kDimension);
        for (size_t cluster = 0; cluster < kClusters; ++cluster) {
            float* row = initial.data() + cluster * kDimension;
            std::copy_n(data.data() + indices[cluster] * kDimension, kDimension, row);
            if (spherical) {
                normalize(row);
            }
        }
        const auto metric = spherical ? METRIC_IP : METRIC_L2;
        for (uint32_t bits : {0U, 4U, 8U}) {
            SCOPED_TRACE(::testing::Message() << spherical << "/" << bits);
            QGAssigner reference(
                kDimension,
                kClusters,
                32,
                240,
                16,
                QGKMeansParameters{}.graph_build_iterations,
                1,
                42,
                metric,
                bits
            );
            std::vector<PID> previous(kPoints);
            std::vector<float> distances(kPoints);
            reference.assign(
                initial.data(), data.data(), kPoints, previous.data(), distances.data()
            );
            QGKMeansParameters parameters;
            parameters.niter = 1;
            parameters.num_threads = 1;
            parameters.min_points_per_centroid = 1;
            parameters.spherical = spherical;
            parameters.quantization_bits = bits;
            QGKMeans clustering(kDimension, kClusters, parameters);
            clustering.train(kPoints, data.data());
            const auto distance = spherical ? dot_product_dis<float> : euclidean_sqr<float>;
            for (size_t point = 0; point < kPoints; ++point) {
                const float previous_distance = distance(
                    data.data() + point * kDimension,
                    clustering.centroids.data() + previous[point] * kDimension,
                    kDimension
                );
                EXPECT_LE(clustering.distances[point], previous_distance);
            }
        }
    }
}

TEST(QGKMeansTest, StopsWhenZeroDataCannotImprove) {
    auto parameters = exact_search_parameters();
    parameters.niter = 10;
    const std::vector<float> data(100 * 64, 0.0F);
    const auto clustering = train_qgkmeans(64, 33, parameters, data.data(), 100);
    EXPECT_LT(clustering.iteration_stats.size(), parameters.niter);
    EXPECT_EQ(
        std::accumulate(clustering.distances.begin(), clustering.distances.end(), 0.0), 0.0
    );
}

TEST(QGKMeansTest, TrainsWithMovingCentroidsAndAnApproximateGraph) {
    constexpr size_t kDim = 65, kCount = 2048, kClusters = 256;
    std::mt19937 rng(739);
    std::normal_distribution<float> normal;
    std::vector<float> data(kCount * kDim);
    std::generate(data.begin(), data.end(), [&] { return normal(rng); });
    for (bool spherical : {false, true}) {
        if (spherical) {
            for (size_t point = 0; point < kCount; ++point) {
                double norm = 0;
                for (size_t dim = 0; dim < kDim; ++dim) {
                    const double value = data[point * kDim + dim];
                    norm += value * value;
                }
                for (size_t dim = 0; dim < kDim; ++dim) {
                    data[point * kDim + dim] /= static_cast<float>(std::sqrt(norm));
                }
            }
        }
        for (uint32_t bits : {0U, 4U, 8U}) {
            SCOPED_TRACE(::testing::Message() << spherical << "/" << bits);
            QGKMeansParameters parameters;
            parameters.niter = 5;
            parameters.num_threads = 2;
            parameters.min_points_per_centroid = 1;
            parameters.spherical = spherical;
            parameters.quantization_bits = bits;
            QGKMeans clustering(kDim, kClusters, parameters);
            clustering.train(kCount, data.data());
            double previous = std::numeric_limits<double>::infinity();
            for (const auto& stats : clustering.iteration_stats) {
                EXPECT_TRUE(std::isfinite(stats.obj));
                if (bits == 0) {
                    EXPECT_LE(stats.obj, previous + 1e-6 * std::abs(previous));
                }
                previous = stats.obj;
            }
            EXPECT_LE(clustering.final_obj, previous + 1e-6 * std::abs(previous));
        }
    }
}

TEST(QGAssignerTest, RejectsTooFewClustersForGraphDegree) {
    EXPECT_THROW(QGAssigner(64, 32, 32, 64, 16, 1, 1), std::invalid_argument);
}

TEST(QGKMeansTest, RejectsTooFewClustersForGraphDegreeInBothFinalModes) {
    constexpr size_t kDim = 64, kPoints = 65;
    const std::vector<float> data(kPoints * kDim, 0.0F);
    for (const auto mode : {FinalAssignmentMode::Approximate, FinalAssignmentMode::Exact}) {
        for (const uint32_t degree : {32U, 64U}) {
            for (const size_t clusters : {1U, 2U, 16U, 31U, 32U, 33U, 64U}) {
                if (clusters > degree) {
                    continue;
                }
                QGKMeansParameters parameters;
                parameters.niter = 1;
                parameters.num_threads = 1;
                parameters.graph_degree = degree;
                parameters.final_assignment = mode;
                try {
                    QGKMeans model(kDim, clusters, parameters);
                    model.train(kPoints, data.data());
                    FAIL() << "Accepted k=" << clusters << " for degree=" << degree;
                } catch (const std::invalid_argument& error) {
                    EXPECT_STREQ(
                        error.what(),
                        "QGKMeans requires more centroids than the graph degree"
                    );
                }
            }
        }
    }
}

TEST(QGKMeansTest, RejectsInvalidNativeParametersBeforeAllocation) {
    QGKMeansParameters parameters;
    parameters.num_threads = std::numeric_limits<uint32_t>::max();
    EXPECT_THROW((QGKMeans(64, 33, parameters)), std::invalid_argument);
    EXPECT_THROW(
        (QGAssigner(64, 33, 32, 64, 16, 2, parameters.num_threads)), std::invalid_argument
    );
    parameters.num_threads = 1;
    parameters.final_assignment = static_cast<FinalAssignmentMode>(255);
    EXPECT_THROW((QGKMeans(64, 33, parameters)), std::invalid_argument);
    parameters.final_assignment = FinalAssignmentMode::Exact;
    QGKMeans clustering(64, 33, parameters);
    const float value = 0;
    EXPECT_THROW(
        clustering.train(std::numeric_limits<size_t>::max() / 64 + 1, &value),
        std::invalid_argument
    );
}

}  // namespace
}  // namespace rabitqlib::qgkmeans

namespace rabitqlib::qgkmeans {
TEST(QGAssignerTest, PropagatesSearchFailureFromWorkers) {
    const std::vector<float> centroids(33 * 64, 0.0F);
    const std::vector<float> data(128 * 64, 1e19F);
    std::vector<PID> labels(128, kPidMax);
    std::vector<float> distances(128, -1.0F);
    QGAssigner assigner(64, 33, 32, 33, 33, 1, 2);
    EXPECT_THROW(
        assigner.assign(
            centroids.data(), data.data(), 128, labels.data(), distances.data()
        ),
        std::runtime_error
    );
    EXPECT_EQ(labels.front(), kPidMax);
    EXPECT_EQ(distances.front(), -1.0F);
}

TEST(QGKMeansTest, RejectsFiniteOutOfRangeInputBeforeConstruction) {
    for (uint32_t bits : {0U, 4U, 8U}) {
        QGKMeansParameters parameters;
        parameters.niter = 1;
        parameters.num_threads = 2;
        parameters.quantization_bits = bits;
        QGKMeans model(64, 33, parameters);
        const std::vector<float> data(66 * 64, 1e19F);
        EXPECT_THROW(model.train(66, data.data()), std::invalid_argument);
        EXPECT_TRUE(model.centroids.empty());
    }
}
}  // namespace rabitqlib::qgkmeans

namespace rabitqlib::qgkmeans {
TEST(QGKMeansTest, SupportsLargerPaddedDimensions) {
    constexpr size_t k = 33;
    for (const size_t d : {4097U, 65535U, 65536U}) {
        SCOPED_TRACE(d);
        std::vector<float> data(k * d, 0.0F);
        for (size_t i = 0; i < k; ++i) {
            data[i * d + i] = 1.0F;
            data[i * d + d - 1] = static_cast<float>(i) / 64.0F;
        }
        for (uint32_t bits : {0U, 4U, 8U}) {
            QGKMeansParameters parameters;
            parameters.niter = 1;
            parameters.num_threads = 2;
            parameters.quantization_bits = bits;
            QGKMeans model(d, k, parameters);
            model.train(k, data.data());
            EXPECT_DOUBLE_EQ(model.final_obj, 0.0);
            EXPECT_EQ(model.assignments.size(), k);
        }
    }
    EXPECT_THROW(QGAssigner(65537, 33, 32, 33, 33, 1, 1), std::invalid_argument);
}

namespace {
void normalize_reference_centroids(std::vector<float>& rows, size_t dim) {
    for (size_t offset = 0; offset < rows.size(); offset += dim) {
        double norm = 0;
        for (size_t j = 0; j < dim; ++j) {
            norm += static_cast<double>(rows[offset + j]) * rows[offset + j];
        }
        if (norm == 0)
            continue;
        const double scale = 1.0 / std::sqrt(norm);
        for (size_t j = 0; j < dim; ++j) {
            rows[offset + j] = static_cast<float>(rows[offset + j] * scale);
        }
    }
}

void check_uncached_reference(size_t clusters, uint32_t ef_search, size_t intrinsic_dim) {
    constexpr size_t kDimension = 1024;
    constexpr size_t kPoints = 257;
    constexpr size_t kIterations = 3;
    std::mt19937 data_rng(823);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> x(kPoints * kDimension);
    std::generate(x.begin(), x.end(), [&] { return distribution(data_rng); });
    // Low intrinsic dimension makes points change clusters during Lloyd updates.
    // Padding keeps the scalar-sum cache enabled without concentrating distances.
    for (size_t point = 0; point < kPoints; ++point) {
        std::fill_n(
            x.data() + point * kDimension + intrinsic_dim, kDimension - intrinsic_dim, 0.0F
        );
    }
    for (const uint32_t bits : {0U, 4U, 8U}) {
        for (const bool spherical : {false, true}) {
            for (const uint32_t threads : {1U, 4U}) {
                SCOPED_TRACE(
                    ::testing::Message() << clusters << "/" << ef_search << "/" << bits
                                         << "/" << spherical << "/" << threads
                );
                QGKMeansParameters parameters;
                parameters.niter = kIterations;
                parameters.num_threads = threads;
                parameters.seed = 42;
                parameters.min_points_per_centroid = 1;
                parameters.quantization_bits = bits;
                parameters.spherical = spherical;
                parameters.ef_build = 128;
                parameters.ef_search = ef_search;
                QGKMeans model(kDimension, clusters, parameters);
                model.train(kPoints, x.data());
                ASSERT_EQ(model.iteration_stats.size(), kIterations);

                std::vector<size_t> initial_ids(kPoints);
                std::iota(initial_ids.begin(), initial_ids.end(), size_t{0});
                std::mt19937 initialization_rng(parameters.seed);
                std::shuffle(initial_ids.begin(), initial_ids.end(), initialization_rng);
                std::vector<float> centroids(clusters * kDimension);
                for (size_t cluster = 0; cluster < clusters; ++cluster) {
                    std::copy_n(
                        x.data() + initial_ids[cluster] * kDimension,
                        kDimension,
                        centroids.data() + cluster * kDimension
                    );
                }
                if (spherical)
                    normalize_reference_centroids(centroids, kDimension);
                QGAssigner uncached(
                    kDimension,
                    clusters,
                    32,
                    parameters.ef_build,
                    ef_search,
                    1,
                    threads,
                    parameters.seed,
                    spherical ? METRIC_IP : METRIC_L2,
                    bits
                );
                std::vector<PID> labels(kPoints);
                std::vector<float> distances(kPoints);
                for (size_t iteration = 0; iteration < kIterations; ++iteration) {
                    SCOPED_TRACE(iteration);
                    // This public call intentionally never receives a prepared sum.
                    uncached.assign(
                        centroids.data(), x.data(), kPoints, labels.data(), distances.data()
                    );
                    const double objective =
                        std::accumulate(distances.begin(), distances.end(), 0.0);
                    EXPECT_EQ(model.iteration_stats[iteration].obj, objective);
                    if (iteration == 0) {
                        EXPECT_FALSE(std::is_sorted(labels.begin(), labels.end()));
                    }
                    std::vector<double> sums(clusters * kDimension, 0.0);
                    std::vector<size_t> counts(clusters, 0);
                    for (size_t point = 0; point < kPoints; ++point) {
                        ++counts[labels[point]];
                        for (size_t j = 0; j < kDimension; ++j) {
                            sums[labels[point] * kDimension + j] +=
                                static_cast<double>(x[point * kDimension + j]);
                        }
                    }
                    for (size_t cluster = 0; cluster < clusters; ++cluster) {
                        // Keep this an independent Lloyd reference, without empty repair.
                        ASSERT_GT(counts[cluster], 0U);
                        const double scale = 1.0 / static_cast<double>(counts[cluster]);
                        for (size_t j = 0; j < kDimension; ++j) {
                            centroids[cluster * kDimension + j] =
                                static_cast<float>(sums[cluster * kDimension + j] * scale);
                        }
                    }
                    if (spherical)
                        normalize_reference_centroids(centroids, kDimension);
                }
                // Compare state before the final assignment, whose safeguard is separate.
                EXPECT_EQ(model.centroids, centroids);
            }
        }
    }
}
TEST(QGKMeansQuerySumTest, FullGraphTrainingMatchesUncachedLloydReference) {
    check_uncached_reference(33, 33, 1024);
}
TEST(QGKMeansQuerySumTest, PartialGraphTrainingMatchesUncachedLloydReference) {
    check_uncached_reference(65, 1, 8);
}
}  // namespace

}  // namespace rabitqlib::qgkmeans
