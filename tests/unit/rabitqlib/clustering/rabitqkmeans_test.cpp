#include "rabitqlib/clustering/rabitqkmeans.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::rabitqkmeans {
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

std::vector<PID> scalar_rabitq_labels(
    const std::vector<float>& centroids,
    const std::vector<float>& data,
    size_t dim,
    uint32_t seed,
    MetricType metric
) {
    const size_t clusters = centroids.size() / dim;
    const size_t points = data.size() / dim;
    const size_t padded =
        rotator_impl::padding_requirement(dim, RotatorType::FhtKacRotator);
    rotator_impl::FhtKacRotator rotator(dim, padded, seed);
    std::vector<float> center(dim), rotated_center(padded), rotated(points * padded);
    for (size_t j = 0; j < dim; ++j) {
        double sum = 0;
        for (size_t point = 0; point < points; ++point) {
            sum += data[point * dim + j];
        }
        center[j] = static_cast<float>(sum / static_cast<double>(points));
    }
    rotator.rotate(center.data(), rotated_center.data());
    for (size_t point = 0; point < points; ++point) {
        rotator.rotate(data.data() + point * dim, rotated.data() + point * padded);
    }
    const size_t batches = (points + 31) / 32;
    const size_t bytes = QGBatchDataMap<float>::data_bytes(padded);
    std::vector<char> storage(batches * bytes);
    for (size_t batch = 0; batch < batches; ++batch) {
        const size_t first = batch * 32;
        quant::quantize_qg_batch(
            rotated.data() + first * padded,
            rotated_center.data(),
            std::min(size_t{32}, points - first),
            padded,
            storage.data() + batch * bytes,
            metric
        );
    }
    const auto distance =
        metric == METRIC_IP ? dot_product_dis<float> : euclidean_sqr<float>;
    std::vector<PID> labels(points);
    std::vector<float> best(points, std::numeric_limits<float>::infinity()), query(padded);
    for (size_t cluster = 0; cluster < clusters; ++cluster) {
        const float* centroid = centroids.data() + cluster * dim;
        rotator.rotate(centroid, query.data());
        BatchQuery<float> prepared(query.data(), padded, metric);
        // This term depends on the moving centroid and cannot be omitted from argmin.
        prepared.set_g_add(distance(centroid, center.data(), dim));
        for (size_t point = 0; point < points; ++point) {
            ConstQGBatchDataMap<float> batch(storage.data() + (point / 32) * bytes, padded);
            const size_t index = point % 32;
            const auto lane = std::find(
                fastscan::kPerm0.begin(),
                fastscan::kPerm0.end(),
                static_cast<int>(index % 16)
            );
            const size_t packed_lane = static_cast<size_t>(lane - fastscan::kPerm0.begin());
            int32_t accumulator = 0;
            // Independent scalar accumulation over the encoded point's nibble sequence.
            for (size_t group = 0; group < padded / 4; ++group) {
                const uint8_t packed = batch.bin_code()[group * 16 + packed_lane];
                const size_t code = (packed >> (index / 16 * 4)) & 15U;
                accumulator += prepared.lut()[group * 16 + code];
            }
            const float ip = prepared.delta() * static_cast<float>(accumulator) +
                             prepared.sum_vl_lut() + prepared.k1xsumq();
            const float estimate =
                batch.f_add()[index] + prepared.g_add() + batch.f_rescale()[index] * ip;
            if (estimate < best[point]) {
                best[point] = estimate;
                labels[point] = static_cast<PID>(cluster);
            }
        }
    }
    return labels;
}

double scalar_centroid_distance(const float* x, const float* c, size_t dim, bool ip) {
    double distance = 0;
    for (size_t j = 0; j < dim; ++j) {
        const double left = x[j], right = c[j];
        const double difference = left - right;
        distance += ip ? -left * right : difference * difference;
    }
    return ip ? 1.0 + distance : distance;
}

TEST(RaBitQKMeansTest, ValidatesCommonParametersBeforeAllocation) {
    EXPECT_EQ(RaBitQKMeansParameters{}.niter, 25U);
    EXPECT_EQ(RaBitQKMeansParameters{}.final_assignment, FinalAssignmentMode::Approximate);
    for (const size_t dimension : {0U, 63U, 65537U}) {
        EXPECT_THROW((RaBitQKMeans(dimension, 1)), std::invalid_argument);
    }
    EXPECT_THROW((RaBitQKMeans(64, 0)), std::invalid_argument);
    if constexpr (sizeof(size_t) > sizeof(PID)) {
        const size_t max_clusters =
            static_cast<size_t>(std::numeric_limits<PID>::max()) + 1;
        EXPECT_NO_THROW((RaBitQKMeans(64, max_clusters)));
        EXPECT_THROW((RaBitQKMeans(64, max_clusters + 1)), std::invalid_argument);
    }
    RaBitQKMeansParameters parameters;
    parameters.niter = 0;
    EXPECT_THROW((RaBitQKMeans(64, 1, parameters)), std::invalid_argument);
    parameters.niter = 1;
    parameters.final_assignment = static_cast<FinalAssignmentMode>(255);
    EXPECT_THROW((RaBitQKMeans(64, 1, parameters)), std::invalid_argument);
    parameters.final_assignment = FinalAssignmentMode::Approximate;
    parameters.num_threads = std::numeric_limits<uint32_t>::max();
    EXPECT_THROW((RaBitQKMeans(64, 1, parameters)), std::invalid_argument);
    parameters.num_threads = 1;
    RaBitQKMeans model(64, 1, parameters);
    const float value = 0.0F;
    EXPECT_THROW(model.train(1, nullptr), std::invalid_argument);
    EXPECT_THROW(model.train(0, &value), std::invalid_argument);
    EXPECT_THROW(
        model.train(std::numeric_limits<size_t>::max() / 64 + 1, &value),
        std::invalid_argument
    );
}

TEST(RaBitQAssignerTest, MatchesScalarPointCodesAcrossBatchTailsAndMovingCentroids) {
    constexpr uint32_t kSeed = 53;
    std::mt19937 rng(157);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    size_t approximate_choices_above_batch_size = 0;
    for (const size_t dim : {64U, 65U, 1025U}) {
        for (const size_t points : {1U, 31U, 32U, 33U, 65U, 129U}) {
            std::vector<float> data(points * dim);
            std::generate(data.begin(), data.end(), [&] { return distribution(rng); });
            const auto original = data;
            for (const size_t clusters : {1U, 2U, 16U, 31U, 32U, 33U, 65U}) {
                if (clusters > points) {
                    continue;
                }
                std::vector<float> initial(clusters * dim);
                std::generate(initial.begin(), initial.end(), [&] {
                    return distribution(rng);
                });
                for (const MetricType metric : {METRIC_L2, METRIC_IP}) {
                    for (const uint32_t threads : {1U, 4U}) {
                        SCOPED_TRACE(
                            ::testing::Message()
                            << dim << '/' << points << '/' << clusters << '/'
                            << static_cast<int>(metric) << '/' << threads
                        );
                        auto centers = initial;
                        RaBitQKMeansParameters parameters;
                        parameters.seed = kSeed;
                        parameters.spherical = metric == METRIC_IP;
                        detail::RaBitQAssigner assigner(
                            dim, clusters, data.data(), points, threads, parameters
                        );
                        for (size_t pass = 0; pass < 2; ++pass) {
                            for (float& value : centers) {
                                value = value * 0.75F + 0.125F;
                            }
                            const auto expected =
                                scalar_rabitq_labels(centers, data, dim, kSeed, metric);
                            std::vector<PID> labels(points, kPidMax);
                            std::vector<float> distances(points);
                            assigner.assign(
                                centers.data(), labels.data(), distances.data()
                            );
                            EXPECT_EQ(labels, expected);
                            for (size_t point = 0; point < points; ++point) {
                                ASSERT_LT(labels[point], clusters);
                                const float* vector = data.data() + point * dim;
                                const auto exact_distance = metric == METRIC_IP
                                                                ? dot_product_dis<float>
                                                                : euclidean_sqr<float>;
                                EXPECT_EQ(
                                    distances[point],
                                    exact_distance(
                                        vector, centers.data() + labels[point] * dim, dim
                                    )
                                );
                                PID nearest = 0;
                                double best = std::numeric_limits<double>::infinity();
                                for (size_t cluster = 0; cluster < clusters; ++cluster) {
                                    const double exact = scalar_centroid_distance(
                                        vector,
                                        centers.data() + cluster * dim,
                                        dim,
                                        metric == METRIC_IP
                                    );
                                    if (exact < best) {
                                        best = exact;
                                        nearest = static_cast<PID>(cluster);
                                    }
                                }
                                if (clusters > 32 && labels[point] != nearest) {
                                    ++approximate_choices_above_batch_size;
                                }
                            }
                        }
                    }
                }
            }
            EXPECT_EQ(data, original);
        }
    }
    EXPECT_GT(approximate_choices_above_batch_size, 0U);
}

TEST(RaBitQAssignerTest, ZeroAndDuplicateResidualsChooseFirstLogicalCentroid) {
    constexpr size_t kDim = 65, kPoints = 65;
    for (const size_t clusters : {1U, 2U, 16U, 31U, 32U, 33U, 65U}) {
        for (const MetricType metric : {METRIC_L2, METRIC_IP}) {
            for (const float value : {0.0F, 0.25F}) {
                std::vector<float> centers(clusters * kDim, value),
                    data(kPoints * kDim, value);
                RaBitQKMeansParameters parameters;
                parameters.spherical = metric == METRIC_IP;
                detail::RaBitQAssigner assigner(
                    kDim, clusters, data.data(), kPoints, 2, parameters
                );
                std::vector<PID> labels(kPoints, kPidMax);
                std::vector<float> distances(kPoints);
                assigner.assign(centers.data(), labels.data(), distances.data());
                EXPECT_EQ(labels, (std::vector<PID>(kPoints, 0)));
                const double expected = scalar_centroid_distance(
                    data.data(), centers.data(), kDim, metric == METRIC_IP
                );
                for (const float distance : distances) {
                    EXPECT_NEAR(distance, expected, 1e-6);
                }
            }
        }
    }
}

TEST(RaBitQAssignerTest, KeepsCentroidDependentCorrectionForZeroResidualPointCodes) {
    constexpr size_t kDim = 65, kPoints = 33;
    for (const MetricType metric : {METRIC_L2, METRIC_IP}) {
        std::vector<float> data(kPoints * kDim), centers(2 * kDim);
        if (metric == METRIC_IP) {
            for (size_t point = 0; point < kPoints; ++point) {
                data[point * kDim] = 1.0F;
            }
            centers[1] = 1.0F;
            centers[kDim] = 1.0F;
        } else {
            centers[0] = 1.0F;
        }
        RaBitQKMeansParameters parameters;
        parameters.spherical = metric == METRIC_IP;
        detail::RaBitQAssigner assigner(kDim, 2, data.data(), kPoints, 2, parameters);
        std::vector<PID> labels(kPoints, kPidMax);
        std::vector<float> distances(kPoints);
        assigner.assign(centers.data(), labels.data(), distances.data());
        EXPECT_EQ(labels, (std::vector<PID>(kPoints, 1)));
        EXPECT_EQ(distances, (std::vector<float>(kPoints, 0.0F)));
    }
}

TEST(RaBitQKMeansTest, FinalModesKeepFlatLloydAssignmentsAcrossClusterCounts) {
    constexpr size_t kDim = 65, kPoints = 67;
    std::mt19937 rng(675);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> data(kPoints * kDim);
    std::generate(data.begin(), data.end(), [&] { return distribution(rng); });
    for (const size_t clusters : {1U, 2U, 16U, 31U, 32U, 33U, 65U}) {
        for (const bool spherical : {false, true}) {
            std::vector<size_t> ids(kPoints);
            std::iota(ids.begin(), ids.end(), size_t{0});
            std::mt19937 initialization(42);
            std::shuffle(ids.begin(), ids.end(), initialization);
            std::vector<float> initial(clusters * kDim);
            for (size_t cluster = 0; cluster < clusters; ++cluster) {
                std::copy_n(
                    data.data() + ids[cluster] * kDim, kDim, initial.data() + cluster * kDim
                );
            }
            if (spherical) {
                normalize_reference_centroids(initial, kDim);
            }
            const auto metric = spherical ? METRIC_IP : METRIC_L2;
            const auto expected = scalar_rabitq_labels(initial, data, kDim, 42, metric);
            const auto distance = spherical ? dot_product_dis<float> : euclidean_sqr<float>;
            double expected_objective = 0;
            for (size_t point = 0; point < kPoints; ++point) {
                expected_objective += distance(
                    data.data() + point * kDim,
                    initial.data() + expected[point] * kDim,
                    kDim
                );
            }
            std::vector<size_t> counts(clusters);
            std::vector<double> sums(clusters * kDim);
            for (size_t point = 0; point < kPoints; ++point) {
                ++counts[expected[point]];
                for (size_t j = 0; j < kDim; ++j) {
                    sums[expected[point] * kDim + j] += data[point * kDim + j];
                }
            }
            std::vector<float> means(clusters * kDim);
            for (size_t cluster = 0; cluster < clusters; ++cluster) {
                if (counts[cluster] != 0) {
                    const double scale = 1.0 / static_cast<double>(counts[cluster]);
                    for (size_t j = 0; j < kDim; ++j) {
                        means[cluster * kDim + j] =
                            static_cast<float>(sums[cluster * kDim + j] * scale);
                    }
                }
            }
            if (spherical) {
                normalize_reference_centroids(means, kDim);
            }
            for (const auto mode :
                 {FinalAssignmentMode::Approximate, FinalAssignmentMode::Exact}) {
                RaBitQKMeansParameters parameters;
                parameters.niter = 1;
                parameters.num_threads = 2;
                parameters.min_points_per_centroid = 1;
                parameters.spherical = spherical;
                parameters.final_assignment = mode;
                RaBitQKMeans model(kDim, clusters, parameters);
                model.train(kPoints, data.data());
                ASSERT_EQ(model.iteration_stats.size(), 1U);
                EXPECT_DOUBLE_EQ(model.iteration_stats.front().obj, expected_objective);
                // Codes only drive assignment; populated means use original float data.
                for (size_t cluster = 0; cluster < clusters; ++cluster) {
                    if (counts[cluster] != 0) {
                        for (size_t j = 0; j < kDim; ++j) {
                            EXPECT_EQ(
                                model.centroids[cluster * kDim + j],
                                means[cluster * kDim + j]
                            );
                        }
                    }
                }
                const auto approximate =
                    scalar_rabitq_labels(model.centroids, data, kDim, 42, metric);
                for (size_t point = 0; point < kPoints; ++point) {
                    PID nearest = 0;
                    double best = std::numeric_limits<double>::infinity();
                    for (size_t cluster = 0; cluster < clusters; ++cluster) {
                        const double exact = scalar_centroid_distance(
                            data.data() + point * kDim,
                            model.centroids.data() + cluster * kDim,
                            kDim,
                            spherical
                        );
                        if (exact < best) {
                            best = exact;
                            nearest = static_cast<PID>(cluster);
                        }
                    }
                    EXPECT_EQ(
                        model.assignments[point],
                        mode == FinalAssignmentMode::Exact ? nearest : approximate[point]
                    );
                    const double selected = scalar_centroid_distance(
                        data.data() + point * kDim,
                        model.centroids.data() + model.assignments[point] * kDim,
                        kDim,
                        spherical
                    );
                    EXPECT_NEAR(
                        model.distances[point],
                        selected,
                        2e-6 * std::max(1.0, std::abs(selected))
                    );
                }
                EXPECT_DOUBLE_EQ(
                    model.final_obj,
                    std::accumulate(model.distances.begin(), model.distances.end(), 0.0)
                );
            }
        }
    }
}

TEST(RaBitQKMeansTest, TrainsAtMaximumDimension) {
    // A partial FastScan batch keeps the maximum-width training check small.
    constexpr size_t kDim = 65536, kPoints = 5, kClusters = 2;
    std::vector<float> data(kPoints * kDim);
    for (size_t point = 0; point < kPoints; ++point) {
        // Exercise the full extent, including the last coordinate, without
        // making the distance check depend on long float accumulation chains.
        for (size_t dim = 1023; dim < kDim; dim += 1024) {
            data[point * kDim + dim] =
                static_cast<float>(static_cast<int>((dim + point * 7) % 19) - 9) / 256.0F;
        }
    }
    const auto original = data;
    for (const auto mode : {FinalAssignmentMode::Approximate, FinalAssignmentMode::Exact}) {
        SCOPED_TRACE(static_cast<int>(mode));
        RaBitQKMeansParameters parameters;
        parameters.niter = 2;
        parameters.num_threads = 2;
        parameters.final_assignment = mode;
        RaBitQKMeans model(kDim, kClusters, parameters);
        model.train(kPoints, data.data());
        ASSERT_EQ(model.iteration_stats.size(), 2U);
        ASSERT_EQ(model.centroids.size(), kClusters * kDim);
        ASSERT_EQ(model.assignments.size(), kPoints);
        ASSERT_EQ(model.distances.size(), kPoints);
        EXPECT_EQ(data, original);
        EXPECT_TRUE(std::all_of(
            model.centroids.begin(),
            model.centroids.end(),
            [](float value) { return std::isfinite(value); }
        ));
        double objective = 0;
        for (size_t point = 0; point < kPoints; ++point) {
            ASSERT_LT(model.assignments[point], kClusters);
            const double selected = scalar_centroid_distance(
                data.data() + point * kDim,
                model.centroids.data() + model.assignments[point] * kDim,
                kDim,
                false
            );
            EXPECT_NEAR(model.distances[point], selected, 2e-6 * std::max(1.0, selected));
            objective += selected;
            if (mode == FinalAssignmentMode::Exact) {
                for (size_t cluster = 0; cluster < kClusters; ++cluster) {
                    EXPECT_LE(
                        selected,
                        scalar_centroid_distance(
                            data.data() + point * kDim,
                            model.centroids.data() + cluster * kDim,
                            kDim,
                            false
                        )
                    );
                }
            }
        }
        EXPECT_NEAR(model.final_obj, objective, 2e-6 * std::max(1.0, objective));
    }
}

TEST(RaBitQKMeansTest, PointCodesPreserveBorrowedCentroidsDuringRetraining) {
    constexpr size_t kDim = 65, kPoints = 129;
    std::mt19937 rng(945);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> data(kPoints * kDim);
    std::generate(data.begin(), data.end(), [&] { return distribution(rng); });
    for (const size_t kInitialClusters : {32U, 65U}) {
        const size_t kClusters = kInitialClusters == 32 ? 16 : 33;
        for (const bool spherical : {false, true}) {
            for (const auto mode :
                 {FinalAssignmentMode::Approximate, FinalAssignmentMode::Exact}) {
                for (const size_t offset : {0U, 1U}) {
                    SCOPED_TRACE(
                        ::testing::Message()
                        << spherical << '/' << static_cast<int>(mode) << '/' << offset
                    );
                    RaBitQKMeansParameters parameters;
                    parameters.niter = 2;
                    parameters.num_threads = 2;
                    parameters.min_points_per_centroid = 1;
                    parameters.spherical = spherical;
                    parameters.final_assignment = mode;
                    RaBitQKMeans model(kDim, kInitialClusters, parameters);
                    model.train(kPoints, data.data());
                    const auto original = model.centroids;
                    const size_t count = kInitialClusters - offset;
                    parameters.seed = 77;
                    RaBitQKMeans reference(kDim, kClusters, parameters);
                    reference.train(count, original.data() + offset * kDim);
                    model.k = kClusters;
                    model.seed = parameters.seed;
                    // The point-code cache must bind to the preserved copy, because
                    // both updates and returned distances still read original input.
                    model.train(count, model.centroids.data() + offset * kDim);
                    EXPECT_EQ(model.centroids, reference.centroids);
                    EXPECT_EQ(model.assignments, reference.assignments);
                    EXPECT_EQ(model.distances, reference.distances);
                    EXPECT_DOUBLE_EQ(model.final_obj, reference.final_obj);
                    ASSERT_EQ(
                        model.iteration_stats.size(), reference.iteration_stats.size()
                    );
                    for (size_t i = 0; i < model.iteration_stats.size(); ++i) {
                        EXPECT_DOUBLE_EQ(
                            model.iteration_stats[i].obj, reference.iteration_stats[i].obj
                        );
                    }
                }
            }
        }
    }
}

}  // namespace
}  // namespace rabitqlib::rabitqkmeans
