#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/index/symqg/qg_builder.hpp"

namespace rabitqlib::symqg {
struct QGConstructionTestAccess {
    static auto codes(const QuantizedGraph<float>& graph) {
        std::vector<char> result;
        for (PID id = 0; id < graph.num_points_; ++id) {
            const char* code = graph.get_quantized_vector(id);
            result.insert(result.end(), code, code + graph.batch_data_offset_);
        }
        return result;
    }
    static const auto& neighbors(const QGBuilder& builder) {
        return builder.new_neighbors_;
    }
    static std::array<double, 2> estimate(
        const QuantizedGraph<float>& graph, PID source, PID target
    ) {
        std::vector<float> query(graph.padded_dim_);
        graph.reconstruct_quantized_vector(source, query.data());
        ConstExDataMap<float> code(
            graph.get_quantized_vector(target), graph.padded_dim_, graph.quantization_bits_
        );
        const double midpoint = ((1U << graph.quantization_bits_) - 1) / 2.0;
        double dot = 0, add = code.f_add_ex(), magnitude = std::abs(add);
        for (size_t d = 0; d < query.size(); ++d) {
            const auto byte = graph.quantization_bits_ == 8
                                  ? code.ex_code()[d]
                                  : code.ex_code()[(d / 16) * 8 + d % 8];
            const auto value = graph.quantization_bits_ == 8
                                   ? byte
                                   : ((d % 16 < 8) ? byte & 15U : byte >> 4);
            dot += query[d] * (value - midpoint);
            // The existing kernel subtracts uncentered float sums; include both
            // operands in its rounding bound when the final distance is small.
            magnitude += std::abs(code.f_rescale_ex() * query[d]) * (value + midpoint);
            const double residual = static_cast<double>(query[d]) - graph.centroid_[d];
            const double term = graph.metric_type_ == METRIC_L2
                                    ? residual * residual
                                    : -static_cast<double>(query[d]) * graph.centroid_[d];
            add += term;
            magnitude += std::abs(term);
        }
        return {
            add + code.f_rescale_ex() * dot,
            8 * std::numeric_limits<float>::epsilon() * std::max(1.0, magnitude)};
    }
};

namespace {

TEST(QGConstructionTest, BuildsAndPrunesAfterInputReleaseUsingExistingCodes) {
    constexpr size_t kCount = 65, kDim = 65;
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (size_t bits : {4U, 8U}) {
            for (bool constant : {false, true}) {
                SCOPED_TRACE(
                    ::testing::Message() << metric << "/" << bits << "/" << constant
                );
                std::vector<float> data(kCount * kDim, 2.0F);
                if (!constant) {
                    for (size_t i = 0; i < data.size(); ++i) {
                        data[i] += std::sin(static_cast<float>(i) * 0.13F) *
                                   static_cast<float>(1 + (i / kDim) % 4);
                    }
                }
                QuantizedGraph<float> graph(
                    kCount, kDim, 32, metric, RotatorType::FhtKacRotator, bits
                );
                QGBuilder builder(graph, 64, data.data(), 1);
                const auto codes = QGConstructionTestAccess::codes(graph);
                std::fill(
                    data.begin(), data.end(), std::numeric_limits<float>::quiet_NaN()
                );
                std::vector<float>().swap(data);
                builder.build();
                EXPECT_EQ(QGConstructionTestAccess::codes(graph), codes);
                EXPECT_FALSE(builder.check_dup());
                EXPECT_FLOAT_EQ(builder.avg_degree(), 32);
                const auto& neighbors = QGConstructionTestAccess::neighbors(builder);
                for (PID id = 0; id < kCount; ++id) {
                    ASSERT_EQ(neighbors[id].size(), 32U);
                    for (const auto& neighbor : neighbors[id]) {
                        EXPECT_NE(neighbor.id, id);
                        ASSERT_LT(neighbor.id, kCount);
                        const auto expected =
                            QGConstructionTestAccess::estimate(graph, id, neighbor.id);
                        EXPECT_NEAR(neighbor.distance, expected[0], expected[1]);
                    }
                }
            }
        }
    }
}

TEST(QuantizedGraphConfigurationTest, RejectsDegreeNotAlignedForFastScan) {
    EXPECT_THROW(
        (QuantizedGraph<float>(64, 64, 16, METRIC_L2, RotatorType::MatrixRotator)),
        std::invalid_argument
    );
}

TEST(QuantizedGraphConfigurationTest, RejectsDegreeThatCannotExcludeSelf) {
    EXPECT_THROW(
        (QuantizedGraph<float>(32, 64, 32, METRIC_L2, RotatorType::MatrixRotator)),
        std::invalid_argument
    );
}

TEST(QuantizedGraphConfigurationTest, AcceptsOnlySupportedVectorQuantizationBits) {
    EXPECT_NO_THROW(
        (QuantizedGraph<float>(33, 64, 32, METRIC_L2, RotatorType::MatrixRotator, 0))
    );
    EXPECT_NO_THROW(
        (QuantizedGraph<float>(33, 64, 32, METRIC_L2, RotatorType::MatrixRotator, 4))
    );
    EXPECT_NO_THROW(
        (QuantizedGraph<float>(33, 64, 32, METRIC_L2, RotatorType::MatrixRotator, 8))
    );
    EXPECT_THROW(
        (QuantizedGraph<float>(33, 64, 32, METRIC_L2, RotatorType::MatrixRotator, 6)),
        std::invalid_argument
    );
}

TEST(QuantizedGraphConfigurationTest, RejectsUnsupportedMetric) {
    EXPECT_THROW(
        (QuantizedGraph<float>(
            33, 64, 32, static_cast<MetricType>(255), RotatorType::MatrixRotator
        )),
        std::invalid_argument
    );
}

TEST(QuantizedGraphLifecycleTest, DestroysConcreteRotatorThroughBasePointer) {
    QuantizedGraph<float> graph(33, 64, 32, METRIC_L2, RotatorType::MatrixRotator);
    EXPECT_EQ(graph.num_vertices(), 33U);
}

TEST(QGBuilderMetricTest, UsesInnerProductDistanceToChooseEntryPoint) {
    constexpr size_t kNumPoints = 33;
    constexpr size_t kDim = 64;
    constexpr size_t kDegree = 32;
    std::vector<float> data(kNumPoints * kDim, 0.0F);
    data[0] = 100.0F;

    const std::vector<float> centroid = compute_centroid(data.data(), kNumPoints, kDim, 1);
    const PID expected =
        exact_nn(data.data(), centroid.data(), kNumPoints, kDim, 1, dot_product_dis<float>);
    const PID euclidean_entry =
        exact_nn(data.data(), centroid.data(), kNumPoints, kDim, 1, euclidean_sqr<float>);
    ASSERT_EQ(expected, 0U);
    ASSERT_NE(euclidean_entry, expected);

    QuantizedGraph<float> graph(
        kNumPoints, kDim, kDegree, METRIC_IP, RotatorType::MatrixRotator
    );
    QGBuilder builder(graph, kDegree, data.data(), 1);

    EXPECT_EQ(graph.entry_point(), expected);
}

TEST(QGEstimatorTest, AccumulatesAcrossUint16SafeChunks) {
    constexpr std::array<size_t, 3> kDimensions = {1024, 1088, 2048};

    for (size_t padded_dim : kDimensions) {
        SCOPED_TRACE(padded_dim);

        std::vector<float> query(padded_dim, 1.0F);
        BatchQuery<float> q_obj(query.data(), padded_dim);

        std::vector<char> batch_data(QGBatchDataMap<float>::data_bytes(padded_dim));
        QGBatchDataMap<float> batch_map(batch_data.data(), padded_dim);
        std::fill(
            batch_map.bin_code(),
            batch_map.bin_code() + (padded_dim * fastscan::kBatchSize / 8),
            uint8_t{0xff}
        );
        std::fill_n(batch_map.f_add(), fastscan::kBatchSize, 0.0F);
        std::fill_n(batch_map.f_rescale(), fastscan::kBatchSize, 1.0F);

        int64_t scalar_accumulator = 0;
        for (size_t codebook = 0; codebook < padded_dim / 4; ++codebook) {
            const uint8_t selected_value = q_obj.lut()[(codebook * 16) + 15];
            ASSERT_EQ(selected_value, uint8_t{0xff});
            scalar_accumulator += selected_value;
        }

        const float expected = q_obj.delta() * static_cast<float>(scalar_accumulator) +
                               q_obj.sum_vl_lut() + q_obj.k1xsumq();
        std::array<float, fastscan::kBatchSize> estimated{};
        qg_batch_estdist(batch_data.data(), q_obj, padded_dim, estimated.data());

        for (float distance : estimated) {
            EXPECT_FLOAT_EQ(distance, expected);
        }
    }
}

TEST(QGQuantTest, SearchesAndRoundTripsFourAndEightBitIndexes) {
    constexpr size_t kNumPoints = 33;
    constexpr size_t kDim = 64;
    constexpr size_t kDegree = 32;
    std::vector<float> data(kNumPoints * kDim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.13F) +
                  std::cos(static_cast<float>(i) * 0.07F);
    }

    for (size_t bits : {4U, 8U}) {
        SCOPED_TRACE(bits);
        QuantizedGraph<float> graph(
            kNumPoints, kDim, kDegree, METRIC_L2, RotatorType::MatrixRotator, bits
        );
        {
            QGBuilder builder(graph, kDegree, data.data(), 1);
            builder.build(2);
        }
        graph.set_ef(kNumPoints);

        std::array<PID, 5> ids{};
        std::array<float, 5> distances{};
        graph.search(data.data(), ids.size(), ids.data(), distances.data());
        for (size_t i = 0; i < ids.size(); ++i) {
            EXPECT_LT(ids[i], kNumPoints);
            EXPECT_TRUE(std::isfinite(distances[i]));
        }

        const std::string path =
            ::testing::TempDir() + "rabitq_qg_quant_" + std::to_string(bits) + ".index";
        graph.save(path.c_str());
        QuantizedGraph<float> loaded;
        loaded.load(path.c_str());
        loaded.set_ef(kNumPoints);
        EXPECT_TRUE(loaded.is_quantized());
        EXPECT_EQ(loaded.quantization_bits(), bits);

        std::array<PID, 5> loaded_ids{};
        std::array<float, 5> loaded_distances{};
        loaded.search(
            data.data(), loaded_ids.size(), loaded_ids.data(), loaded_distances.data()
        );
        EXPECT_EQ(loaded_ids, ids);
        EXPECT_EQ(loaded_distances, distances);
        std::remove(path.c_str());
    }
}

TEST(QGSearchTest, ParallelQueriesMatchSerialAcrossIndexesAndSettings) {
    constexpr size_t kNumQueries = 9;
    constexpr size_t kTopK = 5;
    for (size_t bits : {0U, 4U, 8U}) {
        for (auto metric : {METRIC_L2, METRIC_IP}) {
            for (size_t dim : {65U, 128U}) {
                const size_t num_points = dim + 64;
                std::vector<float> data(num_points * dim);
                std::vector<float> queries(kNumQueries * dim);
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i] = std::sin(static_cast<float>(i) * 0.13F);
                }
                for (size_t i = 0; i < queries.size(); ++i) {
                    queries[i] = std::cos(static_cast<float>(i) * 0.07F);
                }
                QuantizedGraph<float> graph(
                    num_points, dim, 32, metric, RotatorType::FhtKacRotator, bits
                );
                {
                    QGBuilder builder(graph, 64, data.data(), 1);
                    builder.build(2);
                }
                for (size_t ef : {16U, 64U, 128U}) {
                    graph.set_ef(ef);
                    std::array<PID, kNumQueries * kTopK> serial_ids{};
                    std::array<float, kNumQueries * kTopK> serial_distances{};
                    for (size_t i = 0; i < kNumQueries; ++i) {
                        graph.search(
                            queries.data() + (i * dim),
                            kTopK,
                            serial_ids.data() + (i * kTopK),
                            serial_distances.data() + (i * kTopK)
                        );
                    }
                    for (int threads : {2, 4}) {
                        SCOPED_TRACE(threads);
                        std::array<PID, kNumQueries * kTopK> ids{};
                        std::array<float, kNumQueries * kTopK> distances{};
#pragma omp parallel for num_threads(threads) schedule(dynamic)
                        for (size_t i = 0; i < kNumQueries; ++i) {
                            graph.search(
                                queries.data() + (i * dim),
                                kTopK,
                                ids.data() + (i * kTopK),
                                distances.data() + (i * kTopK)
                            );
                        }
                        EXPECT_EQ(ids, serial_ids);
                        EXPECT_EQ(distances, serial_distances);
                    }
                }
            }
        }
    }
}

}  // namespace
}  // namespace rabitqlib::symqg
