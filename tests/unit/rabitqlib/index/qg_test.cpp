#include "rabitqlib/index/symqg/qg.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/lut.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/index/symqg/detail/pipnn.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::symqg {
struct QGConstructionTestAccess {
    static void check_contiguous_rows(QuantizedGraph<float>& graph) {
        const size_t vector_bytes =
            graph.is_quantized()
                ? ExDataMap<float>::data_bytes(graph.padded_dim_, graph.quantization_bits_)
                : graph.dim_ * sizeof(float);
        const size_t batch_bytes = QGBatchDataMap<float>::data_bytes(graph.padded_dim_) *
                                   (graph.degree_bound_ / fastscan::kBatchSize);
        const size_t row_bytes =
            vector_bytes + batch_bytes + (graph.degree_bound_ * sizeof(PID));
        const char* base = graph.get_row_data(0);
        for (PID id = 0; id < graph.num_points_; ++id) {
            const char* row = graph.get_row_data(id);
            EXPECT_EQ(row, base + (id * row_bytes));
            EXPECT_EQ(graph.get_batch_data(id), row + vector_bytes);
            if (graph.is_quantized()) {
                EXPECT_EQ(graph.get_quantized_vector(id), row);
            } else {
                EXPECT_EQ(reinterpret_cast<const char*>(graph.get_vector(id)), row);
                graph.get_vector(id)[graph.dim_ - 1] = static_cast<float>(id);
            }
            graph.get_neighbors(id)[0] = id;
            PID stored_id = kPidMax;
            std::memcpy(&stored_id, row + vector_bytes + batch_bytes, sizeof(stored_id));
            EXPECT_EQ(stored_id, id);
        }
        if (!graph.is_quantized()) {
            for (PID id = 0; id < graph.num_points_; ++id) {
                EXPECT_FLOAT_EQ(
                    graph.get_vector(id)[graph.dim_ - 1], static_cast<float>(id)
                );
            }
        }
    }

    static QGBuilder from_graph(
        QuantizedGraph<float>& graph,
        uint32_t ef,
        const float* data,
        const std::vector<size_t>& offsets,
        const std::vector<PID>& neighbors,
        size_t threads
    ) {
        QGBuilder builder(graph, ef, threads);
        builder.initialize_seed(data, offsets, neighbors);
        return builder;
    }
    static auto codes(const QuantizedGraph<float>& graph) {
        std::vector<char> result;
        for (PID id = 0; id < graph.num_points_; ++id) {
            const char* code = graph.is_quantized()
                                   ? graph.get_quantized_vector(id)
                                   : reinterpret_cast<const char*>(graph.get_vector(id));
            result.insert(result.end(), code, code + graph.batch_data_offset_);
        }
        return result;
    }
    static const auto& neighbors(const QGBuilder& builder) {
        return builder.new_neighbors_;
    }
    static const auto& degrees(const QGBuilder& builder) { return builder.degrees_; }
    static void retain_seed_scores(QGBuilder& builder) {
        auto& graph = builder.qg_;
        for (PID i = 0; i < builder.num_nodes_; ++i) {
            std::vector<float> reconstructed;
            std::optional<QuantizedQuery<float>> prepared;
            const float* source = graph.prepare_build_query(i, reconstructed, prepared);
            for (size_t j = 0; j < builder.degrees_[i]; ++j) {
                const PID id = graph.get_neighbors(i)[j];
                builder.new_neighbors_[i].emplace_back(
                    id, graph.point_distance(source, prepared ? &*prepared : nullptr, id)
                );
            }
        }
    }
    static void search(QGBuilder& builder) { builder.search_new_neighbors(false); }
    static auto encoded_neighbors(const QuantizedGraph<float>& graph, PID id) {
        return graph.get_neighbors(id);
    }
    static void set_encoded_neighbor(
        QuantizedGraph<float>& graph, PID source, size_t lane, PID target
    ) {
        graph.get_neighbors(source)[lane] = target;
    }
    static void copy_vectors(
        QuantizedGraph<float>& graph, const float* data, size_t threads
    ) {
        graph.copy_vectors(data, threads);
    }
    static void update(
        QuantizedGraph<float>& graph,
        PID source,
        const std::vector<AnnCandidate<float>>& neighbors
    ) {
        graph.update_qg(source, neighbors);
    }
    static void fill_batch_factors(QuantizedGraph<float>& graph, PID source, float value) {
        QGBatchDataMap<float> batch(graph.get_batch_data(source), graph.padded_dim_);
        std::fill_n(batch.f_add(), fastscan::kBatchSize, value);
        std::fill_n(batch.f_rescale(), fastscan::kBatchSize, value);
    }
    static std::pair<float, float> batch_factors(
        const QuantizedGraph<float>& graph, PID source, size_t lane
    ) {
        ConstQGBatchDataMap<float> batch(graph.get_batch_data(source), graph.padded_dim_);
        return {batch.f_add()[lane], batch.f_rescale()[lane]};
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

TEST(QuantizedGraphLayoutTest, KeepsVectorsCodesAndNeighborsInContiguousRows) {
    for (size_t bits : {0U, 4U, 8U}) {
        SCOPED_TRACE(bits);
        QuantizedGraph<float> graph(
            33, 65, 32, METRIC_L2, RotatorType::FhtKacRotator, bits
        );
        QGConstructionTestAccess::check_contiguous_rows(graph);
    }
}

static_assert(std::is_copy_constructible_v<Lut<float>>);
static_assert(std::is_move_constructible_v<Lut<float>>);

static_assert(
    !std::is_constructible_v<
        QGBuilder,
        QuantizedGraph<float>&,
        uint32_t,
        const float*,
        const std::vector<size_t>&,
        const std::vector<PID>&,
        size_t>,
    "Intermediate graph input must not be part of the public builder API"
);

TEST(QGEstimatorTest, MatchesExactDistancesForCollinearResiduals) {
    if (!cpu::has_avx2()) {
        GTEST_SKIP() << "FastScan requires AVX2/FMA";
    }
    for (size_t dim : {64U, 1024U, 1088U}) {
        for (auto metric : {METRIC_L2, METRIC_IP}) {
            SCOPED_TRACE(::testing::Message() << dim << "/" << metric);
            std::vector<float> centroid(dim, 0.125F), query(dim, 0.0625F);
            std::vector<float> data(fastscan::kBatchSize * dim);
            for (size_t i = 0; i < fastscan::kBatchSize; ++i) {
                // Collinear residuals have exact codes, including the zero-residual case.
                const float value = 0.125F + (static_cast<float>(i % 3) - 1.0F) * 0.03125F;
                std::fill_n(data.data() + i * dim, dim, value);
            }
            std::vector<char> batch(QGBatchDataMap<float>::data_bytes(dim));
            quant::quantize_qg_batch(
                data.data(),
                centroid.data(),
                fastscan::kBatchSize,
                dim,
                batch.data(),
                metric
            );
            const auto distance =
                metric == METRIC_IP ? dot_product_dis<float> : euclidean_sqr<float>;
            const float vertex_distance = distance(query.data(), centroid.data(), dim);
            BatchQuery<float> q_obj(query.data(), dim, metric);
            q_obj.set_g_add(vertex_distance);
            std::array<float, fastscan::kBatchSize> estimates{};
            qg_batch_estdist(batch.data(), q_obj, dim, estimates.data());
            EXPECT_FLOAT_EQ(
                q_obj.g_add(), metric == METRIC_IP ? vertex_distance - 1 : vertex_distance
            );
            for (size_t i = 0; i < fastscan::kBatchSize; ++i) {
                EXPECT_NEAR(
                    estimates[i], distance(query.data(), data.data() + i * dim, dim), 1e-5F
                );
            }
        }
    }
}

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

TEST(QGConstructionTest, EncodedSeedSearchMatchesRetainedScores) {
    if (!cpu::has_avx2()) {
        GTEST_SKIP() << "FastScan requires AVX2/FMA";
    }
    constexpr size_t kCount = 97, kDim = 65, kDegree = 32;
    std::vector<float> data(kCount * kDim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.13F);
    }
    std::vector<size_t> offsets{0};
    std::vector<PID> edges;
    for (PID i = 0; i < kCount; ++i) {
        for (size_t j = 0; j < i % (kDegree + 1); ++j) {
            edges.push_back((i + j + 1) % kCount);
        }
        offsets.push_back(edges.size());
    }
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (size_t bits : {0U, 4U, 8U}) {
            SCOPED_TRACE(::testing::Message() << metric << "/" << bits);
            QuantizedGraph<float> graph(
                kCount, kDim, kDegree, metric, RotatorType::FhtKacRotator, bits
            );
            // Both builders search the same immutable encoded graph/rotation.
            auto compact = QGConstructionTestAccess::from_graph(
                graph, 8, data.data(), offsets, edges, 2
            );
            auto retained = QGConstructionTestAccess::from_graph(
                graph, 8, data.data(), offsets, edges, 2
            );
            QGConstructionTestAccess::retain_seed_scores(retained);
            QGConstructionTestAccess::search(compact);
            QGConstructionTestAccess::search(retained);
            for (PID i = 0; i < kCount; ++i) {
                const auto& actual = QGConstructionTestAccess::neighbors(compact)[i];
                const auto& expected = QGConstructionTestAccess::neighbors(retained)[i];
                ASSERT_EQ(actual.size(), expected.size());
                for (size_t j = 0; j < actual.size(); ++j) {
                    EXPECT_EQ(actual[j].id, expected[j].id);
                    EXPECT_FLOAT_EQ(actual[j].distance, expected[j].distance);
                }
            }
        }
    }
}

TEST(QGConstructionTest, DefaultsToPipnnInitialization) {
    if (!cpu::has_avx2()) {
        GTEST_SKIP() << "FastScan requires AVX2/FMA";
    }
    constexpr size_t kCount = 97, kDim = 65, kDegree = 32;
    std::vector<float> data(kCount * kDim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.13F);
    }
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        const auto seed =
            detail::build_initial_graph(data.data(), kCount, kDim, kDegree, metric, 1);
        for (size_t bits : {0U, 4U, 8U}) {
            SCOPED_TRACE(::testing::Message() << metric << "/" << bits);
            QuantizedGraph<float> graph(
                kCount, kDim, kDegree, metric, RotatorType::FhtKacRotator, bits
            );
            QGBuilder builder(graph, 64, data.data(), 1);
            for (PID i = 0; i < kCount; ++i) {
                std::vector<PID> expected(
                    seed.neighbors.begin() + static_cast<ptrdiff_t>(seed.offsets[i]),
                    seed.neighbors.begin() + static_cast<ptrdiff_t>(seed.offsets[i + 1])
                );
                std::sort(expected.begin(), expected.end());
                ASSERT_EQ(QGConstructionTestAccess::degrees(builder)[i], expected.size());
                EXPECT_TRUE(QGConstructionTestAccess::neighbors(builder)[i].empty());
                const auto actual = QGConstructionTestAccess::encoded_neighbors(graph, i);
                for (size_t j = 0; j < expected.size(); ++j) {
                    EXPECT_EQ(actual[j], expected[j]);
                }
            }
            builder.build();
            EXPECT_FLOAT_EQ(builder.avg_degree(), kDegree);
            EXPECT_FALSE(builder.check_dup());
        }
    }
}

TEST(QGConstructionTest, SupportsExplicitRandomInitialization) {
    if (!cpu::has_avx2()) {
        GTEST_SKIP() << "FastScan requires AVX2/FMA";
    }
    std::vector<float> data(65 * 65, 0.5F);
    QuantizedGraph<float> graph(65, 65, 32);
    QGBuilder builder(graph, 64, data.data(), 1, QGInitialization::Random);
    for (const auto& row : QGConstructionTestAccess::neighbors(builder)) {
        EXPECT_EQ(row.size(), 32U);
    }
    builder.build();
    EXPECT_FLOAT_EQ(builder.avg_degree(), 32);
    EXPECT_FALSE(builder.check_dup());
    EXPECT_THROW(builder.build(0), std::invalid_argument);
    EXPECT_THROW(builder.build(1), std::invalid_argument);
    EXPECT_THROW(
        (QGBuilder(graph, 64, data.data(), 1, static_cast<QGInitialization>(255))),
        std::invalid_argument
    );
}

TEST(QGConstructionTest, RejectsNullDataForEveryInitializationMode) {
    QuantizedGraph<float> graph(33, 64, 32);
    EXPECT_THROW(
        (QGBuilder(graph, 32, nullptr, 1, QGInitialization::PiPNN)), std::invalid_argument
    );
    EXPECT_THROW(
        (QGBuilder(graph, 32, nullptr, 1, QGInitialization::Random)), std::invalid_argument
    );
}

TEST(QGConstructionTest, UsesExplicitThreadCountsWithoutChangingCallerState) {
    if (!cpu::has_avx2()) {
        GTEST_SKIP() << "FastScan requires AVX2/FMA";
    }
    constexpr size_t kCount = 33, kDim = 64;
    std::vector<float> data(kCount * kDim, 0.25F);
    const int caller_threads = omp_get_max_threads();

    QuantizedGraph<float> first(kCount, kDim, 32);
    QuantizedGraph<float> second(kCount, kDim, 32);
    QGBuilder first_builder(first, 32, data.data(), 1, QGInitialization::Random);
    EXPECT_EQ(omp_get_max_threads(), caller_threads);
    QGBuilder second_builder(second, 32, data.data(), 2, QGInitialization::PiPNN);
    EXPECT_EQ(omp_get_max_threads(), caller_threads);

    first_builder.build(2);
    second_builder.build(2);
    EXPECT_EQ(omp_get_max_threads(), caller_threads);
}

TEST(QGConstructionTest, InitializesUnusedPartialBatchFactors) {
    if (!cpu::has_avx2()) {
        GTEST_SKIP() << "FastScan requires AVX2/FMA";
    }
    constexpr size_t kCount = 33, kDim = 64;
    std::vector<float> data(kCount * kDim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.17F);
    }
    QuantizedGraph<float> graph(kCount, kDim, 32);
    QGConstructionTestAccess::copy_vectors(graph, data.data(), 1);
    QGConstructionTestAccess::fill_batch_factors(
        graph, 0, std::numeric_limits<float>::quiet_NaN()
    );
    std::vector<AnnCandidate<float>> neighbors;
    neighbors.emplace_back(1, 0.0F);
    QGConstructionTestAccess::update(graph, 0, neighbors);

    const auto active = QGConstructionTestAccess::batch_factors(graph, 0, 0);
    EXPECT_TRUE(std::isfinite(active.first));
    EXPECT_TRUE(std::isfinite(active.second));
    for (size_t lane = 1; lane < fastscan::kBatchSize; ++lane) {
        const auto factors = QGConstructionTestAccess::batch_factors(graph, 0, lane);
        EXPECT_FLOAT_EQ(factors.first, 0.0F);
        EXPECT_FLOAT_EQ(factors.second, 0.0F);
    }
}

TEST(QGConstructionTest, RefinesPartialSeedOnceAfterReleasingInputs) {
    if (!cpu::has_avx2()) {
        GTEST_SKIP() << "FastScan requires AVX2/FMA";
    }
    constexpr size_t kCount = 97, kDim = 65, kDegree = 64;
    constexpr std::array<size_t, 6> kDegrees{0, 1, 31, 32, 33, 64};
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (size_t bits : {0U, 4U, 8U}) {
            SCOPED_TRACE(::testing::Message() << metric << "/" << bits);
            std::vector<float> data(kCount * kDim);
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = std::sin(static_cast<float>(i) * 0.13F);
            }
            const std::vector<float> query(data.begin(), data.begin() + kDim);
            std::vector<size_t> offsets{0};
            std::vector<PID> edges;
            for (size_t i = 0; i < kCount; ++i) {
                for (size_t j = 0; j < kDegrees[i % kDegrees.size()]; ++j) {
                    // Repeated IDs exercise import deduplication, including full rows.
                    edges.push_back((i + 1 + (i < kDegrees.size() ? j : j / 2)) % kCount);
                }
                offsets.push_back(edges.size());
            }
            QuantizedGraph<float> graph(
                kCount, kDim, kDegree, metric, RotatorType::FhtKacRotator, bits
            );
            auto builder = QGConstructionTestAccess::from_graph(
                graph, 96, data.data(), offsets, edges, 2
            );
            const auto stored_vectors = QGConstructionTestAccess::codes(graph);
            for (PID i = 0; i < kCount; ++i) {
                std::vector<PID> expected(
                    edges.begin() + static_cast<ptrdiff_t>(offsets[i]),
                    edges.begin() + static_cast<ptrdiff_t>(offsets[i + 1])
                );
                std::sort(expected.begin(), expected.end());
                expected.erase(
                    std::unique(expected.begin(), expected.end()), expected.end()
                );
                ASSERT_EQ(QGConstructionTestAccess::degrees(builder)[i], expected.size());
                const auto& row = QGConstructionTestAccess::neighbors(builder)[i];
                EXPECT_EQ(row.capacity(), 0U);
                for (size_t j = 0; j < expected.size(); ++j) {
                    EXPECT_EQ(
                        QGConstructionTestAccess::encoded_neighbors(graph, i)[j],
                        expected[j]
                    );
                }
            }
            std::vector<float>().swap(data);
            std::vector<size_t>().swap(offsets);
            std::vector<PID>().swap(edges);
            builder.refine();
            EXPECT_EQ(stored_vectors, QGConstructionTestAccess::codes(graph));
            EXPECT_FALSE(builder.check_dup());
            for (PID i = 0; i < kCount; ++i) {
                const auto& row = QGConstructionTestAccess::neighbors(builder)[i];
                ASSERT_EQ(row.size(), kDegree);
                EXPECT_EQ(QGConstructionTestAccess::degrees(builder)[i], kDegree);
                for (size_t j = 0; j < row.size(); ++j) {
                    EXPECT_LT(row[j].id, kCount);
                    EXPECT_NE(row[j].id, i);
                    EXPECT_TRUE(std::isfinite(row[j].distance));
                    EXPECT_EQ(
                        QGConstructionTestAccess::encoded_neighbors(graph, i)[j], row[j].id
                    );
                    if (bits != 0) {
                        const auto expected =
                            QGConstructionTestAccess::estimate(graph, i, row[j].id);
                        EXPECT_NEAR(row[j].distance, expected[0], expected[1]);
                    }
                }
            }
            graph.set_ef(96);
            std::array<PID, 10> ids{}, loaded_ids{};
            std::array<float, 10> distances{}, loaded_distances{};
            graph.search(query.data(), 10, ids.data(), distances.data());
            for (size_t j = 0; j < ids.size(); ++j) {
                EXPECT_LT(ids[j], kCount);
                EXPECT_TRUE(std::isfinite(distances[j]));
            }
            const std::string path = ::testing::TempDir() + "rabitq_qg_seeded.index";
            graph.save(path.c_str());
            QuantizedGraph<float> loaded;
            loaded.load(path.c_str());
            loaded.set_ef(96);
            loaded.search(query.data(), 10, loaded_ids.data(), loaded_distances.data());
            EXPECT_EQ(ids, loaded_ids);
            EXPECT_EQ(distances, loaded_distances);
            EXPECT_EQ(graph.entry_point(), loaded.entry_point());
            std::remove(path.c_str());
        }
    }
}

TEST(QGConstructionTest, RejectsInvalidSeedStructureAndIds) {
    constexpr size_t kCount = 33, kDim = 64;
    std::vector<float> data(kCount * kDim, 0.1F);
    QuantizedGraph<float> graph(kCount, kDim, 32);
    const auto reject = [&](const std::vector<size_t>& offsets,
                            const std::vector<PID>& edges,
                            const char* message) {
        try {
            auto builder = QGConstructionTestAccess::from_graph(
                graph, 32, data.data(), offsets, edges, 1
            );
            FAIL() << "Invalid seed accepted";
        } catch (const std::invalid_argument& error) {
            EXPECT_STREQ(error.what(), message);
        }
    };
    reject({}, {}, "Seed graph offsets must delimit every vertex");
    std::vector<size_t> offsets(kCount + 1, 0);
    offsets.back() = 1;
    reject(offsets, {}, "Seed graph offsets must delimit every vertex");
    offsets.back() = 0;
    offsets[1] = 1;
    reject(offsets, {}, "Seed graph row exceeds degree bound or has invalid offsets");
    std::fill(offsets.begin() + 1, offsets.end(), 1);
    reject(offsets, {kCount}, "Seed graph IDs must be in range and exclude self");
    reject(offsets, {0}, "Seed graph IDs must be in range and exclude self");
    std::fill(offsets.begin() + 1, offsets.end(), 33);
    reject(
        offsets,
        std::vector<PID>(33, 1),
        "Seed graph row exceeds degree bound or has invalid offsets"
    );
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

TEST(QuantizedGraphConfigurationTest, RejectsZeroDimension) {
    EXPECT_THROW(
        (QuantizedGraph<float>(33, 0, 32, METRIC_L2, RotatorType::MatrixRotator)),
        std::invalid_argument
    );
}

TEST(QuantizedGraphConfigurationTest, RejectsOutOfRangeEntryPoint) {
    QuantizedGraph<float> graph(33, 64, 32);
    EXPECT_NO_THROW(graph.set_ep(32));
    EXPECT_THROW(graph.set_ep(33), std::invalid_argument);
    EXPECT_EQ(graph.entry_point(), 32U);
}

TEST(QuantizedGraphPersistenceTest, RejectsMalformedPayloadWithoutChangingTarget) {
    constexpr size_t kNumPoints = 33;
    constexpr size_t kDim = 64;
    constexpr size_t kDegree = 32;
    std::vector<float> data(kNumPoints * kDim, 0.25F);
    QuantizedGraph<float> source(
        kNumPoints, kDim, kDegree, METRIC_L2, RotatorType::MatrixRotator, 4
    );
    QGBuilder builder(source, kDegree, data.data(), 1);
    builder.build(2);

    const std::string path = ::testing::TempDir() + "rabitq_qg_malformed.index";
    QuantizedGraph<float> target(65, kDim, kDegree, METRIC_IP);

    source.save(path.c_str());
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    EXPECT_THROW(target.load(path.c_str()), std::runtime_error);
    EXPECT_EQ(target.num_vertices(), 65U);
    EXPECT_EQ(target.metric_type(), METRIC_IP);

    source.save(path.c_str());
    {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(file.is_open());
        constexpr std::streamoff kPaddedDimensionOffset =
            sizeof(uint64_t) + sizeof(uint32_t) + (3 * sizeof(size_t));
        file.seekp(kPaddedDimensionOffset);
        const size_t invalid_padded_dim = kDim * 2;
        file.write(
            reinterpret_cast<const char*>(&invalid_padded_dim), sizeof(invalid_padded_dim)
        );
        ASSERT_TRUE(file.good());
    }
    EXPECT_THROW(target.load(path.c_str()), std::runtime_error);
    EXPECT_EQ(target.num_vertices(), 65U);
    EXPECT_EQ(target.metric_type(), METRIC_IP);

    source.save(path.c_str());
    {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(file.is_open());
        constexpr std::streamoff kPointCountOffset = sizeof(uint64_t) + sizeof(uint32_t);
        file.seekp(kPointCountOffset);
        const size_t invalid_point_count = std::numeric_limits<size_t>::max();
        file.write(
            reinterpret_cast<const char*>(&invalid_point_count), sizeof(invalid_point_count)
        );
        ASSERT_TRUE(file.good());
    }
    EXPECT_THROW(target.load(path.c_str()), std::invalid_argument);
    EXPECT_EQ(target.num_vertices(), 65U);
    EXPECT_EQ(target.metric_type(), METRIC_IP);

    if (std::filesystem::exists("/dev/full")) {
        EXPECT_THROW(source.save("/dev/full"), std::ios_base::failure);
    }

    QGConstructionTestAccess::set_encoded_neighbor(source, 0, 0, kNumPoints);
    source.save(path.c_str());
    EXPECT_THROW(target.load(path.c_str()), std::runtime_error);
    EXPECT_EQ(target.num_vertices(), 65U);
    EXPECT_EQ(target.metric_type(), METRIC_IP);

    std::remove(path.c_str());
}

TEST(QuantizedGraphLifecycleTest, DestroysConcreteRotatorThroughBasePointer) {
    QuantizedGraph<float> graph(33, 64, 32, METRIC_L2, RotatorType::MatrixRotator);
    EXPECT_EQ(graph.num_vertices(), 33U);
}

TEST(QuantizedGraphLifecycleTest, RejectsSearchAndSaveBeforeBuild) {
    const std::string path = ::testing::TempDir() + "rabitq_qg_unbuilt.index";
    std::remove(path.c_str());
    std::array<float, 64> query{};
    std::array<PID, 1> ids{};
    std::array<float, 1> distances{};

    QuantizedGraph<float> empty;
    EXPECT_THROW(empty.save(path.c_str()), std::logic_error);
    EXPECT_THROW(
        empty.search(query.data(), 1, ids.data(), distances.data()), std::logic_error
    );

    QuantizedGraph<float> configured(33, 64, 32);
    configured.set_ef(1);
    EXPECT_THROW(configured.save(path.c_str()), std::logic_error);
    EXPECT_THROW(
        configured.search(query.data(), 1, ids.data(), distances.data()), std::logic_error
    );
    EXPECT_FALSE(std::filesystem::exists(path));
    std::remove(path.c_str());
}

TEST(QuantizedGraphLifecycleTest, BuilderDoesNotPublishPartialGraph) {
    if (!cpu::has_avx2()) {
        GTEST_SKIP() << "FastScan requires AVX2/FMA";
    }
    constexpr size_t kNumPoints = 33;
    constexpr size_t kDim = 64;
    constexpr size_t kDegree = 32;
    std::vector<float> data(kNumPoints * kDim, 0.25F);
    std::array<float, kDim> query{};
    PID result = 0;
    float distance = 0;

    for (const auto init : {QGInitialization::Random, QGInitialization::PiPNN}) {
        SCOPED_TRACE(static_cast<int>(init));
        const std::string path = ::testing::TempDir() + "rabitq_qg_partial.index";
        std::remove(path.c_str());
        QuantizedGraph<float> graph(kNumPoints, kDim, kDegree);
        QGBuilder builder(graph, kDegree, data.data(), 1, init);
        graph.set_ef(1);
        EXPECT_THROW(graph.save(path.c_str()), std::logic_error);
        EXPECT_THROW(graph.search(query.data(), 1, &result, &distance), std::logic_error);
        EXPECT_FALSE(std::filesystem::exists(path));

        builder.build(2);
        EXPECT_NO_THROW(graph.search(query.data(), 1, &result, &distance));
        EXPECT_NO_THROW(graph.save(path.c_str()));
        EXPECT_TRUE(std::filesystem::exists(path));
        std::remove(path.c_str());
    }
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
        loaded.set_ef(kNumPoints);
        loaded.load(path.c_str());
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

TEST(QGSearchTest, RejectsInvalidKAndEfInsteadOfReturningPartialResults) {
    if (!cpu::has_avx2()) {
        GTEST_SKIP() << "FastScan requires AVX2/FMA";
    }
    constexpr size_t kNumPoints = 33, kDim = 64, kDegree = 32;
    std::vector<float> data(kNumPoints * kDim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.11F);
    }
    QuantizedGraph<float> graph(kNumPoints, kDim, kDegree);
    QGBuilder builder(graph, kDegree, data.data(), 1, QGInitialization::Random);
    builder.build(2);

    std::array<PID, kNumPoints + 1> ids{};
    std::array<float, kNumPoints + 1> distances{};
    EXPECT_THROW(graph.set_ef(0), std::invalid_argument);
    graph.set_ef(4);
    EXPECT_THROW(
        graph.search(data.data(), 5, ids.data(), distances.data()), std::invalid_argument
    );
    graph.set_ef(kNumPoints);
    EXPECT_THROW(
        graph.search(data.data(), 0, ids.data(), distances.data()), std::invalid_argument
    );
    EXPECT_THROW(
        graph.search(
            data.data(), static_cast<uint32_t>(kNumPoints + 1), ids.data(), distances.data()
        ),
        std::invalid_argument
    );
    EXPECT_THROW(
        graph.search(nullptr, 1, ids.data(), distances.data()), std::invalid_argument
    );
    EXPECT_THROW(
        graph.search(data.data(), 1, nullptr, distances.data()), std::invalid_argument
    );
    EXPECT_THROW(graph.search(data.data(), 1, ids.data(), nullptr), std::invalid_argument);

    graph.search(
        data.data(), static_cast<uint32_t>(kNumPoints), ids.data(), distances.data()
    );
    for (size_t i = 0; i < kNumPoints; ++i) {
        EXPECT_LT(ids[i], kNumPoints);
        EXPECT_TRUE(std::isfinite(distances[i]));
    }

    for (PID source = 0; source < kNumPoints; ++source) {
        for (size_t lane = 0; lane < kDegree; ++lane) {
            QGConstructionTestAccess::set_encoded_neighbor(graph, source, lane, 0);
        }
    }
    graph.set_ef(3);
    ids.fill(kPidMax);
    distances.fill(-123.0F);
    EXPECT_THROW(
        graph.search(data.data(), 3, ids.data(), distances.data()), std::runtime_error
    );
    EXPECT_TRUE(std::all_of(ids.begin(), ids.end(), [](PID id) { return id == kPidMax; }));
    EXPECT_TRUE(std::all_of(distances.begin(), distances.end(), [](float distance) {
        return distance == -123.0F;
    }));
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
