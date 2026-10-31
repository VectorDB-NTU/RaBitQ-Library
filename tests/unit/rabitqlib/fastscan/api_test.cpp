#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rabitqlib/core/data_layout.hpp"
#include "rabitqlib/core/estimator.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/detail/lut.hpp"
#include "rabitqlib/fastscan/encoder.hpp"
#include "rabitqlib/fastscan/estimator.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib {
namespace {

template <typename Call>
void expect_invalid(Call call, const char* message) {
    try {
        call();
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& error) { EXPECT_STREQ(error.what(), message); }
}

TEST(FastScanApiTest, RejectsUnpreparedAndMismatchedQueriesBeforeWriting) {
    constexpr size_t kDim = 64;
    std::array<float, kDim> values{};
    values.fill(1);
    std::vector<char> batch(BatchDataMap<float>::data_bytes(kDim));
    std::array<float, 32> distance, lower, ip;
    distance.fill(123);
    lower.fill(123);
    ip.fill(123);
    constexpr auto* kShapeError =
        "FastScan scoring requires a prepared query with matching dimension";
    constexpr auto* kModeError = "FastScan scoring mode must match the prepared query";

    SplitBatchQuery<float> empty_split;
    EXPECT_EQ(empty_split.rotated_query(), nullptr);
    expect_invalid(
        [&] {
            split_batch_estdist(
                batch.data(),
                empty_split,
                kDim,
                distance.data(),
                lower.data(),
                ip.data(),
                true
            );
        },
        kShapeError
    );

    BatchQuery<float> query;
    expect_invalid(
        [&] { qg_batch_estdist(batch.data(), query, kDim, distance.data()); }, kShapeError
    );
    std::array<float, kDim * 4> scratch{};
    query.reset(values.data(), kDim, scratch.data());
    for (size_t dim : {0U, 16U, 32U, 63U, 128U}) {
        expect_invalid(
            [&] { qg_batch_estdist(batch.data(), query, dim, distance.data()); },
            kShapeError
        );
    }
    auto moved_query = std::move(query);
    expect_invalid(
        [&] { qg_batch_estdist(batch.data(), query, kDim, distance.data()); }, kShapeError
    );
    EXPECT_EQ(moved_query.padded_dim(), kDim);

    for (bool hacc : {false, true}) {
        SplitBatchQuery<float> split(values.data(), kDim, 3, METRIC_L2, hacc);
        expect_invalid(
            [&] {
                split_batch_estdist(
                    batch.data(),
                    split,
                    kDim,
                    distance.data(),
                    lower.data(),
                    ip.data(),
                    !hacc
                );
            },
            kModeError
        );
        for (size_t dim : {0U, 32U, 128U}) {
            expect_invalid(
                [&] {
                    split_batch_estdist(
                        batch.data(),
                        split,
                        dim,
                        distance.data(),
                        lower.data(),
                        ip.data(),
                        hacc
                    );
                },
                kShapeError
            );
        }
        auto moved_split = std::move(split);
        expect_invalid(
            [&] {
                split_batch_estdist(
                    batch.data(),
                    split,
                    kDim,
                    distance.data(),
                    lower.data(),
                    ip.data(),
                    hacc
                );
            },
            kShapeError
        );
        EXPECT_EQ(moved_split.padded_dim(), kDim);
        EXPECT_EQ(moved_split.use_hacc(), hacc);
    }
    for (size_t i = 0; i < distance.size(); ++i) {
        EXPECT_EQ(distance[i], 123);
        EXPECT_EQ(lower[i], 123);
        EXPECT_EQ(ip[i], 123);
    }
    EXPECT_NO_THROW(qg_batch_estdist(batch.data(), moved_query, kDim, distance.data()));
}

TEST(FastScanApiTest, SplitQueryResetMatchesFreshPreparationAndScoring) {
    std::array<std::array<float, 128>, 2> queries{};
    std::array<float, 128 * 4> scratch{};
    std::array<uint16_t, 128 * 4> hacc_scratch{};
    SplitBatchQuery<float> reused;
    struct Case {
        size_t dim;
        size_t ex_bits;
        MetricType metric;
        bool hacc;
    };
    const std::array<Case, 4> cases{{
        {64, 0, METRIC_L2, false},
        {96, 3, METRIC_IP, true},
        {32, 8, METRIC_IP, false},
        {128, 1, METRIC_L2, true},
    }};
    for (size_t step = 0; step < cases.size(); ++step) {
        const auto [dim, ex_bits, metric, hacc] = cases[step];
        SCOPED_TRACE(step);
        auto& values = queries[step % queries.size()];
        std::vector<float> data(17 * dim), centroid(dim);
        for (size_t i = 0; i < dim; ++i) {
            values[i] = std::cos(static_cast<float>(i + step) * 0.13F);
            centroid[i] = 0.1F * std::sin(static_cast<float>(i) * 0.23F);
        }
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = std::sin(static_cast<float>(i) * 0.17F);
        }
        SplitBatchQuery<float> fresh(values.data(), dim, ex_bits, metric, hacc);
        reused.set_g_add(3, 2);
        reused.reset(
            values.data(),
            dim,
            ex_bits,
            scratch.data(),
            hacc ? hacc_scratch.data() : nullptr,
            metric,
            hacc
        );
        EXPECT_EQ(reused.rotated_query(), values.data());
        EXPECT_EQ(reused.padded_dim(), dim);
        EXPECT_EQ(reused.use_hacc(), hacc);
        EXPECT_EQ(reused.g_add(), 0);
        EXPECT_EQ(reused.g_error(), 0);
        EXPECT_EQ(reused.delta(), fresh.delta());
        EXPECT_EQ(reused.sum_vl_lut(), fresh.sum_vl_lut());
        EXPECT_EQ(reused.g_k1xsumq(), fresh.g_k1xsumq());
        EXPECT_EQ(reused.g_kbxsumq(), fresh.g_kbxsumq());
        const size_t bytes = dim * 4 * (hacc ? 2 : 1);
        EXPECT_TRUE(std::equal(fresh.lut(), fresh.lut() + bytes, reused.lut()));
        scratch.fill(123);
        hacc_scratch.fill(123);

        const float norm = std::sqrt(euclidean_sqr(values.data(), centroid.data(), dim));
        const float ip = dot_product(values.data(), centroid.data(), dim);
        fresh.set_g_add(norm, ip);
        reused.set_g_add(norm, ip);
        EXPECT_EQ(reused.g_add(), metric == METRIC_L2 ? norm * norm : -ip);
        EXPECT_EQ(reused.g_error(), norm);
        std::vector<char> batch(BatchDataMap<float>::data_bytes(dim));
        const size_t stride = ExDataMap<float>::data_bytes(dim, ex_bits);
        std::vector<char> extra(32 * stride);
        quant::quantize_split_batch(
            data.data(),
            centroid.data(),
            17,
            dim,
            ex_bits,
            batch.data(),
            extra.data(),
            metric
        );
        std::array<float, 32> fresh_dist{}, fresh_low{}, fresh_ip{};
        std::array<float, 32> reused_dist{}, reused_low{}, reused_ip{};
        split_batch_estdist(
            batch.data(),
            fresh,
            dim,
            fresh_dist.data(),
            fresh_low.data(),
            fresh_ip.data(),
            hacc
        );
        split_batch_estdist(
            batch.data(),
            reused,
            dim,
            reused_dist.data(),
            reused_low.data(),
            reused_ip.data(),
            hacc
        );
        for (size_t row = 0; row < 17; ++row) {
            EXPECT_EQ(reused_dist[row], fresh_dist[row]);
            EXPECT_EQ(reused_low[row], fresh_low[row]);
            EXPECT_EQ(reused_ip[row], fresh_ip[row]);
            if (ex_bits != 0) {
                const auto kernel = select_excode_ipfunc(ex_bits);
                EXPECT_EQ(
                    split_distance_boosting(
                        extra.data() + row * stride,
                        kernel,
                        reused,
                        dim,
                        ex_bits,
                        reused_ip[row]
                    ),
                    split_distance_boosting(
                        extra.data() + row * stride,
                        kernel,
                        fresh,
                        dim,
                        ex_bits,
                        fresh_ip[row]
                    )
                );
            }
        }
    }
}

TEST(FastScanApiTest, SplitQueryInitializationValidatesParametersConsistently) {
    std::array<float, 64> values{};
    values.fill(0.5F);
    std::array<float, 64 * 4> scratch{};
    std::array<uint16_t, 64 * 4> hacc_scratch{};
    SplitBatchQuery<float> query;
    for (bool hacc : {false, true}) {
        for (size_t bits : {size_t{9}, std::numeric_limits<size_t>::max()}) {
            const auto* message = "SplitBatchQuery requires ex_bits in [0, 8]";
            expect_invalid(
                [&] { SplitBatchQuery<float> invalid(nullptr, 64, bits, METRIC_L2, hacc); },
                message
            );
            expect_invalid(
                [&] { query.reset(nullptr, 64, bits, nullptr, nullptr, METRIC_L2, hacc); },
                message
            );
        }
        for (size_t bits : {0U, 8U}) {
            const auto invalid_metric = static_cast<MetricType>(7);
            const auto* message = "Metric type must be METRIC_L2 or METRIC_IP";
            expect_invalid(
                [&] {
                    SplitBatchQuery<float> invalid(nullptr, 64, bits, invalid_metric, hacc);
                },
                message
            );
            expect_invalid(
                [&] {
                    query.reset(nullptr, 64, bits, nullptr, nullptr, invalid_metric, hacc);
                },
                message
            );
            for (auto metric : {METRIC_L2, METRIC_IP}) {
                SplitBatchQuery<float> fresh(values.data(), 64, bits, metric, hacc);
                query.reset(
                    values.data(),
                    64,
                    bits,
                    scratch.data(),
                    hacc_scratch.data(),
                    metric,
                    hacc
                );
                EXPECT_EQ(query.g_kbxsumq(), fresh.g_kbxsumq());
                EXPECT_EQ(query.use_hacc(), fresh.use_hacc());
            }
        }
    }
}

TEST(FastScanApiTest, RejectedSplitQueryResetPreservesPreparedState) {
    std::array<float, 64> values{}, next{};
    values.fill(0.5F);
    next.fill(0.25F);
    std::array<float, 64 * 4> scratch{};
    std::array<uint16_t, 64 * 4> hacc_scratch{};
    SplitBatchQuery<float> query(values.data(), 64, 3, METRIC_IP, false);
    query.set_g_add(3, 2);
    const std::vector<uint8_t> bytes(query.lut(), query.lut() + 64 * 4);
    const auto original = query;
    const auto reset =
        [&](const float* input, size_t dim, size_t bits, float* lut, uint16_t* u16) {
            query.reset(input, dim, bits, lut, u16, METRIC_L2, true);
        };
    for (size_t bits : {size_t{9}, std::numeric_limits<size_t>::max()}) {
        expect_invalid(
            [&] { reset(next.data(), 64, bits, scratch.data(), hacc_scratch.data()); },
            "SplitBatchQuery requires ex_bits in [0, 8]"
        );
    }
    expect_invalid(
        [&] {
            query.reset(
                next.data(),
                64,
                3,
                scratch.data(),
                hacc_scratch.data(),
                static_cast<MetricType>(7),
                true
            );
        },
        "Metric type must be METRIC_L2 or METRIC_IP"
    );
    expect_invalid(
        [&] { reset(nullptr, 64, 3, scratch.data(), hacc_scratch.data()); },
        "Lut::reset requires query and LUT scratch"
    );
    expect_invalid(
        [&] { reset(next.data(), 64, 3, nullptr, hacc_scratch.data()); },
        "Lut::reset requires query and LUT scratch"
    );
    expect_invalid(
        [&] { reset(next.data(), 64, 3, scratch.data(), nullptr); },
        "Lut::reset requires uint16 scratch for high-accuracy lookup tables"
    );
    for (size_t dim : {0U, 17U}) {
        expect_invalid(
            [&] { reset(next.data(), dim, 3, scratch.data(), hacc_scratch.data()); },
            "FastScan dimension must be a positive multiple of 16"
        );
    }
    EXPECT_THROW(
        reset(
            next.data(),
            std::numeric_limits<size_t>::max() - 15,
            3,
            scratch.data(),
            hacc_scratch.data()
        ),
        std::length_error
    );
    EXPECT_EQ(query.rotated_query(), original.rotated_query());
    EXPECT_EQ(query.padded_dim(), original.padded_dim());
    EXPECT_EQ(query.use_hacc(), original.use_hacc());
    EXPECT_EQ(query.delta(), original.delta());
    EXPECT_EQ(query.sum_vl_lut(), original.sum_vl_lut());
    EXPECT_EQ(query.g_k1xsumq(), original.g_k1xsumq());
    EXPECT_EQ(query.g_kbxsumq(), original.g_kbxsumq());
    EXPECT_EQ(query.g_add(), original.g_add());
    EXPECT_EQ(query.g_error(), original.g_error());
    EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(), query.lut()));
    query.set_g_add(5, 4);
    EXPECT_EQ(query.g_add(), -4);  // Failed reset did not change the metric to L2.
    EXPECT_EQ(query.g_error(), 5);
    query.reset(next.data(), 64, 8, scratch.data(), hacc_scratch.data());
    EXPECT_EQ(query.rotated_query(), next.data());
    EXPECT_EQ(query.g_add(), 0);
}

TEST(FastScanApiTest, SingleBatchEncodersRejectOverflowAndPreserveEmptyOutput) {
    constexpr size_t kDim = 64;
    std::array<float, 33 * kDim> data{};
    std::array<float, kDim> centroid{};
    std::vector<char> batch(BatchDataMap<float>::data_bytes(kDim), 0x5A);
    std::vector<char> qg(QGBatchDataMap<float>::data_bytes(kDim), 0x5A);
    std::vector<char> extra(33 * ExDataMap<float>::data_bytes(kDim, 3), 0x5A);
    const auto original = batch, original_qg = qg, original_extra = extra;
    const auto encode = [&](int api, size_t count) {
        switch (api) {
            case 0:
                quant::quantize_one_batch(
                    data.data(), centroid.data(), count, kDim, batch.data()
                );
                break;
            case 1:
                quant::quantize_one_batch<float, true>(
                    data.data(), count, kDim, batch.data()
                );
                break;
            case 2:
                quant::quantize_qg_batch(
                    data.data(), centroid.data(), count, kDim, qg.data()
                );
                break;
            case 3:
                quant::quantize_qg_batch(data.data(), count, kDim, qg.data());
                break;
            default:
                quant::quantize_split_batch(
                    data.data(), centroid.data(), count, kDim, 3, batch.data(), extra.data()
                );
        }
    };
    for (int api = 0; api < 5; ++api) {
        SCOPED_TRACE(api);
        for (size_t count : {size_t{33}, std::numeric_limits<size_t>::max()}) {
            expect_invalid(
                [&] { encode(api, count); },
                "FastScan single-batch encoding accepts at most 32 vectors"
            );
        }
        encode(api, 0);
        EXPECT_EQ(batch, original);
        EXPECT_EQ(qg, original_qg);
        EXPECT_EQ(extra, original_extra);
    }
}

TEST(FastScanApiTest, PublicSerialAndParallelEncodingMatchIncludingTails) {
    constexpr size_t kDim = 96;
    std::array<float, 32 * kDim> data{};
    std::array<float, kDim> centroid{};
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i));
    }
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (size_t count : {1U, 17U, 32U}) {
            std::vector<char> serial(BatchDataMap<float>::data_bytes(kDim));
            auto parallel = serial, implicit_centroid = serial;
            quant::quantize_one_batch<float, false>(
                data.data(), centroid.data(), count, kDim, serial.data(), metric
            );
            quant::quantize_one_batch<float, true>(
                data.data(), centroid.data(), count, kDim, parallel.data(), metric
            );
            quant::quantize_one_batch<float, true>(
                data.data(), count, kDim, implicit_centroid.data(), metric
            );
            EXPECT_EQ(serial, parallel);
            EXPECT_EQ(serial, implicit_centroid);
        }
    }
}

TEST(FastScanApiTest, DoubleEncodingQueryAndScoringWithNonzeroCentroid) {
    for (size_t dim : {16U, 32U, 96U}) {
        const double dimension = static_cast<double>(dim);
        const double value = 0.5 + 1e-10;
        std::vector<double> data(17 * dim, value), centroid(dim, 0.125), values(dim, 1);
        for (auto metric : {METRIC_L2, METRIC_IP}) {
            std::vector<char> batch(QGBatchDataMap<double>::data_bytes(dim));
            quant::quantize_qg_batch(
                data.data(), centroid.data(), 17, dim, batch.data(), metric
            );
            BatchQuery<double> query(values.data(), dim, metric), reset;
            std::vector<double> scratch(dim * 4);
            reset.reset(values.data(), dim, scratch.data(), metric);
            EXPECT_EQ(reset.delta(), query.delta());
            EXPECT_TRUE(std::equal(query.lut(), query.lut() + dim * 4, reset.lut()));
            const double correction =
                metric == METRIC_L2 ? dimension * 0.875 * 0.875 : 1 - dimension * 0.125;
            query.set_g_add(correction);
            auto copy = query;
            std::array<double, 32> distance{};
            qg_batch_estdist(batch.data(), copy, dim, distance.data());
            const double expected = metric == METRIC_L2
                                        ? dimension * (1 - value) * (1 - value)
                                        : 1 - dimension * value;
            for (size_t row = 0; row < 17; ++row) {
                EXPECT_NEAR(distance[row], expected, 2e-13 * dimension);
            }
            reset.set_g_add(correction);
            reset.reset(values.data(), dim, scratch.data(), metric);
            EXPECT_EQ(reset.g_add(), 0);
            for (bool hacc : {false, true}) {
                SplitBatchQuery<double> split(values.data(), dim, 3, metric, hacc);
                EXPECT_EQ(split.padded_dim(), dim);
                EXPECT_EQ(split.use_hacc(), hacc);
                EXPECT_DOUBLE_EQ(split.delta(), 4.0 / (hacc ? 65535 : 255));
                Lut<double> reused;
                std::vector<uint16_t> hacc_scratch(dim * 4);
                reused.reset(values.data(), dim, scratch.data(), hacc_scratch.data(), hacc);
                const size_t bytes = dim * 4 * (hacc ? 2 : 1);
                EXPECT_TRUE(std::equal(split.lut(), split.lut() + bytes, reused.lut()));
                SplitBatchQuery<double> reset_split;
                reset_split.reset(
                    values.data(), dim, 3, scratch.data(), hacc_scratch.data(), metric, hacc
                );
                EXPECT_EQ(reset_split.delta(), split.delta());
                EXPECT_EQ(reset_split.g_kbxsumq(), split.g_kbxsumq());
                EXPECT_TRUE(
                    std::equal(split.lut(), split.lut() + bytes, reset_split.lut())
                );
            }
        }
    }
}

TEST(FastScanApiTest, DoubleLutMatchesIndependentSubsetSumQuantization) {
    constexpr size_t kDim = 96;
    std::array<double, kDim> query{};
    for (size_t i = 0; i < kDim; ++i) {
        // Avoid exact half-bin ties: the independent division below and the
        // production reciprocal multiplication can round on opposite sides.
        query[i] = std::sin(static_cast<double>(i) * 0.37) + 1e-10 * static_cast<double>(i);
    }
    std::array<double, kDim * 4> reference{};
    for (size_t group = 0; group < kDim / 4; ++group) {
        for (size_t mask = 0; mask < 16; ++mask) {
            for (size_t d = 0; d < 4; ++d) {
                if ((mask & (8U >> d)) != 0) {
                    reference[group * 16 + mask] += query[group * 4 + d];
                }
            }
        }
    }
    const auto range = std::minmax_element(reference.begin(), reference.end());
    std::vector<uint8_t> compact(32 * kDim / 8, 0x5A), packed(compact.size());
    fastscan::pack_codes(kDim, compact.data(), 32, packed.data());
    for (bool hacc : {false, true}) {
        const Lut<double> lut(query.data(), kDim, hacc);
        const double delta = (*range.second - *range.first) / (hacc ? 65535 : 255);
        EXPECT_DOUBLE_EQ(lut.delta(), delta);
        EXPECT_DOUBLE_EQ(lut.sum_vl(), *range.first * (kDim / 4));
        int32_t expected = 0;
        for (size_t group = 0; group < kDim / 4; ++group) {
            const size_t mask = group % 2 == 0 ? 5 : 10;
            expected += static_cast<int32_t>(
                std::round((reference[group * 16 + mask] - *range.first) / delta)
            );
        }
        std::array<int32_t, 32> accumulated{};
        if (hacc) {
            fastscan::accumulate_hacc(packed.data(), lut.lut(), accumulated.data(), kDim);
        } else {
            fastscan::accumulate(packed.data(), lut.lut(), accumulated.data(), kDim);
        }
        for (auto value : accumulated) {
            EXPECT_EQ(value, expected);
        }
    }
    query.fill(0);
    for (bool hacc : {false, true}) {
        const Lut<double> zero(query.data(), kDim, hacc);
        EXPECT_EQ(zero.delta(), 0);
        EXPECT_EQ(zero.sum_vl(), 0);
        EXPECT_TRUE(
            std::all_of(
                zero.lut(), zero.lut() + kDim * 4 * (hacc ? 2 : 1), [](uint8_t value) {
                    return value == 0;
                }
            )
        );
    }
}

}  // namespace
}  // namespace rabitqlib
