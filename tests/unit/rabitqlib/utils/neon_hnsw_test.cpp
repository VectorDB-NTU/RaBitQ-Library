#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/simd/hnsw_dispatch.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/simd/warmup_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

#if defined(__aarch64__) || defined(_M_ARM64)
namespace rabitqlib {
TEST(NeonHnsw, WarmupHandlesUnalignedCodesTailsAndWideCounts) {
    if (!cpu::has_neon()) {
        GTEST_SKIP();
    }
    std::mt19937 random(764);
    for (size_t dim : {0U, 64U, 128U, 448U, 512U, 576U, 960U, 1024U, 1088U, 65536U}) {
        for (size_t bits = 0; bits <= 8; ++bits) {
            for (bool all_ones : {false, true}) {
                SCOPED_TRACE(dim);
                SCOPED_TRACE(bits);
                SCOPED_TRACE(all_ones);
                std::vector<uint8_t> query(dim), storage(dim / 8 + 1);
                std::vector<uint64_t> words(dim / 64), transposed(dim / 64 * bits);
                uint64_t ip = 0, count = 0;
                for (size_t i = 0; i < dim; ++i) {
                    query[i] = static_cast<uint8_t>(
                        (all_ones ? 255 : random()) & ((1U << bits) - 1)
                    );
                    if (all_ones || random() % 2 != 0) {
                        words[i / 64] |= uint64_t{1} << (63 - i % 64);
                        ip += query[i];
                        ++count;
                    }
                }
                if (dim != 0) {
                    std::memcpy(storage.data() + 1, words.data(), dim / 8);
                }
                simd::new_transpose_bin_512_generic(
                    query.data(), transposed.data(), dim, bits
                );
                const float expected =
                    0.25F * static_cast<float>(ip) - 2.0F * static_cast<float>(count);
                EXPECT_EQ(
                    simd::warmup_ip_x0_q_512_neon(
                        storage.data() + 1, transposed.data(), 0.25F, -2.0F, dim, bits
                    ),
                    expected
                );
                EXPECT_EQ(
                    simd::warmup_ip_x0_q_512_neon(
                        words.data(), transposed.data(), 0.25F, -2.0F, dim, bits
                    ),
                    expected
                );
            }
        }
    }
    EXPECT_THROW(
        simd::warmup_ip_x0_q_512_neon(
            static_cast<const uint8_t*>(nullptr), nullptr, 1, 0, 0, 9
        ),
        std::invalid_argument
    );
}

TEST(NeonHnsw, SearchMatchesGenericOnSameGraphForBothMetricsAndAllWidths) {
    if (!cpu::has_neon()) {
        GTEST_SKIP();
    }
    constexpr size_t kCount = 64;
    for (size_t dim : {64U, 576U, 960U}) {
        std::mt19937 random(42);
        std::uniform_real_distribution<float> distribution(-1, 1);
        std::vector<float> data(kCount * dim), centroid(dim, 0), query(dim);
        for (auto& value : data) {
            value = distribution(random);
        }
        for (auto metric : {METRIC_L2, METRIC_IP}) {
            for (size_t bits = 1; bits <= 9; ++bits) {
                SCOPED_TRACE(dim);
                SCOPED_TRACE(metric);
                SCOPED_TRACE(bits);
                std::vector<PID> clusters(kCount, 0);
                hnsw::HierarchicalNSW index(kCount, dim, bits, 8, 64, 42, metric);
                index.construct(
                    1, centroid.data(), kCount, data.data(), clusters.data(), 1, true
                );
                // The direct entry points accept rotated coordinates. Generate queries
                // in that domain and pass the exact same values to both backends.
                for (size_t q = 0; q < 3; ++q) {
                    for (auto& value : query) {
                        value = distribution(random);
                    }
                    for (size_t ef : {10U, 64U}) {
                        // Configure ef through the public search API.
                        index.search(query.data(), 1, 5, ef, 1);
                        auto expected =
                            hnsw::detail::search_knn_generic(index, query.data(), 5);
                        auto actual = hnsw::detail::search_knn_neon(index, query.data(), 5);
                        ASSERT_EQ(expected.size(), actual.size());
                        while (!expected.empty()) {
                            EXPECT_EQ(actual.top().second, expected.top().second);
                            EXPECT_NEAR(
                                actual.top().first,
                                expected.top().first,
                                2e-5F * std::max(1.0F, std::abs(expected.top().first))
                            );
                            actual.pop();
                            expected.pop();
                        }
                    }
                }
            }
        }
    }
}
}  // namespace rabitqlib
#endif
