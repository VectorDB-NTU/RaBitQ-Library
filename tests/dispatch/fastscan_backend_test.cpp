#include "fastscan_backend.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/detail/estimator_dispatch.hpp"
#include "rabitqlib/fastscan/detail/simd.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#include "rabitqlib/fastscan/query.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

namespace rabitqlib::fastscan {
namespace {

TEST(FastScanBackendContract, SelectedHaccLayoutMatchesProducerAndConsumer) {
    const auto& backend = detail::selected_backend();
    const size_t tables_per_plane =
        backend.hacc_layout == detail::HaccLutLayout::Quad64 ? 4 : 2;
    EXPECT_EQ(&backend, &detail::selected_backend());
    for (size_t dim : {16U, 32U, 64U, 128U, 1024U, 1040U}) {
        SCOPED_TRACE(::testing::Message() << backend.name << " dim=" << dim);
        std::vector<uint16_t> lut(dim * 4);
        std::vector<uint8_t> bytes(dim * 8 + 2, 0xAB);
        std::vector<uint8_t> expected = bytes;
        for (size_t i = 0; i < lut.size(); ++i) {
            lut[i] = static_cast<uint16_t>(i * 911 + i / 16 * 37);
            const size_t table = i / 16;
            const size_t low = table / tables_per_plane * tables_per_plane * 32 +
                               table % tables_per_plane * 16 + i % 16;
            expected[1 + low] = static_cast<uint8_t>(lut[i]);
            expected[1 + low + tables_per_plane * 16] = static_cast<uint8_t>(lut[i] >> 8);
        }
        transfer_lut_hacc(lut.data(), dim, bytes.data() + 1);
        EXPECT_EQ(bytes, expected);
        backend.transfer_hacc(lut.data(), dim, bytes.data() + 1);
        EXPECT_EQ(bytes, expected);

        // Construct codes independently of pack_codes and its lane permutation.
        std::vector<uint8_t> codes(dim * 4);
        std::array<int32_t, 32> reference{};
        for (size_t group = 0; group < dim / 4; ++group) {
            for (size_t row = 0; row < 32; ++row) {
                const size_t entry = (group * 3 + row * 7) % 16;
                const size_t lane = 2 * (row % 8) + (row % 16) / 8;
                codes[group * 16 + lane] |= static_cast<uint8_t>(entry << (row / 16 * 4));
                reference[row] += lut[group * 16 + entry];
            }
        }
        std::array<int32_t, 32> result{};
        accumulate_hacc(codes.data(), bytes.data() + 1, result.data(), dim);
        EXPECT_EQ(result, reference);
        backend.accumulate_hacc(codes.data(), bytes.data() + 1, result.data(), dim);
        EXPECT_EQ(result, reference);
    }
}

TEST(FastScanBackendContract, StandardLutIsSharedAcrossSupportedConsumers) {
    constexpr size_t kDim = 64;
    const auto& backend = detail::selected_backend();
    std::array<float, kDim> query{};
    for (size_t i = 0; i < kDim; ++i) {
        query[i] = static_cast<float>(i % 7);
    }
    std::array<float, kDim * 4> packed{}, expected{};
    backend.pack_lut(kDim, query.data(), packed.data());
    simd::pack_lut_generic(kDim, query.data(), expected.data());
    EXPECT_EQ(packed, expected);
    std::array<uint8_t, kDim * 4> lut{}, codes{};
    for (size_t i = 0; i < lut.size(); ++i) {
        lut[i] = static_cast<uint8_t>(packed[i]);
        codes[i] = static_cast<uint8_t>(i * 73 + i / 7);
    }
    std::array<int32_t, 32> reference{}, result{};
    simd::accumulate_generic(codes.data(), lut.data(), reference.data(), kDim);
    if (cpu::has_avx2() || cpu::has_neon()) {
        backend.accumulate(codes.data(), lut.data(), result.data(), kDim);
        EXPECT_EQ(result, reference);
    }
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        simd::accumulate_avx2(codes.data(), lut.data(), result.data(), kDim);
        EXPECT_EQ(result, reference);
    }
    if (cpu::has_avx512_core()) {
        simd::accumulate_avx512(codes.data(), lut.data(), result.data(), kDim);
        EXPECT_EQ(result, reference);
    }
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    if (cpu::has_neon()) {
        simd::accumulate_neon(codes.data(), lut.data(), result.data(), kDim);
        EXPECT_EQ(result, reference);
    }
#endif
}

TEST(FastScanBackendContract, ExplicitCorrectionBackendsConsumeHostLut) {
    const auto& selected = detail::selected_backend();
    using Estimator = decltype(selected.split);
    std::vector<Estimator> backends{selected.split};
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2())
        backends.push_back(rabitqlib::simd::split_batch_estdist_avx2);
    if (cpu::has_avx512_core())
        backends.push_back(rabitqlib::simd::split_batch_estdist_avx512);
#endif
    for (size_t dim : {32U, 96U, 1040U}) {
        std::vector<float> values(dim);
        for (size_t i = 0; i < dim; ++i)
            values[i] = static_cast<float>(i % 23) / 7 - 1;
        for (bool hacc : {false, true}) {
            if (!hacc && !cpu::has_avx2() && !cpu::has_neon())
                continue;
            for (auto metric : {METRIC_L2, METRIC_IP}) {
                SCOPED_TRACE(
                    ::testing::Message()
                    << selected.name << " " << dim << " " << hacc << " " << metric
                );
                SplitBatchQuery<float> query(values.data(), dim, 3, metric, hacc);
                query.set_g_add(3, 2);
                std::vector<char> storage(BatchDataMap<float>::data_bytes(dim));
                BatchDataMap<float> batch(storage.data(), dim);
                for (size_t i = 0; i < dim * 4; ++i)
                    batch.bin_code()[i] = static_cast<uint8_t>(i * 73 + i / 7);
                for (size_t lane = 0; lane < 32; ++lane) {
                    batch.f_add()[lane] = static_cast<float>(lane) + 10;
                    batch.f_rescale()[lane] = lane % 2 == 0 ? 0.125F : -0.25F;
                    batch.f_error()[lane] = static_cast<float>(lane) / 32;
                }
                std::array<float, 32> reference_distance{}, reference_lower{},
                    reference_ip{};
                rabitqlib::simd::split_batch_estdist_generic(
                    storage.data(),
                    query,
                    dim,
                    reference_distance.data(),
                    reference_lower.data(),
                    reference_ip.data(),
                    hacc
                );
                for (auto score : backends) {
                    std::array<float, 32> distance{}, lower{}, ip{};
                    score(
                        storage.data(),
                        query,
                        dim,
                        distance.data(),
                        lower.data(),
                        ip.data(),
                        hacc
                    );
                    for (size_t lane = 0; lane < 32; ++lane) {
                        const double tolerance = 2e-6 * static_cast<double>(dim);
                        EXPECT_NEAR(ip[lane], reference_ip[lane], tolerance);
                        EXPECT_NEAR(distance[lane], reference_distance[lane], tolerance);
                        EXPECT_NEAR(lower[lane], reference_lower[lane], tolerance);
                    }
                }
            }
        }
    }
}

}  // namespace
}  // namespace rabitqlib::fastscan
