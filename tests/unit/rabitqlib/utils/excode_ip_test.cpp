#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rabitqlib/simd/pack_excode_dispatch.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/space.hpp"

TEST(ExcodeIp, BackendsMatchScalarAcrossWidthsAndBlockBoundaries) {
    using namespace rabitqlib;
    if (!cpu::has_avx2()) {
        GTEST_SKIP() << "Packed-code tests require AVX2/FMA";
    }

    const std::array<ex_ipfunc, 8> avx2_functions{
        simd::excode_ipimpl::ip16_fxu1_avx2,
        simd::excode_ipimpl::ip64_fxu2_avx2,
        simd::excode_ipimpl::ip64_fxu3_avx2,
        simd::excode_ipimpl::ip16_fxu4_avx2,
        simd::excode_ipimpl::ip64_fxu5_avx2,
        simd::excode_ipimpl::ip64_fxu6_avx2,
        simd::excode_ipimpl::ip64_fxu7_avx2,
        simd::excode_ipimpl::ip16_fxu8_avx2,
    };
    const std::array<ex_ipfunc, 8> avx512_functions{
        simd::excode_ipimpl::ip16_fxu1_avx512,
        simd::excode_ipimpl::ip64_fxu2_avx512,
        simd::excode_ipimpl::ip64_fxu3_avx512,
        simd::excode_ipimpl::ip16_fxu4_avx512,
        simd::excode_ipimpl::ip64_fxu5_avx512,
        simd::excode_ipimpl::ip64_fxu6_avx512,
        simd::excode_ipimpl::ip64_fxu7_avx512,
        simd::excode_ipimpl::ip16_fxu8_avx512,
    };
    using PackFunction = void (*)(const uint8_t*, uint8_t*, size_t);
    const std::array<PackFunction, 6> pack_functions{
        simd::packing_2bit_excode_avx2,
        simd::packing_3bit_excode_avx2,
        simd::packing_4bit_excode_avx2,
        simd::packing_5bit_excode_avx2,
        simd::packing_6bit_excode_avx2,
        simd::packing_7bit_excode_avx2,
    };

    for (size_t bits = 1; bits <= 8; ++bits) {
        SCOPED_TRACE(bits);
        const size_t block_dim = (bits == 1 || bits == 4 || bits == 8) ? 16 : 64;
        const auto max_code = static_cast<uint8_t>((1U << bits) - 1);
        for (size_t dim : std::array<size_t, 15>{
                 0, 16, 32, 48, 64, 80, 96, 112, 128, 192, 256, 576, 960, 1024, 4096}) {
            if (dim % block_dim != 0) {
                continue;
            }
            SCOPED_TRACE(dim);
            for (size_t pattern = 0; pattern < 4; ++pattern) {
                SCOPED_TRACE(pattern);
                std::vector<float> query_storage(dim + 1);
                auto* query = query_storage.data() + 1;
                std::vector<uint8_t> codes(dim);
                std::vector<uint8_t> storage((dim * bits / 8) + 1, 0);
                auto* compact = storage.data() + 1;
                double expected = 0;
                double sum_abs = 0;
                for (size_t i = 0; i < dim; ++i) {
                    query[i] =
                        pattern == 0
                            ? 0.0F
                            : static_cast<float>(static_cast<int>(i % 23) - 11) / 7.0F;
                    if (pattern == 3) {
                        query[i] = (i % 2 == 0) ? 1.0F : -1.0F;
                    }
                    codes[i] = pattern >= 2
                                   ? max_code
                                   : static_cast<uint8_t>((i * 37U + 19U) & max_code);
                    const double product = static_cast<double>(query[i]) * codes[i];
                    expected += product;
                    sum_abs += std::abs(product);
                }
                if (bits == 1) {
                    for (size_t i = 0; i < dim; ++i) {
                        compact[i / 8] |= static_cast<uint8_t>(codes[i] << (i % 8));
                    }
                } else if (bits == 8) {
                    std::copy(codes.begin(), codes.end(), compact);
                } else if (dim != 0) {
                    pack_functions[bits - 2](codes.data(), compact, dim);
                }

                // Scale by product magnitudes, since cancellation can make the dot
                // product itself near zero. Different reduction trees may round
                // differently.
                const double tolerance = 2e-6 * std::max(1.0, sum_abs);
                EXPECT_NEAR(
                    avx2_functions[bits - 1](query, compact, dim), expected, tolerance
                );
                if (cpu::has_avx512_core()) {
                    EXPECT_NEAR(
                        avx512_functions[bits - 1](query, compact, dim), expected, tolerance
                    );
                }
                EXPECT_NEAR(
                    select_excode_ipfunc(bits)(query, compact, dim), expected, tolerance
                );
            }
        }
    }
}
