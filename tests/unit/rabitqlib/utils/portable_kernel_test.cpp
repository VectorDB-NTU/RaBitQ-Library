#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "rabitqlib/quantization/pack_excode.hpp"
#include "rabitqlib/simd/pack_excode_dispatch.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/simd/warmup_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/warmup_space.hpp"

namespace rabitqlib {
TEST(PortableKernels, PackingMatchesDispatchedBytesAndDotProducts) {
    using Pack = void (*)(const uint8_t*, uint8_t*, size_t);
    const std::array<Pack, 6> pack{
        simd::packing_2bit_excode_generic,
        simd::packing_3bit_excode_generic,
        simd::packing_4bit_excode_generic,
        simd::packing_5bit_excode_generic,
        simd::packing_6bit_excode_generic,
        simd::packing_7bit_excode_generic};
    const std::array<ex_ipfunc, 8> ip{
        simd::excode_ipimpl::ip16_fxu1_generic,
        simd::excode_ipimpl::ip64_fxu2_generic,
        simd::excode_ipimpl::ip64_fxu3_generic,
        simd::excode_ipimpl::ip16_fxu4_generic,
        simd::excode_ipimpl::ip64_fxu5_generic,
        simd::excode_ipimpl::ip64_fxu6_generic,
        simd::excode_ipimpl::ip64_fxu7_generic,
        simd::excode_ipimpl::ip16_fxu8_generic};
    for (size_t bits = 1; bits <= 8; ++bits) {
        for (size_t dim : {64U, 96U, 128U, 192U, 544U, 576U}) {
            SCOPED_TRACE(bits);
            SCOPED_TRACE(dim);
            std::vector<uint8_t> raw(dim), dispatched(dim * bits / 8),
                scalar(dispatched.size());
            std::vector<float> query(dim);
            double expected = 0;
            for (size_t i = 0; i < dim; ++i) {
                raw[i] = static_cast<uint8_t>((i * 37 + i / 7) & ((1U << bits) - 1));
                query[i] = static_cast<float>(static_cast<int>(i % 17) - 8);
                expected += query[i] * raw[i];
            }
            quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
                raw.data(), dispatched.data(), dim, bits
            );
            if (bits >= 2 && bits <= 7) {
                pack[bits - 2](raw.data(), scalar.data(), dim);
                EXPECT_EQ(scalar, dispatched);
            }

            // The inverse has to return exactly what was packed, whichever
            // backend did the packing.
            std::vector<uint8_t> roundtrip(dim, 0xFF);
            quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(
                dispatched.data(), roundtrip.data(), dim, bits
            );
            EXPECT_EQ(roundtrip, raw);
            EXPECT_EQ(ip[bits - 1](query.data(), dispatched.data(), dim), expected);
            EXPECT_EQ(
                select_excode_ipfunc(bits)(query.data(), dispatched.data(), dim), expected
            );
        }
    }
}

TEST(PortableKernels, QueryTransposeAndWarmupMatchIntegerReference) {
    for (size_t dim : {64U, 96U, 128U, 448U, 512U, 544U, 576U, 960U, 1024U, 1088U}) {
        for (size_t bits = 0; bits <= 8; ++bits) {
            SCOPED_TRACE(dim);
            SCOPED_TRACE(bits);
            std::vector<uint8_t> query(dim);
            std::vector<uint64_t> data((dim + 63) / 64), transposed((dim + 63) / 64 * bits),
                scalar(transposed.size());
            uint64_t expected_ip = 0, expected_count = 0;
            for (size_t i = 0; i < dim; ++i) {
                query[i] = static_cast<uint8_t>((i * 19 + 7) & ((1U << bits) - 1));
                if (i % 3 == 0 || i % 63 == 0) {
                    data[i / 64] |=
                        uint64_t{1}
                        << (std::min(size_t{64}, dim - i / 64 * 64) - 1 - i % 64);
                    expected_ip += query[i];
                    ++expected_count;
                }
            }
            new_transpose_bin_512(query.data(), transposed.data(), dim, bits);
            simd::new_transpose_bin_512_generic(query.data(), scalar.data(), dim, bits);
            EXPECT_EQ(scalar, transposed);
            size_t offset = 0;
            for (size_t block = 0; block < dim; block += 512) {
                const size_t chunks = (std::min(size_t{512}, dim - block) + 63) / 64;
                for (size_t b = 0; b < bits; ++b) {
                    for (size_t c = 0; c < chunks; ++c) {
                        const size_t width = std::min(size_t{64}, dim - block - c * 64);
                        for (size_t i = 0; i < width; ++i) {
                            EXPECT_EQ(
                                (transposed[offset + b * chunks + c] >> (width - 1 - i)) &
                                    1U,
                                (query[block + c * 64 + i] >> b) & 1U
                            );
                        }
                    }
                }
                offset += chunks * bits;
            }
            const float expected = 0.25F * expected_ip - 2.0F * expected_count;
            EXPECT_EQ(
                simd::warmup_ip_x0_q_512_generic(
                    data.data(), scalar.data(), 0.25F, -2.0F, dim, bits
                ),
                expected
            );
            EXPECT_EQ(
                warmup_ip_x0_q_512(data.data(), transposed.data(), 0.25F, -2.0F, dim, bits),
                expected
            );
        }
    }
    EXPECT_THROW(
        simd::warmup_ip_x0_q_512_generic(
            static_cast<const uint8_t*>(nullptr), nullptr, 1, 0, 0, 9
        ),
        std::invalid_argument
    );
}

TEST(PortableKernels, Uint16TransposePreservesAllBitPlanes) {
    constexpr size_t kDim = 192;
    std::array<uint16_t, kDim> query{};
    for (size_t i = 0; i < kDim; ++i)
        query[i] = static_cast<uint16_t>(i * 977);
    for (size_t bits = 1; bits <= 16; ++bits) {
        std::vector<uint64_t> actual(kDim / 64 * bits), scalar(actual.size());
        new_transpose_bin(query.data(), actual.data(), kDim, bits);
        simd::new_transpose_bin_generic(query.data(), scalar.data(), kDim, bits);
        EXPECT_EQ(actual, scalar);
        for (size_t i = 0; i < kDim; ++i) {
            for (size_t b = 0; b < bits; ++b) {
                EXPECT_EQ(
                    (actual[i / 64 * bits + b] >> (63 - i % 64)) & 1U, (query[i] >> b) & 1U
                );
            }
        }
    }
}
}  // namespace rabitqlib

TEST(PortableKernels, ExtraBitTailsPreserveFullBlockBytesAndUseCompactBitStream) {
    using namespace rabitqlib;
    for (size_t bits : {2U, 3U, 5U, 6U, 7U}) {
        constexpr size_t tail = 32;
        const size_t dim = 128 + tail;
        std::vector<uint8_t> raw(dim), packed(dim * bits / 8 + 2, 0xA5);
        std::vector<uint8_t> full(128 * bits / 8);
        for (size_t i = 0; i < dim; ++i)
            raw[i] = (i * 37 + 3) & ((1U << bits) - 1);
        quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
            raw.data(), full.data(), 128, bits
        );
        quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
            raw.data(), packed.data() + 1, dim, bits
        );
        EXPECT_EQ(packed.front(), 0xA5);
        EXPECT_EQ(packed.back(), 0xA5);
        EXPECT_TRUE(std::equal(full.begin(), full.end(), packed.begin() + 1));
        // Independent byte-by-byte definition: bit j holds bit j % bits
        // of coordinate j / bits, with no gaps or rounded word storage.
        for (size_t byte = 0; byte < tail * bits / 8; ++byte) {
            uint8_t expected = 0;
            for (size_t b = 0; b < 8; ++b) {
                const size_t pos = byte * 8 + b;
                expected |= ((raw[128 + pos / bits] >> (pos % bits)) & 1U) << b;
            }
            EXPECT_EQ(packed[1 + full.size() + byte], expected);
        }
    }
}

TEST(PortableKernels, PackingMatchesReferenceForMaximumAndIsolatedCodes) {
    using namespace rabitqlib;
    using Pack = void (*)(const uint8_t*, uint8_t*, size_t);
    const std::array<Pack, 6> generic{
        simd::packing_2bit_excode_generic,
        simd::packing_3bit_excode_generic,
        simd::packing_4bit_excode_generic,
        simd::packing_5bit_excode_generic,
        simd::packing_6bit_excode_generic,
        simd::packing_7bit_excode_generic};
    std::vector<std::array<Pack, 6>> backends{generic};
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::has_avx2()) {
        backends.push_back({
            simd::packing_2bit_excode_avx2,
            simd::packing_3bit_excode_avx2,
            simd::packing_4bit_excode_avx2,
            simd::packing_5bit_excode_avx2,
            simd::packing_6bit_excode_avx2,
            simd::packing_7bit_excode_avx2,
        });
    }
    if (cpu::has_avx512_core()) {
        backends.push_back({
            simd::packing_2bit_excode_avx512,
            simd::packing_3bit_excode_avx512,
            simd::packing_4bit_excode_avx512,
            simd::packing_5bit_excode_avx512,
            simd::packing_6bit_excode_avx512,
            simd::packing_7bit_excode_avx512,
        });
    }
#endif
    const std::array<size_t, 6> widths{2, 3, 4, 5, 6, 7};
    for (size_t b = 0; b < widths.size(); ++b) {
        const size_t bits = widths[b];
        const auto max_code = static_cast<uint8_t>((1U << bits) - 1);
        for (size_t dim : {64U, 96U, 128U, 256U}) {
            for (size_t pattern = 0; pattern <= dim; ++pattern) {
                SCOPED_TRACE(bits);
                SCOPED_TRACE(dim);
                SCOPED_TRACE(pattern);
                std::vector<uint8_t> raw(dim + 1);
                for (size_t i = 0; i < dim; ++i) {
                    raw[i + 1] = pattern == dim || i == pattern ? max_code : 0;
                }
                std::vector<uint8_t> expected(dim * bits / 8 + 2, 0xA5);
                generic[b](raw.data() + 1, expected.data() + 1, dim);
                for (const auto& backend : backends) {
                    std::vector<uint8_t> actual(expected.size(), 0xA5);
                    backend[b](raw.data() + 1, actual.data() + 1, dim);
                    EXPECT_EQ(actual, expected);
                }
            }
        }
    }
}
