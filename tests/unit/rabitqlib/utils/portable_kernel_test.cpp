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
        for (size_t dim : {64U, 128U, 192U, 576U}) {
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
            EXPECT_EQ(ip[bits - 1](query.data(), dispatched.data(), dim), expected);
            EXPECT_EQ(
                select_excode_ipfunc(bits)(query.data(), dispatched.data(), dim), expected
            );
        }
    }
}

TEST(PortableKernels, QueryTransposeAndWarmupMatchIntegerReference) {
    for (size_t dim : {64U, 128U, 448U, 512U, 576U, 960U, 1024U, 1088U}) {
        for (size_t bits = 0; bits <= 8; ++bits) {
            SCOPED_TRACE(dim);
            SCOPED_TRACE(bits);
            std::vector<uint8_t> query(dim);
            std::vector<uint64_t> data(dim / 64), transposed(dim / 64 * bits),
                scalar(transposed.size());
            uint64_t expected_ip = 0, expected_count = 0;
            for (size_t i = 0; i < dim; ++i) {
                query[i] = static_cast<uint8_t>((i * 19 + 7) & ((1U << bits) - 1));
                if (i % 3 == 0 || i % 63 == 0) {
                    data[i / 64] |= uint64_t{1} << (63 - i % 64);
                    expected_ip += query[i];
                    ++expected_count;
                }
            }
            new_transpose_bin_512(query.data(), transposed.data(), dim, bits);
            simd::new_transpose_bin_512_generic(query.data(), scalar.data(), dim, bits);
            EXPECT_EQ(scalar, transposed);
            size_t offset = 0;
            for (size_t block = 0; block < dim; block += 512) {
                const size_t chunks = std::min(size_t{512}, dim - block) / 64;
                for (size_t b = 0; b < bits; ++b) {
                    for (size_t c = 0; c < chunks; ++c) {
                        for (size_t i = 0; i < 64; ++i) {
                            EXPECT_EQ(
                                (transposed[offset + b * chunks + c] >> (63 - i)) & 1U,
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
