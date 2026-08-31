#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/quantization/rabitq_impl.hpp"
#include "rabitqlib/utils/space.hpp"
#include "test_helpers.hpp"

using namespace rabitqlib;
using namespace rabitq_test;

namespace {

std::vector<float> RandomVec(size_t dim, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> v(dim);
    for (auto& x : v) {
        x = dist(gen);
    }
    return v;
}

// The function writes packed codes into the data blocks and no unpacker exists.
// Recover the raw per-dimension codes through the library's own reader instead
// of reimplementing seven SIMD bit layouts: a one-hot query makes the ex-code
// inner product return exactly code[j]. This asserts the codes as the search
// path actually interprets them, which is the property that matters.
std::vector<uint8_t> UnpackViaIpKernel(const uint8_t* packed, size_t dim, size_t bits) {
    std::vector<uint8_t> out(dim, 0);
    if (bits == 0) {
        return out;
    }
    auto ip = select_excode_ipfunc(bits);
    std::vector<float> probe(dim, 0.0F);
    for (size_t j = 0; j < dim; ++j) {
        probe[j] = 1.0F;
        out[j] = static_cast<uint8_t>(std::lround(ip(probe.data(), packed, dim)));
        probe[j] = 0.0F;
    }
    return out;
}

// Convenience wrapper: split_code_with_factor with both factor triplets.
struct XySplitResult {
    std::vector<uint8_t> base_code;
    std::vector<uint8_t> extra_code;
    float f_add_base = 0;
    float f_rescale_base = 0;
    float f_error_base = 0;
    float f_add_full = 0;
    float f_rescale_full = 0;
};

XySplitResult SplitCode(
    const std::vector<float>& data,
    const std::vector<float>& centroid,
    size_t dim,
    size_t base_bits,
    size_t extra_bits
) {
    XySplitResult res;
    res.base_code.resize(dim);
    res.extra_code.resize(dim);

    std::vector<char> base_block(BaseDataMap<float>::data_bytes(dim, base_bits), 0);
    std::vector<char> extra_block(
        extra_bits > 0 ? ExDataMap<float>::data_bytes(dim, extra_bits) : 1, 0
    );

    // Through the public entry point: rabitq_impl now takes raw buffers, and
    // rabitq.hpp is what maps them onto BaseDataMap/ExDataMap.
    quant::quantize_xy_single(
        data.data(),
        centroid.data(),
        dim,
        base_bits,
        extra_bits,
        base_block.data(),
        extra_bits > 0 ? extra_block.data() : nullptr,
        METRIC_L2
    );

    BaseDataMap<float> base_map(base_block.data(), dim, base_bits);
    res.f_add_base = base_map.f_add();
    res.f_rescale_base = base_map.f_rescale();
    res.f_error_base = base_map.f_error();
    res.base_code = UnpackViaIpKernel(base_map.base_code(), dim, base_bits);

    if (extra_bits > 0) {
        ExDataMap<float> extra_map(extra_block.data(), dim, extra_bits);
        res.f_add_full = extra_map.f_add_ex();
        res.f_rescale_full = extra_map.f_rescale_ex();
        res.extra_code = UnpackViaIpKernel(extra_map.ex_code(), dim, extra_bits);
    } else {
        res.f_add_full = res.f_add_base;
        res.f_rescale_full = res.f_rescale_base;
    }
    return res;
}

}  // namespace

// Backward-compat anchor: at base_bits == 1, split_code_with_factor must
// match today's one_bit_code_with_factor + ex_bits_code_with_factor -- codes
// *and* both factor triplets. The base layer's factors are what BinDataMap
// holds in the 1+y path, the full layer's are what ExDataMap holds.
TEST(XyQuantization, BaseBitsOneMatchesExistingOneBitPlusExBits) {
    constexpr size_t kDim = 64;
    constexpr size_t kExtraBits = 3;

    std::mt19937 gen(42);
    std::vector<float> data = RandomVec(kDim, gen);
    std::vector<float> centroid(kDim, 0.0F);

    // Reference: today's separate one_bit_code_with_factor (sign bit + the
    // 1-bit factors) + ex_bits_code_with_factor (magnitude + full factors).
    std::vector<int> ref_binary_code(kDim);
    float ref_f_add_bin = 0;
    float ref_f_rescale_bin = 0;
    float ref_f_error_bin = 0;
    quant::rabitq_impl::one_bit::one_bit_code_with_factor(
        data.data(),
        centroid.data(),
        kDim,
        ref_binary_code.data(),
        ref_f_add_bin,
        ref_f_rescale_bin,
        ref_f_error_bin,
        METRIC_L2
    );

    std::vector<uint8_t> ref_ex_code(kDim);
    float ref_f_add = 0;
    float ref_f_rescale = 0;
    float ref_f_error = 0;
    quant::rabitq_impl::ex_bits::ex_bits_code_with_factor<float, uint8_t>(
        data.data(),
        centroid.data(),
        kDim,
        kExtraBits,
        ref_ex_code.data(),
        ref_f_add,
        ref_f_rescale,
        ref_f_error,
        METRIC_L2
    );

    // New path: base_bits=1, extra_bits=kExtraBits.
    XySplitResult res = SplitCode(data, centroid, kDim, /*base_bits=*/1, kExtraBits);

    for (size_t i = 0; i < kDim; ++i) {
        EXPECT_EQ(res.base_code[i], static_cast<uint8_t>(ref_binary_code[i]))
            << "dim " << i;
        EXPECT_EQ(res.extra_code[i], ref_ex_code[i]) << "dim " << i;
    }

    EXPECT_FLOAT_NEARLY_EQUAL(res.f_add_base, ref_f_add_bin, 1e-4F);
    EXPECT_FLOAT_NEARLY_EQUAL(res.f_rescale_base, ref_f_rescale_bin, 1e-4F);
    EXPECT_FLOAT_NEARLY_EQUAL(res.f_error_base, ref_f_error_bin, 1e-4F);

    EXPECT_FLOAT_NEARLY_EQUAL(res.f_add_full, ref_f_add, 1e-4F);
    EXPECT_FLOAT_NEARLY_EQUAL(res.f_rescale_full, ref_f_rescale, 1e-4F);
    // The combined code's f_error is not exposed by the function, so it is not
    // asserted here; ref_f_error stays as documentation of the reference value.
}

// ip(base)*2^extra_bits + ip(extra) must equal ip(total_code) -- the
// linearity the boosting step relies on.
// The sign bit must be assigned the same way here as in the 1-bit path,
// including for an exactly-zero residual component. one_bit_code() uses
// `residual > 0`, so a zero lands on the 0 side; combined_code() has to agree
// or the same vector gets different sign bits from the two paths.
//
// Random gaussian data never produces an exact zero, so the other tests here
// cannot catch a divergence -- this one constructs the case directly, by
// putting the centroid exactly on the data value in half the dimensions.
TEST(XyQuantization, SignBitMatchesOneBitPathOnZeroResiduals) {
    constexpr size_t kDim = 64;
    constexpr size_t kExtraBits = 3;

    std::vector<float> centroid(kDim, 1.0F);
    std::vector<float> data(kDim);
    size_t zero_dims = 0;
    for (size_t j = 0; j < kDim; ++j) {
        if (j % 2 == 0) {
            data[j] = 1.0F;  // exact zero residual
            ++zero_dims;
        } else {
            data[j] = 1.0F + ((static_cast<float>(j % 3) - 1.0F) * 0.7F);
        }
    }
    ASSERT_GT(zero_dims, 0U);

    std::vector<int> ref_bin(kDim, -1);
    quant::rabitq_impl::one_bit::one_bit_code(
        data.data(), centroid.data(), kDim, ref_bin.data()
    );

    std::vector<float> residual(kDim);
    for (size_t j = 0; j < kDim; ++j) {
        residual[j] = data[j] - centroid[j];
    }

    std::vector<int> total_code(kDim, -1);
    quant::rabitq_impl::total_bits::combined_code<float, int>(
        residual.data(), kDim, 1 + kExtraBits, total_code.data()
    );

    for (size_t j = 0; j < kDim; ++j) {
        EXPECT_EQ(total_code[j] >> kExtraBits, ref_bin[j]) << "dim " << j;
    }
}

// A point sitting exactly on its centroid is normal (an IVF cluster of one,
// or a duplicate of the centroid). code_factors must return the same finite
// zeros one_bit_code_with_factor and ex_bits_code_with_factor return, not a
// NaN f_error -- a NaN propagates into low_dist and makes every pruning
// comparison false, so such a node is never pruned.
//
// Goes through quantize_xy_single rather than code_factors directly: the
// combined_code-level test above cannot reach this, which is how it was
// missed.
TEST(XyQuantization, ZeroResidualGivesFiniteFactors) {
    constexpr size_t kDim = 64;
    std::vector<float> centroid(kDim, 0.7F);
    std::vector<float> data = centroid;  // residual is exactly zero

    for (size_t base_bits : {1U, 2U, 4U, 8U}) {
        for (size_t ex_bits : {0U, 1U, 3U}) {
            if (base_bits + ex_bits > quant::rabitq_impl::xy_bits::kMaxCombinedBits) {
                continue;
            }
            std::vector<char> base_block(
                BaseDataMap<float>::data_bytes(kDim, base_bits), 0
            );
            std::vector<char> ex_block(
                ex_bits > 0 ? ExDataMap<float>::data_bytes(kDim, ex_bits) : 1, 0
            );
            quant::quantize_xy_single(
                data.data(),
                centroid.data(),
                kDim,
                base_bits,
                ex_bits,
                base_block.data(),
                ex_bits > 0 ? ex_block.data() : nullptr,
                METRIC_L2
            );

            ConstBaseDataMap<float> base_map(base_block.data(), kDim, base_bits);
            EXPECT_TRUE(std::isfinite(base_map.f_add())) << base_bits << "+" << ex_bits;
            EXPECT_TRUE(std::isfinite(base_map.f_rescale())) << base_bits << "+" << ex_bits;
            EXPECT_TRUE(std::isfinite(base_map.f_error())) << base_bits << "+" << ex_bits;
            EXPECT_FLOAT_EQ(base_map.f_error(), 0.0F) << base_bits << "+" << ex_bits;

            if (ex_bits > 0) {
                ConstExDataMap<float> ex_map(ex_block.data(), kDim, ex_bits);
                EXPECT_TRUE(std::isfinite(ex_map.f_add_ex()))
                    << base_bits << "+" << ex_bits;
                EXPECT_TRUE(std::isfinite(ex_map.f_rescale_ex()))
                    << base_bits << "+" << ex_bits;
            }
        }
    }
}

TEST(XyQuantization, SplitInnerProductRecombinesExactly) {
    constexpr size_t kDim = 64;
    constexpr size_t kBaseBits = 3;
    constexpr size_t kExtraBits = 4;

    std::mt19937 gen(7);
    std::vector<float> data = RandomVec(kDim, gen);
    std::vector<float> centroid(kDim, 0.0F);
    std::vector<float> query = RandomVec(kDim, gen);

    XySplitResult res = SplitCode(data, centroid, kDim, kBaseBits, kExtraBits);

    double split_ip = 0;
    double direct_ip = 0;
    for (size_t i = 0; i < kDim; ++i) {
        uint32_t total = (static_cast<uint32_t>(res.base_code[i]) << kExtraBits) |
                         static_cast<uint32_t>(res.extra_code[i]);
        direct_ip += static_cast<double>(query[i]) * static_cast<double>(total);
    }
    for (size_t i = 0; i < kDim; ++i) {
        split_ip += static_cast<double>(query[i]) * static_cast<double>(res.base_code[i]) *
                    static_cast<double>(1U << kExtraBits);
        split_ip += static_cast<double>(query[i]) * static_cast<double>(res.extra_code[i]);
    }

    EXPECT_NEAR(split_ip, direct_ip, 1e-6 * std::abs(direct_ip) + 1e-6);
}

// The base code must be stored exactly once: total storage is
// (base_bits + extra_bits) bits per dimension plus 5 floats (3 base factors +
// 2 full factors), never (2*base_bits + extra_bits) as an independently
// quantized second layer would cost.
TEST(XyQuantization, StorageCostsBasePlusExtraBitsOnly) {
    constexpr size_t kDim = 128;
    constexpr size_t kBaseBits = 3;
    constexpr size_t kExtraBits = 4;

    size_t base_bytes = BaseDataMap<float>::data_bytes(kDim, kBaseBits);
    size_t extra_bytes = ExDataMap<float>::data_bytes(kDim, kExtraBits);

    EXPECT_EQ(base_bytes + extra_bytes, (kDim * (kBaseBits + kExtraBits) / 8) + (5 * 4));
    EXPECT_EQ(ExDataMap<float>::data_bytes(kDim, 0), 0U);
}

// End-to-end backward-compat check: at base_bits=1, quantize_xy_single +
// xy_single_base_dist + split_distance_boosting must agree with today's production
// formula ((1<<ex_bits)*ip_x0_qr + ip_func_(ex_code) + kbxsumq), using a
// plain reference dot product in place of the popcount kernel for ip_x0_qr.
// xy_single_full_dist must land on the same estimate and, like
// split_single_fulldist_direct, bound it with f_error_base / 2^ex_bits.
TEST(XyQuantization, EndToEndMatchesExistingFormulaAtBaseBitsOne) {
    constexpr size_t kDim = 128;
    constexpr size_t kExtraBits = 5;

    std::mt19937 gen(123);
    std::vector<float> data = RandomVec(kDim, gen);
    std::vector<float> centroid = RandomVec(kDim, gen);
    std::vector<float> query = RandomVec(kDim, gen);

    constexpr float kGAdd = 3.5F;
    constexpr float kGError = 0.75F;

    // --- Reference: today's production formula, built from the existing,
    // unchanged one_bit_code_with_factor + ex_bits_code_with_factor. ---
    std::vector<int> ref_binary_code(kDim);
    float ref_f_add_bin = 0;
    float ref_f_rescale_bin = 0;
    float ref_f_error_bin = 0;
    quant::rabitq_impl::one_bit::one_bit_code_with_factor(
        data.data(),
        centroid.data(),
        kDim,
        ref_binary_code.data(),
        ref_f_add_bin,
        ref_f_rescale_bin,
        ref_f_error_bin,
        METRIC_L2
    );

    std::vector<uint8_t> ref_ex_code(kDim);
    float ref_f_add_ex = 0;
    float ref_f_rescale_ex = 0;
    float ref_f_error_ex = 0;
    quant::rabitq_impl::ex_bits::ex_bits_code_with_factor<float, uint8_t>(
        data.data(),
        centroid.data(),
        kDim,
        kExtraBits,
        ref_ex_code.data(),
        ref_f_add_ex,
        ref_f_rescale_ex,
        ref_f_error_ex,
        METRIC_L2
    );

    double ip_x0_qr = 0;
    double ex_ip = 0;
    double sumq = 0;
    for (size_t i = 0; i < kDim; ++i) {
        ip_x0_qr += static_cast<double>(query[i]) * static_cast<double>(ref_binary_code[i]);
        ex_ip += static_cast<double>(query[i]) * static_cast<double>(ref_ex_code[i]);
        sumq += query[i];
    }
    double ref_c_b = -(static_cast<double>(1 << (kExtraBits + 1)) - 1) / 2.0;
    double ref_kbxsumq = sumq * ref_c_b;

    // 1-bit stage, as split_single_estdist_direct computes it.
    float ref_base_est = static_cast<float>(
        ref_f_add_bin + kGAdd + (ref_f_rescale_bin * (ip_x0_qr - (0.5 * sumq)))
    );

    // Boosted stage, as split_distance_boosting computes it.
    float ref_full_est = static_cast<float>(
        ref_f_add_ex + kGAdd +
        (ref_f_rescale_ex *
         (static_cast<double>(1 << kExtraBits) * ip_x0_qr + ex_ip + ref_kbxsumq))
    );

    // --- New path ---
    std::vector<char> base_data(BaseDataMap<float>::data_bytes(kDim, 1));
    std::vector<char> extra_data(ExDataMap<float>::data_bytes(kDim, kExtraBits));
    quant::quantize_xy_single(
        data.data(),
        centroid.data(),
        kDim,
        /*base_bits=*/1,
        kExtraBits,
        base_data.data(),
        extra_data.data()
    );

    SplitSingleQuery<float> q_obj(
        query.data(), kDim, kExtraBits, quant::RabitqConfig(), METRIC_L2, /*base_bits=*/1
    );

    auto base_ip_func = select_excode_ipfunc(1);
    auto extra_ip_func = select_excode_ipfunc(kExtraBits);

    float ip_base = 0;
    float base_est = 0;
    float base_low = 0;
    xy_single_base_dist(
        base_data.data(),
        base_ip_func,
        q_obj,
        kDim,
        /*base_bits=*/1,
        ip_base,
        base_est,
        base_low,
        kGAdd,
        kGError
    );

    EXPECT_FLOAT_NEARLY_EQUAL(ip_base, static_cast<float>(ip_x0_qr), 1e-3F);
    EXPECT_FLOAT_NEARLY_EQUAL(base_est, ref_base_est, 1e-2F);
    EXPECT_FLOAT_NEARLY_EQUAL(base_low, base_est - (ref_f_error_bin * kGError), 1e-4F);

    // split_distance_boosting reads g_add off the query rather than taking
    // it as a parameter; there is no x+y-specific boosting function.
    q_obj.set_g_add(std::sqrt(kGAdd));
    float boosted_est = split_distance_boosting(
        extra_data.data(), extra_ip_func, q_obj, kDim, kExtraBits, ip_base
    );

    EXPECT_FLOAT_NEARLY_EQUAL(boosted_est, ref_full_est, 1e-2F);

    // The one-shot path recomputes ip_base itself and must agree exactly with
    // the boosted path, plus supply the lower bound boosting omits.
    float full_est = 0;
    float full_low = 0;
    float full_ip_base = 0;
    xy_single_full_dist(
        base_data.data(),
        extra_data.data(),
        base_ip_func,
        extra_ip_func,
        q_obj,
        kDim,
        /*base_bits=*/1,
        kExtraBits,
        full_est,
        full_low,
        full_ip_base,
        kGAdd,
        kGError
    );

    EXPECT_FLOAT_EQ(full_ip_base, ip_base);
    EXPECT_FLOAT_EQ(full_est, boosted_est);
    EXPECT_FLOAT_NEARLY_EQUAL(
        full_low,
        full_est - (ref_f_error_bin * kGError / static_cast<float>(1 << kExtraBits)),
        1e-4F
    );
}

// The boosted estimate must equal the estimate you would get by reading the
// whole (base_bits+extra_bits)-bit code directly -- i.e. reusing the filter
// stage's base inner product loses nothing. This is the property the old
// two-layer implementation could not have, since its two layers held
// different codes.
TEST(XyQuantization, BoostedEstimateMatchesUnsplitCombinedCode) {
    constexpr size_t kDim = 128;
    constexpr size_t kBaseBits = 3;
    constexpr size_t kExtraBits = 4;

    std::mt19937 gen(2024);
    std::vector<float> data = RandomVec(kDim, gen);
    std::vector<float> centroid = RandomVec(kDim, gen);
    std::vector<float> query = RandomVec(kDim, gen);

    // g_add / g_error as the L2 search path actually supplies them
    // (||q - centroid||^2 and ||q - centroid||), so the filter's lower bound
    // below is the real one rather than an arbitrary scaling of f_error.
    float q_to_cent_sqr = 0;
    for (size_t i = 0; i < kDim; ++i) {
        float d = query[i] - centroid[i];
        q_to_cent_sqr += d * d;
    }
    const float kGError = std::sqrt(q_to_cent_sqr);
    const float kGAdd = q_to_cent_sqr;

    XySplitResult res = SplitCode(data, centroid, kDim, kBaseBits, kExtraBits);

    double total_ip = 0;
    double sumq = 0;
    for (size_t i = 0; i < kDim; ++i) {
        uint32_t total = (static_cast<uint32_t>(res.base_code[i]) << kExtraBits) |
                         static_cast<uint32_t>(res.extra_code[i]);
        total_ip += static_cast<double>(query[i]) * static_cast<double>(total);
        sumq += query[i];
    }
    double c_b = -(static_cast<double>(1 << (kBaseBits + kExtraBits)) - 1) / 2.0;
    float ref_est = static_cast<float>(
        res.f_add_full + kGAdd + (res.f_rescale_full * (total_ip + (sumq * c_b)))
    );

    std::vector<char> base_data(BaseDataMap<float>::data_bytes(kDim, kBaseBits));
    std::vector<char> extra_data(ExDataMap<float>::data_bytes(kDim, kExtraBits));
    quant::quantize_xy_single(
        data.data(),
        centroid.data(),
        kDim,
        kBaseBits,
        kExtraBits,
        base_data.data(),
        extra_data.data()
    );

    SplitSingleQuery<float> q_obj(
        query.data(), kDim, kExtraBits, quant::RabitqConfig(), METRIC_L2, kBaseBits
    );

    float ip_base = 0;
    float base_est = 0;
    float base_low = 0;
    xy_single_base_dist(
        base_data.data(),
        select_excode_ipfunc(kBaseBits),
        q_obj,
        kDim,
        kBaseBits,
        ip_base,
        base_est,
        base_low,
        kGAdd,
        kGError
    );

    q_obj.set_g_add(std::sqrt(kGAdd));
    float full_est = split_distance_boosting(
        extra_data.data(),
        select_excode_ipfunc(kExtraBits),
        q_obj,
        kDim,
        kExtraBits,
        ip_base
    );

    EXPECT_FLOAT_NEARLY_EQUAL(full_est, ref_est, 1e-2F);
    // The cheap filter's lower bound must not exclude the refined estimate,
    // otherwise the base layer would prune candidates the refine layer would
    // have kept.
    EXPECT_LE(base_low, full_est);
}

TEST(XyQuantization, CombinedBitsBeyondCapAborts) {
    constexpr size_t kDim = 64;
    std::mt19937 gen(1);
    std::vector<float> data = RandomVec(kDim, gen);
    std::vector<float> centroid(kDim, 0.0F);

    auto call_with_bad_bits = [&]() {
        SplitCode(data, centroid, kDim, /*base_bits=*/8, /*extra_bits=*/8);
    };
    EXPECT_DEATH(call_with_bad_bits(), "");
}
