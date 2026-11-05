#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/core/api.hpp"
#include "rabitqlib/core/detail/rabitq_impl.hpp"
#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/provisional/quantize.hpp"

namespace {
namespace api = rabitqlib::quant::v1;
namespace provisional = rabitqlib::quant::provisional;
namespace rabitq_impl = rabitqlib::quant::rabitq_impl;
using rabitqlib::METRIC_IP;
using rabitqlib::METRIC_L2;
using rabitqlib::MetricType;

std::vector<float> gaussian(size_t dim, std::mt19937& generator) {
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    std::vector<float> values(dim);
    for (auto& value : values) {
        value = distribution(generator);
    }
    return values;
}

struct Encoded {
    std::vector<uint16_t> codes;
    api::Factors factors;
};

// quant::v1::encode with UnpackedU16 codes, the reference for codes and factors.
Encoded v1_encode(
    const std::vector<float>& input,
    const std::vector<float>& centroid,
    size_t total_bits,
    MetricType metric,
    double t_const
) {
    const api::Parameters parameters{input.size(), total_bits, metric, t_const};
    const auto sizes = api::requirements(parameters, api::CodeLayout::UnpackedU16);
    std::vector<std::max_align_t> workspace(
        (sizes.workspace_bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t)
    );
    std::vector<uint8_t> bytes(sizes.code_bytes);
    const api::CodeOutput output{
        api::CodeLayout::UnpackedU16, {bytes.data(), bytes.size()}, {}};
    Encoded result;
    result.factors = api::encode(
        parameters,
        input.data(),
        centroid.data(),
        output,
        {workspace.data(), sizes.workspace_bytes}
    );
    result.codes.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result.codes[i] = static_cast<uint16_t>(bytes[2 * i] | (bytes[2 * i + 1] << 8U));
    }
    return result;
}

// The detail code_factors on int codes, the bit-exact reference for the wrapper.
template <typename TP>
api::Factors detail_factors(
    const std::vector<float>& residual,
    const std::vector<float>& centroid,
    const std::vector<TP>& code,
    size_t total_bits,
    MetricType metric
) {
    const std::vector<int> widened(code.begin(), code.end());
    const float cb = -static_cast<float>((1U << total_bits) - 1) / 2.0F;
    api::Factors factors;
    rabitq_impl::xy_bits::code_factors<float>(
        residual.data(),
        centroid.data(),
        residual.size(),
        widened.data(),
        cb,
        factors.add,
        factors.rescale,
        factors.error,
        metric
    );
    return factors;
}

void expect_same_factors(const api::Factors& actual, const api::Factors& expected) {
    EXPECT_EQ(actual.add, expected.add);
    EXPECT_EQ(actual.rescale, expected.rescale);
    EXPECT_EQ(actual.error, expected.error);
}

template <typename TP>
void check_code_and_factors(
    const std::vector<float>& input,
    const std::vector<float>& centroid,
    size_t total_bits,
    MetricType metric,
    double t_const
) {
    const size_t dim = input.size();
    // The residual v1::encode forms internally; elementwise subtraction is exact IEEE.
    std::vector<float> residual(dim);
    for (size_t i = 0; i < dim; ++i) {
        residual[i] = input[i] - centroid[i];
    }
    const Encoded reference = v1_encode(input, centroid, total_bits, metric, t_const);

    std::vector<TP> code(dim);
    provisional::combined_code(residual.data(), dim, total_bits, code.data(), t_const);
    std::vector<TP> detail_code(dim);
    rabitq_impl::total_bits::combined_code<float, TP>(
        residual.data(), dim, total_bits, detail_code.data(), t_const
    );
    EXPECT_EQ(code, detail_code);
    EXPECT_EQ(std::vector<uint16_t>(code.begin(), code.end()), reference.codes);

    const api::Factors factors = provisional::code_factors(
        residual.data(), centroid.data(), dim, code.data(), total_bits, metric
    );
    expect_same_factors(
        factors, detail_factors(residual, centroid, code, total_bits, metric)
    );
    // v1::encode derives rescale from the normalized code in a different rounding
    // order; add and error come from the same terms.
    EXPECT_EQ(factors.add, reference.factors.add);
    EXPECT_EQ(factors.error, reference.factors.error);
    EXPECT_NEAR(
        factors.rescale,
        reference.factors.rescale,
        1e-5F * std::abs(reference.factors.rescale)
    );

    // Every prefix of the code is itself a code of that width for the same residual.
    for (size_t prefix_bits = 1; prefix_bits < total_bits; ++prefix_bits) {
        SCOPED_TRACE(::testing::Message() << "prefix_bits " << prefix_bits);
        std::vector<TP> prefix(dim);
        for (size_t i = 0; i < dim; ++i) {
            prefix[i] = static_cast<TP>(code[i] >> (total_bits - prefix_bits));
        }
        expect_same_factors(
            provisional::code_factors(
                residual.data(), centroid.data(), dim, prefix.data(), prefix_bits, metric
            ),
            detail_factors(residual, centroid, prefix, prefix_bits, metric)
        );
    }
}

TEST(ProvisionalQuantizeTest, MatchesV1EncodeAndTheDetailFunctions) {
    std::mt19937 generator(7);
    for (size_t dim : {32U, 64U, 96U, 1056U}) {
        const std::vector<float> input = gaussian(dim, generator);
        const std::vector<float> centroid = gaussian(dim, generator);
        for (size_t total_bits = 1; total_bits <= 9; ++total_bits) {
            const double fast = rabitqlib::quant::faster_config(dim, total_bits).t_const;
            for (double t_const : {-1.0, fast}) {
                for (MetricType metric : {METRIC_L2, METRIC_IP}) {
                    SCOPED_TRACE(
                        ::testing::Message()
                        << "dim " << dim << " total_bits " << total_bits << " t_const "
                        << t_const << " metric " << metric
                    );
                    check_code_and_factors<uint16_t>(
                        input, centroid, total_bits, metric, t_const
                    );
                    if (total_bits <= 8) {
                        check_code_and_factors<uint8_t>(
                            input, centroid, total_bits, metric, t_const
                        );
                    }
                }
            }
        }
    }
}

TEST(ProvisionalQuantizeTest, NullCentroidMeansZero) {
    std::mt19937 generator(11);
    constexpr size_t kDim = 64;
    const std::vector<float> residual = gaussian(kDim, generator);
    const std::vector<float> zero(kDim, 0.0F);
    std::vector<uint8_t> code(kDim);
    provisional::combined_code(residual.data(), kDim, 4, code.data());
    for (MetricType metric : {METRIC_L2, METRIC_IP}) {
        expect_same_factors(
            provisional::code_factors(
                residual.data(), nullptr, kDim, code.data(), 4, metric
            ),
            provisional::code_factors(
                residual.data(), zero.data(), kDim, code.data(), 4, metric
            )
        );
    }
}

TEST(ProvisionalQuantizeTest, ZeroAndUnderflowingResiduals) {
    constexpr size_t kDim = 64;
    std::mt19937 generator(3);
    const std::vector<float> centroid = gaussian(kDim, generator);
    std::vector<float> tiny(kDim, 0.0F);
    // A nonzero residual whose squared norm underflows to zero keeps its sign bits.
    tiny[3] = 1e-30F;
    tiny[40] = -1e-30F;
    for (const auto& residual : {std::vector<float>(kDim, 0.0F), tiny}) {
        for (size_t total_bits = 1; total_bits <= 9; ++total_bits) {
            std::vector<float> input(kDim);
            for (size_t i = 0; i < kDim; ++i) {
                input[i] = centroid[i] + residual[i];
            }
            std::vector<uint16_t> code(kDim, 0xFFFF);
            provisional::combined_code(residual.data(), kDim, total_bits, code.data());
            for (size_t i = 0; i < kDim; ++i) {
                EXPECT_EQ(code[i], residual[i] > 0 ? 1U << (total_bits - 1) : 0U);
            }
            for (MetricType metric : {METRIC_L2, METRIC_IP}) {
                const api::Factors factors = provisional::code_factors(
                    residual.data(), centroid.data(), kDim, code.data(), total_bits, metric
                );
                EXPECT_EQ(factors.add, metric == METRIC_L2 ? 0.0F : 1.0F);
                EXPECT_EQ(factors.rescale, 0.0F);
                EXPECT_EQ(factors.error, 0.0F);
            }
        }
    }
}

template <typename Exception, typename Operation>
void expect_rejected(Operation operation, const std::string& message) {
    try {
        operation();
        ADD_FAILURE() << "accepted; expected: " << message;
    } catch (const Exception& error) { EXPECT_EQ(error.what(), message); }
}

TEST(ProvisionalQuantizeTest, RejectsInvalidArgumentsWithoutWritingOutput) {
    constexpr size_t kDim = 64;
    std::mt19937 generator(5);
    std::vector<float> residual = gaussian(kDim, generator);
    std::vector<uint8_t> code8(kDim, 0x5A);
    std::vector<uint16_t> code16(kDim, 0x5A5A);
    const auto code8_untouched = code8;
    const auto code16_untouched = code16;
    const auto encode8 = [&](size_t dim, size_t total_bits, double t_const = -1) {
        provisional::combined_code(residual.data(), dim, total_bits, code8.data(), t_const);
    };

    for (size_t dim : {0U, 31U, 33U}) {
        expect_rejected<std::invalid_argument>(
            [&] { encode8(dim, 1); }, "dim must be a positive multiple of 32"
        );
    }
    for (size_t total_bits : {0U, 9U}) {
        expect_rejected<std::invalid_argument>(
            [&] { encode8(kDim, total_bits); },
            "total_bits must be in [1, 8] for uint8_t codes"
        );
    }
    expect_rejected<std::invalid_argument>(
        [&] { provisional::combined_code(residual.data(), kDim, 10, code16.data()); },
        "total_bits must be in [1, 9] for uint16_t codes"
    );
    for (double t_const :
         {std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::infinity(),
          static_cast<double>(std::numeric_limits<int>::max())}) {
        expect_rejected<std::invalid_argument>(
            [&] { encode8(kDim, 4, t_const); },
            "t_const must be finite and at most INT_MAX / 2"
        );
    }
    expect_rejected<std::invalid_argument>(
        [&] { provisional::combined_code(nullptr, kDim, 4, code8.data()); },
        "residual must not be null"
    );
    expect_rejected<std::invalid_argument>(
        [&] {
            provisional::combined_code(
                residual.data(), kDim, 4, static_cast<uint8_t*>(nullptr)
            );
        },
        "code must not be null"
    );
    expect_rejected<std::invalid_argument>(
        [&] {
            provisional::combined_code(
                residual.data(), kDim, 4, reinterpret_cast<uint8_t*>(residual.data() + 8)
            );
        },
        "Writable buffers must not overlap other buffers or inputs"
    );

    std::vector<float> bad = residual;
    bad[17] = std::numeric_limits<float>::quiet_NaN();
    expect_rejected<std::invalid_argument>(
        [&] { provisional::combined_code(bad.data(), kDim, 4, code8.data()); },
        "Input vectors must contain only finite values"
    );
    std::vector<float> huge(kDim, 1e30F);
    expect_rejected<std::overflow_error>(
        [&] { provisional::combined_code(huge.data(), kDim, 4, code8.data()); },
        "Residual squared norm is not representable in float32"
    );
    EXPECT_EQ(code8, code8_untouched);
    EXPECT_EQ(code16, code16_untouched);

    std::vector<uint8_t> code(kDim);
    provisional::combined_code(residual.data(), kDim, 4, code.data());
    const auto factors = [&](const float* r,
                             const float* c,
                             const uint8_t* q,
                             size_t bits,
                             MetricType metric = METRIC_L2) {
        return provisional::code_factors(r, c, kDim, q, bits, metric);
    };
    expect_rejected<std::invalid_argument>(
        [&] {
            provisional::code_factors(
                residual.data(), nullptr, 48, code.data(), 4, METRIC_L2
            );
        },
        "dim must be a positive multiple of 32"
    );
    expect_rejected<std::invalid_argument>(
        [&] {
            factors(residual.data(), nullptr, code.data(), 4, static_cast<MetricType>(7));
        },
        "Metric type must be METRIC_L2 or METRIC_IP"
    );
    expect_rejected<std::invalid_argument>(
        [&] { factors(residual.data(), bad.data(), code.data(), 4); },
        "Input vectors must contain only finite values"
    );
    expect_rejected<std::invalid_argument>(
        [&] { factors(residual.data(), nullptr, nullptr, 4); }, "code must not be null"
    );
    expect_rejected<std::invalid_argument>(
        [&] { factors(residual.data(), nullptr, code.data(), 3); },
        "Unpacked code exceeds the configured bit width"
    );
    // The code of the negated residual points the other way.
    std::vector<float> negated(kDim);
    for (size_t i = 0; i < kDim; ++i) {
        negated[i] = -residual[i];
    }
    expect_rejected<std::invalid_argument>(
        [&] { factors(negated.data(), nullptr, code.data(), 4); },
        "code does not quantize residual: <residual, centered code> must be positive"
    );
}

}  // namespace
