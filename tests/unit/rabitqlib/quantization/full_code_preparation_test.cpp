#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rabitqlib/core/api.hpp"
#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/metric.hpp"

namespace rabitqlib::quant {
namespace {

TEST(FullCodePreparation, SquaredNormUnderflowPreservesSigns) {
    namespace api = rabitqlib::quant::v1;
    for (const size_t dim : {32U, 64U, 96U}) {
        std::vector<float> data(dim), centroid(dim, 0);
        for (size_t i = 0; i < dim; ++i) {
            data[i] = i % 3 == 0 ? 0.0F : i % 3 == 1 ? 1e-30F : -1e-30F;
        }
        // Values remain nonzero floats, but every squared term underflows to zero.
        ASSERT_GT(data[1], 0.0F);
        ASSERT_EQ(data[1] * data[1], 0.0F);
        for (const auto metric : {METRIC_L2, METRIC_IP}) {
            for (const size_t bits : {1U, 2U, 4U, 9U}) {
                for (const double scale : {-1.0, 7.0}) {
                    std::vector<uint16_t> code(dim);
                    RabitqConfig config;
                    config.t_const = scale;
                    float add, rescale, error;
                    quantize_full_single(
                        data.data(),
                        centroid.data(),
                        dim,
                        bits,
                        code.data(),
                        add,
                        rescale,
                        error,
                        metric,
                        config
                    );
                    EXPECT_EQ(add, metric == METRIC_L2 ? 0.0F : 1.0F);
                    EXPECT_EQ(rescale, 0.0F);
                    EXPECT_EQ(error, 0.0F);
                    for (size_t i = 0; i < dim; ++i) {
                        EXPECT_EQ(code[i], data[i] > 0 ? 1U << (bits - 1) : 0U);
                    }

                    const api::Parameters parameters{dim, bits, metric, scale};
                    const auto layout = api::CodeLayout::UnpackedU16;
                    const auto requirements = api::requirements(parameters, layout);
                    std::vector<uint8_t> bytes(requirements.code_bytes);
                    std::vector<std::max_align_t> workspace(
                        (requirements.workspace_bytes + sizeof(std::max_align_t) - 1) /
                        sizeof(std::max_align_t)
                    );
                    const auto factors = api::encode(
                        parameters,
                        data.data(),
                        centroid.data(),
                        {layout, {bytes.data(), bytes.size()}, {}},
                        {workspace.data(), workspace.size() * sizeof(std::max_align_t)}
                    );
                    EXPECT_EQ(factors.add, add);
                    EXPECT_EQ(factors.rescale, rescale);
                    EXPECT_EQ(factors.error, error);
                    for (size_t i = 0; i < dim; ++i) {
                        const uint16_t value = static_cast<uint16_t>(
                            bytes[2 * i] | (uint16_t{bytes[2 * i + 1]} << 8U)
                        );
                        EXPECT_EQ(value, code[i]);
                    }
                }
            }
        }
    }
}

}  // namespace
}  // namespace rabitqlib::quant
