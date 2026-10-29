#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rabitqlib/core/api.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"

namespace {
// This translation unit is linked before the core archive: this must also work
// before any library translation unit has performed dynamic initialization.
const bool kBeforeMain = [] {
    namespace api = rabitqlib::quant::v1;
    std::array<float, 64> input{}, rotated{}, query{};
    double input_norm_sqr = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 11) - 5.0F;
        input_norm_sqr += static_cast<double>(input[i]) * input[i];
    }
    rabitqlib::rotator_impl::FhtKacRotator rotator(64, 64, 42);
    rotator.rotate(input.data(), rotated.data());
    // A half-length collinear query gives known, nonzero results for both metrics.
    for (size_t i = 0; i < query.size(); ++i) {
        query[i] = rotated[i] * 0.5F;
    }
    for (const auto metric : {rabitqlib::METRIC_L2, rabitqlib::METRIC_IP}) {
        const double expected = metric == rabitqlib::METRIC_L2 ? 0.25 * input_norm_sqr
                                                               : 1 - 0.5 * input_norm_sqr;
        // Reference the original coordinates, allowing float rotation/scoring roundoff.
        const double tolerance = 2e-5 * (1 + input_norm_sqr);
        for (size_t bits = 1; bits <= 9; ++bits) {
            const api::Parameters params{64, bits, metric};
            const auto req = api::requirements(params, api::CodeLayout::SplitPacked);
            std::vector<uint8_t> codes(req.code_bytes), extra(req.extra_bytes);
            std::vector<std::max_align_t> work(
                (req.workspace_bytes + sizeof(std::max_align_t) - 1) /
                sizeof(std::max_align_t)
            );
            const auto factors = api::encode(
                params,
                rotated.data(),
                nullptr,
                {api::CodeLayout::SplitPacked,
                 {codes.data(), codes.size()},
                 {extra.data(), extra.size()}},
                {work.data(), req.workspace_bytes}
            );
            const auto prepared = api::prepare_query(params, query.data(), nullptr);
            const float distance = api::compute(
                prepared,
                {api::CodeLayout::SplitPacked,
                 {codes.data(), codes.size()},
                 {extra.data(), extra.size()}},
                factors
            );
            if (!std::isfinite(distance) || std::abs(distance - expected) > tolerance) {
                return false;
            }
        }
    }
    return true;
}();
}  // namespace

int main() { return kBeforeMain ? 0 : 1; }
