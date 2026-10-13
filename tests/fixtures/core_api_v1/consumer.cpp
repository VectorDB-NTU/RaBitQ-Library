// Frozen v1 source consumer. Keep this independent of implementation/legacy headers.
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "rabitqlib/core/api.hpp"

namespace api = rabitqlib::quant::v1;
static_assert(RABITQLIB_QUANT_API_VERSION_MAJOR == 1);
static_assert(RABITQLIB_QUANT_API_VERSION_MINOR >= 0);
static_assert(static_cast<unsigned>(api::CodeLayout::UnpackedU16) == 0);
static_assert(static_cast<unsigned>(api::CodeLayout::SplitPacked) == 1);
static_assert(static_cast<unsigned>(rabitqlib::METRIC_L2) == 0);
static_assert(static_cast<unsigned>(rabitqlib::METRIC_IP) == 1);
static_assert(std::is_trivially_copyable_v<api::Factors>);
static_assert(std::is_trivially_copyable_v<api::QueryState>);
static_assert(std::is_same_v<
              decltype(&api::requirements),
              api::Requirements (*)(const api::Parameters&, api::CodeLayout)>);
static_assert(std::is_same_v<
              decltype(&api::encode),
              api::Factors (*)(
                  const api::Parameters&,
                  const float*,
                  const float*,
                  api::CodeOutput,
                  api::Buffer
              )>);
static_assert(std::is_same_v<
              decltype(&api::convert_codes),
              void (*)(
                  const api::Parameters&, api::CodeView, api::CodeOutput, api::Buffer
              )>);
static_assert(std::is_same_v<
              decltype(&api::prepare_query),
              api::QueryState (*)(const api::Parameters&, const float*, const float*)>);
static_assert(std::is_same_v<
              decltype(&api::compute),
              float (*)(const api::QueryState&, api::CodeView, const api::Factors&)>);

int main() {
    constexpr size_t dim = 32;
    std::array<float, dim> input{}, query{};
    for (size_t i = 0; i < dim; ++i) {
        input[i] = i % 2 == 0 ? 1.0F : -1.0F;
        query[i] = input[i] * 0.5F;
    }
    // Deliberately not the library's record layout: user metadata, separated factors.
    struct Record {
        float scale = 0;
        std::uint64_t id = 123;
        std::array<std::uint8_t, 4> signs{};
        float error = 0;
        float add = 0;
    } record;
    for (auto metric : {rabitqlib::METRIC_L2, rabitqlib::METRIC_IP}) {
        api::Parameters p{dim, 1, metric};
        auto sizes = api::requirements(p, api::CodeLayout::SplitPacked);
        if (sizes.code_bytes != 4 || sizes.extra_bytes != 0)
            return 1;
        std::vector<std::max_align_t> scratch(
            (sizes.workspace_bytes + sizeof(std::max_align_t) - 1) /
            sizeof(std::max_align_t)
        );
        api::Buffer work{scratch.data(), sizes.workspace_bytes};
        auto factors = api::encode(
            p,
            input.data(),
            nullptr,
            {api::CodeLayout::SplitPacked, {record.signs.data(), record.signs.size()}, {}},
            work
        );
        record.add = factors.add;
        record.scale = factors.rescale;
        record.error = factors.error;
        if (record.id != 123 || record.error != 0)
            return 2;
        for (auto value : record.signs)
            if (value != 0xAA)
                return 3;
        if (record.add != (metric == rabitqlib::METRIC_L2 ? 32 : 1) ||
            record.scale != (metric == rabitqlib::METRIC_L2 ? -4 : -2))
            return 4;
        auto state = api::prepare_query(p, query.data(), nullptr);
        api::CodeView packed{
            api::CodeLayout::SplitPacked, {record.signs.data(), record.signs.size()}, {}};
        if (api::compute(state, packed, {record.add, record.scale, record.error}) !=
            (metric == rabitqlib::METRIC_L2 ? 8 : -15))
            return 5;
        std::array<uint8_t, 2 * dim> raw{};
        api::convert_codes(
            p, packed, {api::CodeLayout::UnpackedU16, {raw.data(), raw.size()}, {}}, work
        );
        for (size_t i = 0; i < dim; ++i) {
            if (raw[2 * i] != (i % 2 == 0 ? 1 : 0) || raw[2 * i + 1] != 0)
                return 6;
        }
    }
}
