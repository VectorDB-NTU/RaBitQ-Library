#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "rabitqlib/core/api.hpp"
#include "rabitqlib/core/metric.hpp"

namespace {
constexpr size_t kDim = 65;
using Vector = std::array<float, kDim>;

// Stand-in for the application's existing orthogonal transform. Apply the same
// transform to data, queries and nonzero centroids. The library owns none of it.
Vector external_rotation(const Vector& input) {
    Vector output{};
    for (size_t i = 0; i < kDim; ++i) {
        output[i] = input[(17 * i) % kDim] * (i % 2 == 0 ? 1.0F : -1.0F);
    }
    return output;
}
}  // namespace

int main() {
    namespace api = rabitqlib::quant::v1;
    const api::Parameters parameters{kDim, 4, rabitqlib::METRIC_L2};
    const auto sizes = api::requirements(parameters, api::CodeLayout::UnpackedU16);
    // Allocate once during setup, or substitute a block from your memory pool.
    std::vector<std::max_align_t> scratch(
        (sizes.workspace_bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t)
    );
    const api::Buffer workspace{scratch.data(), sizes.workspace_bytes};

    struct Record {
        uint64_t application_id;
        std::array<uint8_t, kDim * 2> code;
    };
    std::array<Record, 2> records{};
    // Factors can live in separate columns, completely independent of the codes.
    std::array<float, 2> add{}, rescale{}, error{};
    Vector raw_query{};
    for (size_t i = 0; i < kDim; ++i) {
        raw_query[i] = std::cos(static_cast<float>(i) * 0.11F);
    }
    const auto query = external_rotation(raw_query);
    const auto state = api::prepare_query(parameters, query.data(), nullptr);
    // query must stay alive and immutable while state (or any copy) is in use.
    for (size_t row = 0; row < records.size(); ++row) {
        Vector input{};
        for (size_t i = 0; i < kDim; ++i) {
            input[i] = std::sin(static_cast<float>(i + row) * 0.17F);
        }
        const auto transformed = external_rotation(input);
        records[row].application_id = 100 + row;
        auto& code = records[row].code;
        const auto factors = api::encode(
            parameters,
            transformed.data(),
            nullptr,  // A zero centroid; pass a transformed centroid for residual coding.
            {api::CodeLayout::UnpackedU16, {code.data(), code.size()}, {}},
            workspace
        );
        add[row] = factors.add;
        rescale[row] = factors.rescale;
        error[row] = factors.error;
        const float distance = api::compute(
            state,
            {api::CodeLayout::UnpackedU16, {code.data(), code.size()}, {}},
            {add[row], rescale[row], error[row]}
        );
        if (!std::isfinite(distance)) {
            return 1;
        }
        std::cout << records[row].application_id << ": " << distance << '\n';
    }
}
