#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "rabitqlib/core/api.hpp"
#include "rabitqlib/core/data_layout.hpp"
#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/estimator.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/query.hpp"
#include "rabitqlib/utils/space.hpp"

namespace {
namespace api = rabitqlib::quant::v1;
using rabitqlib::METRIC_IP;
using rabitqlib::METRIC_L2;

struct Workspace {
    std::vector<std::max_align_t> storage;
    size_t bytes;
    explicit Workspace(size_t n)
        : storage((n + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t))
        , bytes(n) {}
    api::Buffer view() { return {storage.data(), bytes}; }
};

api::CodeView view(api::CodeOutput output) {
    return {
        output.layout,
        {output.codes.data, output.codes.bytes},
        {output.extra.data, output.extra.bytes}};
}

TEST(CoreApiTest, RejectsNonFiniteValuesAtEveryInputPosition) {
    for (size_t dim : {2U, 3U, 15U, 16U, 17U, 31U, 32U, 33U, 65U}) {
        const api::Parameters parameters{dim, 1};
        const auto sizes = api::requirements(parameters, api::CodeLayout::UnpackedU16);
        Workspace workspace(sizes.workspace_bytes);
        std::vector<uint8_t> codes(sizes.code_bytes, 0x5A);
        const api::CodeOutput output{
            api::CodeLayout::UnpackedU16, {codes.data(), codes.size()}, {}};
        std::vector<float> values(dim, 0.0F), finite(dim, -0.0F);
        for (size_t position = 0; position < dim; ++position) {
            for (float bad :
                 {std::numeric_limits<float>::quiet_NaN(),
                  std::numeric_limits<float>::infinity(),
                  -std::numeric_limits<float>::infinity()}) {
                SCOPED_TRACE(::testing::Message() << dim << "/" << position << "/" << bad);
                values[position] = bad;
                const auto rejects = [](auto operation) {
                    try {
                        operation();
                        FAIL() << "Non-finite input was accepted";
                    } catch (const std::invalid_argument& error) {
                        EXPECT_STREQ(
                            error.what(), "Input vectors must contain only finite values"
                        );
                    }
                };
                rejects([&] { api::prepare_query(parameters, values.data(), nullptr); });
                rejects([&] {
                    api::prepare_query(parameters, finite.data(), values.data());
                });
                rejects([&] {
                    api::encode(
                        parameters, values.data(), nullptr, output, workspace.view()
                    );
                });
                rejects([&] {
                    api::encode(
                        parameters, finite.data(), values.data(), output, workspace.view()
                    );
                });
                values[position] = 0.0F;
            }
        }
        EXPECT_TRUE(std::all_of(codes.begin(), codes.end(), [](uint8_t v) {
            return v == 0x5A;
        }));
        EXPECT_NO_THROW(api::prepare_query(parameters, finite.data(), nullptr));
    }
}

TEST(CoreApiTest, ExactWorkspaceSizeProtectsBoundaryBytes) {
    for (size_t dim : {2U, 3U, 31U, 33U, 65U, 96U}) {
        for (size_t bits : {1U, 4U, 9U}) {
            SCOPED_TRACE(dim);
            SCOPED_TRACE(bits);
            const api::Parameters p{dim, bits};
            const auto r = api::requirements(p, api::CodeLayout::UnpackedU16);
            std::vector<std::max_align_t> storage(
                (r.workspace_bytes + 2 * r.workspace_alignment + sizeof(std::max_align_t) -
                 1) /
                sizeof(std::max_align_t)
            );
            auto* bytes = reinterpret_cast<uint8_t*>(storage.data());
            const size_t capacity = storage.size() * sizeof(std::max_align_t);
            std::fill_n(bytes, capacity, uint8_t{0xA5});
            api::Buffer workspace{bytes + r.workspace_alignment, r.workspace_bytes};
            std::vector<float> data(dim);
            for (size_t i = 0; i < dim; ++i) {
                data[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 11;
            }
            std::vector<uint8_t> codes(r.code_bytes), converted(r.code_bytes);
            api::CodeOutput out{
                api::CodeLayout::UnpackedU16, {codes.data(), codes.size()}, {}};
            const auto factors = api::encode(p, data.data(), nullptr, out, workspace);
            EXPECT_TRUE(std::isfinite(factors.rescale));
            api::convert_codes(
                p,
                view(out),
                {out.layout, {converted.data(), converted.size()}, {}},
                workspace
            );
            EXPECT_EQ(codes, converted);
            for (size_t i = 0; i < r.workspace_alignment; ++i) {
                EXPECT_EQ(bytes[i], 0xA5);
            }
            for (size_t i = r.workspace_alignment + r.workspace_bytes; i < capacity; ++i) {
                EXPECT_EQ(bytes[i], 0xA5);
            }
            --workspace.bytes;
            EXPECT_THROW(
                api::encode(p, data.data(), nullptr, out, workspace), std::invalid_argument
            );
        }
    }
}

TEST(CoreApiTest, CodesFactorsAndBothLayoutsMatchExistingEncoder) {
    std::mt19937 generator(123);
    std::uniform_real_distribution<float> distribution(-1, 1);
    for (size_t dim : {32U, 64U, 96U, 128U, 160U, 256U}) {
        std::vector<float> data(dim), centroid(dim), query(dim);
        for (size_t i = 0; i < dim; ++i) {
            centroid[i] = distribution(generator);
            data[i] = i % 7 == 0 ? centroid[i] : distribution(generator);
            query[i] = distribution(generator);
        }
        for (size_t bits = 1; bits <= 9; ++bits) {
            for (auto metric : {METRIC_L2, METRIC_IP}) {
                for (double scale : {-1.0, 7.25}) {
                    SCOPED_TRACE(
                        ::testing::Message()
                        << dim << "," << bits << "," << metric << "," << scale
                    );
                    api::Parameters p{dim, bits, metric, scale};
                    auto raw = api::requirements(p, api::CodeLayout::UnpackedU16);
                    auto packed = api::requirements(p, api::CodeLayout::SplitPacked);
                    Workspace workspace(raw.workspace_bytes);
                    // Deliberately unaligned code buffers, with canaries and unrelated
                    // records.
                    std::vector<uint8_t> code(raw.code_bytes + 2, 0xA5),
                        roundtrip(raw.code_bytes + 2, 0xA5);
                    std::vector<uint8_t> sign(packed.code_bytes + 2, 0xA5),
                        extra(packed.extra_bytes + 2, 0xA5);
                    api::CodeOutput out{
                        api::CodeLayout::UnpackedU16,
                        {code.data() + 1, raw.code_bytes},
                        {}};
                    auto factors =
                        api::encode(p, data.data(), centroid.data(), out, workspace.view());
                    std::vector<uint16_t> reference(dim);
                    float add = 0, rescale = 0, error = 0;
                    rabitqlib::quant::RabitqConfig config;
                    config.t_const = scale;
                    rabitqlib::quant::quantize_full_single(
                        data.data(),
                        centroid.data(),
                        dim,
                        bits,
                        reference.data(),
                        add,
                        rescale,
                        error,
                        metric,
                        config
                    );
                    EXPECT_EQ(factors.add, add);
                    EXPECT_EQ(factors.rescale, rescale);
                    EXPECT_EQ(factors.error, error);
                    for (size_t i = 0; i < dim; ++i) {
                        EXPECT_EQ(
                            code[1 + 2 * i] | (uint16_t{code[2 + 2 * i]} << 8U),
                            reference[i]
                        );
                    }
                    api::CodeOutput split{
                        api::CodeLayout::SplitPacked,
                        {sign.data() + 1, packed.code_bytes},
                        {extra.data() + 1, packed.extra_bytes}};
                    api::convert_codes(p, view(out), split, workspace.view());
                    api::CodeOutput restored{
                        api::CodeLayout::UnpackedU16,
                        {roundtrip.data() + 1, raw.code_bytes},
                        {}};
                    api::convert_codes(p, view(split), restored, workspace.view());
                    EXPECT_EQ(code, roundtrip);
                    const auto state = api::prepare_query(p, query.data(), centroid.data());
                    const float g_add =
                        metric == METRIC_L2
                            ? rabitqlib::euclidean_sqr(query.data(), centroid.data(), dim)
                            : -rabitqlib::dot_product(query.data(), centroid.data(), dim);
                    float sum = 0;
                    for (float x : query) {
                        sum += x;
                    }
                    const float expected = rabitqlib::quant::full_est_dist(
                        reference.data(),
                        query.data(),
                        rabitqlib::excode_ipimpl::ip_fxi<float, uint16_t>,
                        dim,
                        bits,
                        add,
                        rescale,
                        g_add,
                        -0.5F * sum
                    );
                    // The final result may cancel large dot-product terms. Use the
                    // standard gamma_n bound on absolute terms of both reductions.
                    double absolute_dot = 0;
                    for (size_t i = 0; i < dim; ++i) {
                        absolute_dot +=
                            std::abs(static_cast<double>(query[i]) * reference[i]);
                    }
                    const double eps = std::numeric_limits<float>::epsilon();
                    const double gamma = (dim + 8) * eps / (1 - (dim + 8) * eps);
                    const float tolerance = static_cast<float>(
                        2 * gamma * std::abs(rescale) *
                            (absolute_dot + std::abs(sum) * 0.5 * ((1U << bits) - 1)) +
                        8 * eps * (1 + std::abs(add) + std::abs(g_add) + std::abs(expected))
                    );
                    EXPECT_NEAR(
                        api::compute(state, view(out), factors), expected, tolerance
                    );
                    EXPECT_NEAR(
                        api::compute(state, view(split), factors), expected, tolerance
                    );
                    if (bits > 1) {
                        // A stronger oracle uses the existing packed estimator and
                        // independently computed legacy factors with the same layout.
                        std::vector<char> bin_data(
                            rabitqlib::BinDataMap<float>::data_bytes(dim)
                        );
                        std::vector<char> ex_data(
                            rabitqlib::ExDataMap<float>::data_bytes(dim, bits - 1)
                        );
                        rabitqlib::BinDataMap<float> bin_map(bin_data.data(), dim);
                        rabitqlib::ExDataMap<float> ex_map(ex_data.data(), dim, bits - 1);
                        std::copy_n(sign.data() + 1, packed.code_bytes, bin_map.bin_code());
                        std::copy_n(extra.data() + 1, packed.extra_bytes, ex_map.ex_code());
                        ex_map.f_add_ex() = add;
                        ex_map.f_rescale_ex() = rescale;
                        rabitqlib::SplitSingleQuery<float> prepared(
                            query.data(), dim, bits - 1, config, metric
                        );
                        float legacy_distance = 0, lower = 0, product = 0;
                        rabitqlib::split_single_fulldist(
                            bin_data.data(),
                            ex_data.data(),
                            rabitqlib::select_excode_ipfunc(bits - 1),
                            prepared,
                            dim,
                            bits - 1,
                            legacy_distance,
                            lower,
                            product,
                            g_add,
                            0
                        );
                        EXPECT_NEAR(
                            api::compute(state, view(split), factors),
                            legacy_distance,
                            2 * tolerance
                        );
                    }
                    const auto direct = api::encode(
                        p, data.data(), centroid.data(), split, workspace.view()
                    );
                    EXPECT_EQ(direct.add, factors.add);
                    EXPECT_EQ(direct.rescale, factors.rescale);
                    EXPECT_NEAR(
                        api::compute(state, view(split), direct), expected, tolerance
                    );
                    for (const auto* buffer : {&code, &roundtrip, &sign, &extra}) {
                        EXPECT_EQ(buffer->front(), 0xA5);
                        EXPECT_EQ(buffer->back(), 0xA5);
                    }
                }
            }
        }
    }
}

TEST(CoreApiTest, UnpackedCodesDoNotRequireLibraryPadding) {
    for (size_t dim : {2U, 3U, 15U, 33U, 65U}) {
        std::vector<float> data(dim, 1.0F), query(dim, 0.5F);
        api::Parameters p{dim, 4};
        const auto r = api::requirements(p, api::CodeLayout::UnpackedU16);
        Workspace workspace(r.workspace_bytes);
        std::vector<uint8_t> codes(r.code_bytes);
        api::CodeOutput out{api::CodeLayout::UnpackedU16, {codes.data(), codes.size()}, {}};
        auto f = api::encode(p, data.data(), nullptr, out, workspace.view());
        auto state = api::prepare_query(p, query.data(), nullptr);
        EXPECT_NEAR(
            api::compute(state, view(out), f), static_cast<float>(dim) * 0.25F, 1e-4F
        );
        EXPECT_THROW(
            api::requirements(p, api::CodeLayout::SplitPacked), std::invalid_argument
        );
    }
}

TEST(CoreApiTest, NullCentroidZeroResidualAndCallerFactorColumns) {
    constexpr size_t kDim = 96;
    std::array<float, kDim> data{}, query{};
    query.fill(0.25F);
    for (auto metric : {METRIC_L2, METRIC_IP}) {
        for (size_t bits : {1U, 4U, 9U}) {
            api::Parameters p{kDim, bits, metric};
            const auto sizes = api::requirements(p, api::CodeLayout::SplitPacked);
            Workspace workspace(sizes.workspace_bytes);
            std::vector<uint8_t> signs(sizes.code_bytes), extra(sizes.extra_bytes);
            api::CodeOutput out{
                api::CodeLayout::SplitPacked,
                {signs.data(), signs.size()},
                {extra.data(), extra.size()}};
            auto f = api::encode(p, data.data(), nullptr, out, workspace.view());
            // Store factors as separate columns, not adjacent to either code buffer.
            std::array<float, 1> add{f.add}, scale{f.rescale}, error{f.error};
            auto q = api::prepare_query(p, query.data(), nullptr);
            EXPECT_EQ(q.dimension(), kDim);
            EXPECT_EQ(q.total_bits(), bits);
            EXPECT_EQ(q.metric_type(), metric);
            EXPECT_FLOAT_EQ(
                api::compute(q, view(out), {add[0], scale[0], error[0]}),
                metric == METRIC_L2 ? 6.0F : 1.0F
            );
            EXPECT_EQ(f.rescale, 0);
            EXPECT_EQ(f.error, 0);
            EXPECT_TRUE(std::all_of(signs.begin(), signs.end(), [](uint8_t v) {
                return v == 0;
            }));
        }
    }
}

TEST(CoreApiTest, RejectsBadParametersAndBuffersBeforeWritingCodes) {
    api::Parameters p{64, 4};
    auto r = api::requirements(p, api::CodeLayout::UnpackedU16);
    Workspace workspace(r.workspace_bytes);
    std::array<float, 64> data{};
    std::vector<uint8_t> code(r.code_bytes, 0x5A);
    api::CodeOutput out{api::CodeLayout::UnpackedU16, {code.data(), code.size()}, {}};
    for (size_t dim : {0U, 1U}) {
        auto bad = p;
        bad.dim = dim;
        EXPECT_THROW(api::requirements(bad, out.layout), std::invalid_argument);
    }
    for (size_t bits : {0U, 10U}) {
        auto bad = p;
        bad.total_bits = bits;
        EXPECT_THROW(api::requirements(bad, out.layout), std::invalid_argument);
    }
    auto bad = p;
    bad.metric = static_cast<rabitqlib::MetricType>(255);
    EXPECT_THROW(api::requirements(bad, out.layout), std::invalid_argument);
    bad = p;
    bad.t_const = std::numeric_limits<double>::infinity();
    EXPECT_THROW(api::requirements(bad, out.layout), std::invalid_argument);
    bad = p;
    bad.dim = std::numeric_limits<size_t>::max() & ~size_t{31};
    EXPECT_THROW(api::requirements(bad, out.layout), std::length_error);
    bad.dim = std::numeric_limits<size_t>::max() / sizeof(float);
    EXPECT_THROW(api::requirements(bad, out.layout), std::length_error);
    EXPECT_THROW(
        api::requirements(p, static_cast<api::CodeLayout>(255)), std::invalid_argument
    );
    EXPECT_THROW(
        api::encode(p, nullptr, nullptr, out, workspace.view()), std::invalid_argument
    );
    EXPECT_THROW(
        api::encode(
            p, data.data(), nullptr, out, {workspace.storage.data(), r.workspace_bytes - 1}
        ),
        std::invalid_argument
    );
    EXPECT_THROW(
        api::encode(
            p,
            data.data(),
            nullptr,
            out,
            {reinterpret_cast<char*>(workspace.storage.data()) + 1, r.workspace_bytes}
        ),
        std::invalid_argument
    );
    auto small = out;
    small.codes.bytes--;
    EXPECT_THROW(
        api::encode(p, data.data(), nullptr, small, workspace.view()), std::invalid_argument
    );
    auto overlap = out;
    overlap.codes = {data.data(), sizeof(data)};
    EXPECT_THROW(
        api::encode(p, data.data(), nullptr, overlap, workspace.view()),
        std::invalid_argument
    );
    overlap.codes = workspace.view();
    EXPECT_THROW(
        api::encode(p, data.data(), nullptr, overlap, workspace.view()),
        std::invalid_argument
    );
    data[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(
        api::encode(p, data.data(), nullptr, out, workspace.view()), std::invalid_argument
    );
    data[0] = std::numeric_limits<float>::max();
    EXPECT_THROW(
        api::encode(p, data.data(), nullptr, out, workspace.view()), std::overflow_error
    );
    EXPECT_TRUE(std::all_of(code.begin(), code.end(), [](uint8_t v) { return v == 0x5A; }));
    data[0] = 0;
    auto q = api::prepare_query(p, data.data(), nullptr);
    EXPECT_THROW(api::compute({}, view(out), {}), std::invalid_argument);
    EXPECT_THROW(
        api::compute(q, view(out), {}), std::invalid_argument
    );  // Invalid logical codes.
    EXPECT_THROW(
        api::compute(q, view(out), {std::numeric_limits<float>::quiet_NaN(), 0, 0}),
        std::invalid_argument
    );
    EXPECT_THROW(
        api::convert_codes(p, view(out), out, workspace.view()), std::invalid_argument
    );
}

TEST(CoreApiTest, ReusedQueryAndWorkspaceStillValidateEachRecord) {
    constexpr size_t kDim = 32;
    const api::Parameters parameters{kDim, 4, METRIC_L2};
    const auto raw = api::requirements(parameters, api::CodeLayout::UnpackedU16);
    const auto packed = api::requirements(parameters, api::CodeLayout::SplitPacked);
    Workspace workspace(raw.workspace_bytes);
    std::array<float, kDim> data{};
    data.fill(0.25F);
    std::vector<uint8_t> codes(raw.code_bytes), sign(packed.code_bytes),
        extra(packed.extra_bytes);
    api::CodeOutput output{api::CodeLayout::UnpackedU16, {codes.data(), codes.size()}, {}};
    api::CodeOutput destination{
        api::CodeLayout::SplitPacked,
        {sign.data(), sign.size()},
        {extra.data(), extra.size()}};
    const auto factors =
        api::encode(parameters, data.data(), nullptr, output, workspace.view());
    const auto query = api::prepare_query(parameters, data.data(), nullptr);
    const float expected = api::compute(query, view(output), factors);
    api::convert_codes(parameters, view(output), destination, workspace.view());
    const auto expected_sign = sign, expected_extra = extra;

    auto bad_factors = factors;
    bad_factors.error = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(api::compute(query, view(output), bad_factors), std::invalid_argument);
    const uint8_t first = codes[0];
    codes[0] = 255;
    EXPECT_THROW(api::compute(query, view(output), factors), std::invalid_argument);
    EXPECT_THROW(
        api::convert_codes(parameters, view(output), destination, workspace.view()),
        std::invalid_argument
    );
    EXPECT_EQ(sign, expected_sign);
    EXPECT_EQ(extra, expected_extra);
    codes[0] = first;
    auto short_destination = destination;
    --short_destination.codes.bytes;
    EXPECT_THROW(
        api::convert_codes(parameters, view(output), short_destination, workspace.view()),
        std::invalid_argument
    );
    auto invalid_layout = destination;
    invalid_layout.layout = static_cast<api::CodeLayout>(255);
    EXPECT_THROW(
        api::convert_codes(parameters, view(output), invalid_layout, workspace.view()),
        std::invalid_argument
    );
    EXPECT_EQ(api::compute(query, view(output), factors), expected);
    api::convert_codes(parameters, view(output), destination, workspace.view());
    EXPECT_EQ(sign, expected_sign);
    EXPECT_EQ(extra, expected_extra);
}

TEST(CoreApiTest, ScoringRetainsBufferAndNumericalChecks) {
    constexpr size_t kDim = 64;
    std::array<float, kDim> values{};
    std::array<uint8_t, kDim * 2> codes{}, extra{};
    float (*volatile compute_pointer)(
        const api::QueryState&, api::CodeView, const api::Factors&
    ) = &api::compute;
    alignas(float) std::array<uint8_t, sizeof(float) * kDim + 1> misaligned{};
    const auto* bad_alignment = reinterpret_cast<const float*>(misaligned.data() + 1);
    EXPECT_THROW(
        api::prepare_query({kDim, 4}, bad_alignment, nullptr), std::invalid_argument
    );
    EXPECT_THROW(
        api::prepare_query({kDim, 4}, values.data(), bad_alignment), std::invalid_argument
    );
    for (size_t bits : {1U, 4U, 9U}) {
        for (auto metric : {METRIC_L2, METRIC_IP}) {
            const api::Parameters parameters{kDim, bits, metric};
            const auto query = api::prepare_query(parameters, values.data(), nullptr);
            for (auto layout :
                 {api::CodeLayout::SplitPacked, api::CodeLayout::UnpackedU16}) {
                const auto sizes = api::requirements(parameters, layout);
                const api::CodeView valid{
                    layout,
                    {codes.data(), sizes.code_bytes},
                    {extra.data(), sizes.extra_bytes}};
                EXPECT_EQ(
                    compute_pointer(query, valid, {}), api::compute(query, valid, {})
                );
                auto invalid = valid;
                invalid.codes.data = nullptr;
                EXPECT_THROW(api::compute(query, invalid, {}), std::invalid_argument);
                invalid = valid;
                invalid.codes.bytes--;
                EXPECT_THROW(api::compute(query, invalid, {}), std::invalid_argument);
                invalid = valid;
                invalid.codes.data = reinterpret_cast<const void*>(
                    std::numeric_limits<uintptr_t>::max() - sizes.code_bytes + 1
                );
                EXPECT_THROW(api::compute(query, invalid, {}), std::invalid_argument);
                invalid = valid;
                invalid.layout = static_cast<api::CodeLayout>(255);
                EXPECT_THROW(api::compute(query, invalid, {}), std::invalid_argument);
                invalid = valid;
                invalid.extra.data = nullptr;
                if (sizes.extra_bytes != 0) {
                    EXPECT_THROW(api::compute(query, invalid, {}), std::invalid_argument);
                    invalid = valid;
                    invalid.extra.bytes--;
                    EXPECT_THROW(api::compute(query, invalid, {}), std::invalid_argument);
                    invalid = valid;
                    invalid.extra.data = reinterpret_cast<const void*>(
                        std::numeric_limits<uintptr_t>::max() - sizes.extra_bytes + 1
                    );
                    EXPECT_THROW(api::compute(query, invalid, {}), std::invalid_argument);
                } else {
                    EXPECT_NO_THROW(api::compute(query, invalid, {}));
                }
                for (float bad :
                     {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
                    EXPECT_THROW(
                        api::compute(query, valid, {bad, 0, 0}), std::invalid_argument
                    );
                    EXPECT_THROW(
                        api::compute(query, valid, {0, bad, 0}), std::invalid_argument
                    );
                    EXPECT_THROW(
                        api::compute(query, valid, {0, 0, bad}), std::invalid_argument
                    );
                }
            }
        }
    }
    values.fill(1e18F);
    const auto large_query = api::prepare_query({kDim, 1}, values.data(), nullptr);
    EXPECT_THROW(
        api::compute(
            large_query,
            {api::CodeLayout::SplitPacked, {codes.data(), kDim / 8}, {}},
            {std::numeric_limits<float>::max(), 0, 0}
        ),
        std::overflow_error
    );
    std::array<float, 65> unpadded{};
    const auto raw_query = api::prepare_query({65, 4}, unpadded.data(), nullptr);
    EXPECT_THROW(
        api::compute(raw_query, {api::CodeLayout::SplitPacked, {}, {}}, {}),
        std::invalid_argument
    );
}

TEST(CoreApiTest, ConcurrentCallsUseOnlyTheirOwnWorkspace) {
    struct Case {
        api::Parameters parameters;
        std::vector<float> data, query;
        std::vector<uint8_t> signs, extra, unpacked;
        api::Factors factors;
        float score = 0;
    };
    std::array<Case, 4> cases;
    for (size_t i = 0; i < cases.size(); ++i) {
        auto& c = cases[i];
        c.parameters = {
            32 + i * 64,
            i == 0   ? 1U
            : i == 1 ? 4U
                     : 9U,
            i % 2 == 0 ? METRIC_L2 : METRIC_IP};
        const size_t dim = c.parameters.dim;
        c.data.resize(dim);
        c.query.resize(dim);
        for (size_t j = 0; j < dim; ++j) {
            c.data[j] = std::sin(static_cast<float>(j + i * 7)) * static_cast<float>(i + 1);
            c.query[j] = std::cos(static_cast<float>(j + i * 11));
        }
        const auto r = api::requirements(c.parameters, api::CodeLayout::SplitPacked);
        Workspace scratch(r.workspace_bytes);
        c.signs.resize(r.code_bytes);
        c.extra.resize(r.extra_bytes);
        c.unpacked.resize(dim * sizeof(uint16_t));
        api::CodeOutput out{
            api::CodeLayout::SplitPacked,
            {c.signs.data(), c.signs.size()},
            {c.extra.data(), c.extra.size()}};
        c.factors = api::encode(c.parameters, c.data.data(), nullptr, out, scratch.view());
        c.score = api::compute(
            api::prepare_query(c.parameters, c.query.data(), nullptr), view(out), c.factors
        );
        api::convert_codes(
            c.parameters,
            view(out),
            {api::CodeLayout::UnpackedU16, {c.unpacked.data(), c.unpacked.size()}, {}},
            scratch.view()
        );
    }

    std::promise<void> start;
    const auto gate = start.get_future().share();
    std::atomic<size_t> ready{0};
    std::vector<std::future<bool>> workers;
    workers.reserve(cases.size());
    for (size_t worker = 0; worker < cases.size(); ++worker) {
        workers.push_back(std::async(std::launch::async, [&, worker] {
            ++ready;
            gate.wait();
            for (size_t run = 0; run < 20; ++run) {
                // Each round uses distinct inputs across workers, and changes shape
                // within each worker. Compare every result to a serial oracle.
                const auto& c = cases[(worker + run) % cases.size()];
                const auto r =
                    api::requirements(c.parameters, api::CodeLayout::SplitPacked);
                Workspace scratch(r.workspace_bytes);
                std::vector<uint8_t> signs(r.code_bytes), extra(r.extra_bytes);
                std::vector<uint8_t> unpacked(c.unpacked.size());
                api::CodeOutput out{
                    api::CodeLayout::SplitPacked,
                    {signs.data(), signs.size()},
                    {extra.data(), extra.size()}};
                const auto f =
                    api::encode(c.parameters, c.data.data(), nullptr, out, scratch.view());
                const auto query =
                    api::prepare_query(c.parameters, c.query.data(), nullptr);
                if (signs != c.signs || extra != c.extra || f.add != c.factors.add ||
                    f.rescale != c.factors.rescale || f.error != c.factors.error ||
                    api::compute(query, view(out), f) != c.score) {
                    return false;
                }
                api::CodeOutput raw{
                    api::CodeLayout::UnpackedU16, {unpacked.data(), unpacked.size()}, {}};
                api::convert_codes(c.parameters, view(out), raw, scratch.view());
                if (unpacked != c.unpacked) {
                    return false;
                }
                std::fill(signs.begin(), signs.end(), 0xA5);
                std::fill(extra.begin(), extra.end(), 0xA5);
                api::convert_codes(c.parameters, view(raw), out, scratch.view());
                if (signs != c.signs || extra != c.extra ||
                    api::compute(query, view(out), f) != c.score) {
                    return false;
                }
            }
            return true;
        }));
    }
    while (ready.load() != cases.size()) {
        std::this_thread::yield();
    }
    start.set_value();
    for (auto& worker : workers) {
        EXPECT_TRUE(worker.get());
    }
}
}  // namespace
