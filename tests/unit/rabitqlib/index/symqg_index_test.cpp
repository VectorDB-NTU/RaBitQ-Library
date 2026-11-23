#include "rabitqlib/index/symqg_index.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/types.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/index/types.hpp"
#include "temp_file.hpp"

namespace rabitqlib {
namespace {

using symqg::QGInitialization;

static_assert(!std::is_copy_constructible_v<SymqgIndex>);
static_assert(!std::is_copy_assignable_v<SymqgIndex>);
static_assert(std::is_nothrow_move_constructible_v<SymqgIndex>);
static_assert(std::is_nothrow_move_assignable_v<SymqgIndex>);
static_assert(!std::is_default_constructible_v<SymqgIndex>);

constexpr size_t kDim = 64;
constexpr size_t kNumPoints = 400;

std::vector<float> gaussian(size_t num_points, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> data(num_points * kDim);
    for (auto& value : data) {
        value = normal(rng);
    }
    return data;
}

SymqgOptions options(uint32_t seed) {
    SymqgOptions result;
    result.ef_construction = 40;
    result.seed = seed;
    return result;
}

// Throws exactly Error, not a subclass, with exactly this message.
template <typename Error, typename Call>
void expect_error(const Call& call, const std::string& message) {
    try {
        call();
        ADD_FAILURE() << "no exception; expected: " << message;
    } catch (const std::exception& error) {
        EXPECT_EQ(typeid(error), typeid(Error)) << error.what();
        EXPECT_EQ(error.what(), message);
    }
}

struct Results {
    std::vector<Id> ids;
    std::vector<float> distances;
};

Results search(
    const SymqgIndex& index,
    const float* queries,
    size_t num_queries,
    size_t k,
    const GraphSearchOptions& search_options = {}
) {
    Results results{
        std::vector<Id>(num_queries * k, 0), std::vector<float>(num_queries * k, 0.0F)};
    index.search(
        queries,
        num_queries,
        k,
        results.ids.data(),
        results.distances.data(),
        search_options
    );
    return results;
}

std::vector<char> file_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<char> saved_bytes(const SymqgIndex& index, const std::string& name) {
    const rabitq_test::TempFile file(name);
    index.save(file.path().c_str());
    return file_bytes(file.path());
}

TEST(SymqgIndexTest, DefaultsFollowThePythonClass) {
    const SymqgIndex index(kDim);
    EXPECT_EQ(index.dim(), kDim);
    EXPECT_EQ(index.nbits(), 32U);
    EXPECT_EQ(index.max_degree(), 32U);
    EXPECT_EQ(index.ef_construction(), std::optional<size_t>(200));
    EXPECT_EQ(index.init(), std::optional(QGInitialization::PiPNN));
    EXPECT_EQ(index.metric(), METRIC_L2);
    EXPECT_TRUE(index.seed().has_value());
    EXPECT_TRUE(index.is_trained());
    EXPECT_EQ(index.num_points(), 0U);
    EXPECT_EQ(index.num_stored_points(), 0U);

    SymqgOptions configured;
    configured.metric = METRIC_COSINE;
    configured.nbits = 4;
    configured.max_degree = 64;
    configured.ef_construction = 50;
    configured.init = QGInitialization::Random;
    configured.seed = 9;
    const SymqgIndex cosine(65536, configured);
    EXPECT_EQ(cosine.dim(), 65536U);
    EXPECT_EQ(cosine.metric(), METRIC_COSINE);
    EXPECT_EQ(cosine.nbits(), 4U);
    EXPECT_EQ(cosine.max_degree(), 64U);
    EXPECT_EQ(cosine.ef_construction(), std::optional<size_t>(50));
    EXPECT_EQ(cosine.init(), std::optional(QGInitialization::Random));
    EXPECT_EQ(cosine.seed(), std::optional<uint32_t>(9));
}

TEST(SymqgIndexTest, ConstructorRejectsInvalidSettings) {
    const auto invalid = [](size_t dim, auto change, const std::string& message) {
        SymqgOptions settings;
        change(settings);
        expect_error<std::invalid_argument>([&] { SymqgIndex(dim, settings); }, message);
    };
    const auto keep = [](SymqgOptions&) {};
    invalid(63, keep, "Unsupported dimension for FhtKacRotator");
    invalid(65537, keep, "Unsupported dimension for FhtKacRotator");
    for (size_t bad : {size_t{0}, size_t{48}}) {
        invalid(
            kDim,
            [&](SymqgOptions& o) { o.max_degree = bad; },
            "max_degree must be a positive multiple of 32"
        );
    }
    for (size_t bad : {size_t{0}, size_t{7}, size_t{16}}) {
        invalid(kDim, [&](SymqgOptions& o) { o.nbits = bad; }, "nbits must be 4, 8, or 32");
    }
    invalid(
        kDim,
        [](SymqgOptions& o) { o.init = static_cast<QGInitialization>(7); },
        "init must be PiPNN or Random"
    );
    for (size_t bad : {size_t{0}, size_t{std::numeric_limits<uint32_t>::max()} + 1}) {
        invalid(
            kDim,
            [&](SymqgOptions& o) { o.ef_construction = bad; },
            "ef_construction must be positive and fit uint32"
        );
    }
    invalid(
        kDim,
        [](SymqgOptions& o) { o.metric = static_cast<MetricType>(7); },
        "Metric type must be METRIC_L2 or METRIC_IP"
    );
    // The checks run in this order.
    SymqgOptions several;
    several.max_degree = 48;
    several.nbits = 16;
    several.ef_construction = 0;
    expect_error<std::invalid_argument>(
        [&] { SymqgIndex(10, several); }, "Unsupported dimension for FhtKacRotator"
    );
    expect_error<std::invalid_argument>(
        [&] { SymqgIndex(kDim, several); }, "max_degree must be a positive multiple of 32"
    );
    several.max_degree = 32;
    expect_error<std::invalid_argument>(
        [&] { SymqgIndex(kDim, several); }, "nbits must be 4, 8, or 32"
    );
}

TEST(SymqgIndexTest, UnsetSeedsAreDrawn) {
    // Four equal draws of 32 bits would be a coincidence of probability 2^-96.
    std::set<uint32_t> seeds;
    for (int i = 0; i < 4; ++i) {
        const SymqgIndex index(kDim);
        ASSERT_TRUE(index.seed().has_value());
        seeds.insert(*index.seed());
    }
    EXPECT_GT(seeds.size(), 1U);
}

TEST(SymqgIndexTest, AddBuildsTheGraphWithTheConstructorSettings) {
    const auto data = gaussian(kNumPoints, 1);
    SymqgOptions defaults;
    defaults.seed = 5;
    SymqgOptions quantized;
    quantized.metric = METRIC_COSINE;
    quantized.nbits = 4;
    quantized.max_degree = 64;
    quantized.ef_construction = 40;
    quantized.init = QGInitialization::Random;
    quantized.seed = 9;
    SymqgOptions eight_bits;
    eight_bits.metric = METRIC_IP;
    eight_bits.nbits = 8;
    eight_bits.ef_construction = 30;
    eight_bits.seed = 2;
    for (const auto& settings : {defaults, quantized, eight_bits}) {
        SCOPED_TRACE(settings.nbits);
        SymqgIndex index(kDim, settings);
        EXPECT_EQ(index.add(data.data(), kNumPoints, 1), 0);
        EXPECT_EQ(index.num_points(), kNumPoints);
        EXPECT_EQ(index.num_stored_points(), kNumPoints);

        // The file records every setting, so equal bytes mean they were all forwarded.
        symqg::QuantizedGraph<float> graph(
            kNumPoints,
            kDim,
            settings.max_degree,
            settings.metric,
            RotatorType::FhtKacRotator,
            settings.nbits == 32 ? 0 : settings.nbits,
            *settings.seed
        );
        symqg::QGBuilder(
            graph,
            static_cast<uint32_t>(settings.ef_construction),
            data.data(),
            1,
            settings.init,
            *settings.seed
        )
            .build();
        const rabitq_test::TempFile file("symqg_index_reference.index");
        graph.save(file.path().c_str());
        EXPECT_EQ(saved_bytes(index, "symqg_index_built.index"), file_bytes(file.path()));
    }
}

TEST(SymqgIndexTest, AddBuildsOnlyOnce) {
    const auto data = gaussian(kNumPoints, 2);
    SymqgIndex index(kDim, options(3));
    expect_error<std::invalid_argument>(
        [&] { index.add(nullptr, kNumPoints); }, "SymqgIndex add data must not be null"
    );
    index.add(data.data(), kNumPoints, 1);
    const auto before = search(index, data.data(), 2, 5);
    const std::string message =
        "SymqgIndex does not support incremental addition; create a new index and add "
        "all vectors at once";
    expect_error<std::logic_error>([&] { index.add(data.data(), 50); }, message);
    expect_error<std::logic_error>([&] { index.add(nullptr, 0); }, message);
    EXPECT_EQ(index.num_points(), kNumPoints);
    EXPECT_EQ(search(index, data.data(), 2, 5).ids, before.ids);
}

TEST(SymqgIndexTest, FailedAddLeavesTheIndexWithoutAGraph) {
    auto data = gaussian(kNumPoints, 3);
    SymqgIndex index(kDim, options(4));
    const std::string too_few =
        "QuantizedGraph degree bound must be smaller than the number of points";
    expect_error<std::invalid_argument>([&] { index.add(data.data(), 32, 1); }, too_few);
    expect_error<std::invalid_argument>([&] { index.add(nullptr, 0); }, too_few);
    // This failure comes after the new graph exists.
    data[7 * kDim + 1] = std::numeric_limits<float>::quiet_NaN();
    expect_error<std::invalid_argument>(
        [&] { index.add(data.data(), kNumPoints, 1); },
        "QuantizedGraph build data must contain only finite values"
    );
    EXPECT_EQ(index.num_points(), 0U);
    Id ids[1];
    expect_error<std::logic_error>(
        [&] { index.search(data.data(), 1, 1, ids, nullptr); },
        "SymqgIndex has no vectors; add them or load an index before search"
    );
    data[7 * kDim + 1] = 0.5F;
    EXPECT_EQ(index.add(data.data(), kNumPoints, 1), 0);
    EXPECT_EQ(index.num_points(), kNumPoints);
}

TEST(SymqgIndexTest, OperationsBeforeAddThrowLogicError) {
    const auto data = gaussian(4, 4);
    const SymqgIndex index(kDim, options(1));
    const rabitq_test::TempFile file("symqg_index_unbuilt.index");
    Id ids[1];
    const auto message = [](const std::string& operation) {
        return "SymqgIndex has no vectors; add them or load an index before " + operation;
    };
    expect_error<std::logic_error>(
        [&] { index.search(data.data(), 1, 1, ids, nullptr); }, message("search")
    );
    // The state is checked before the arguments.
    expect_error<std::logic_error>(
        [&] { index.search(data.data(), 1, 0, nullptr, nullptr); }, message("search")
    );
    expect_error<std::logic_error>(
        [&] { index.save(file.path().c_str()); }, message("save")
    );
    EXPECT_FALSE(std::filesystem::exists(file.path()));
}

TEST(SymqgIndexTest, TrainOnlyChecksItsInput) {
    auto data = gaussian(kNumPoints, 5);
    const SymqgIndex index(kDim, options(1));
    index.train(data.data(), kNumPoints);
    index.train(nullptr, 0);
    EXPECT_EQ(index.num_points(), 0U);
    expect_error<std::invalid_argument>(
        [&] { index.train(nullptr, 1); }, "SymqgIndex train data must not be null"
    );
    data[(kNumPoints - 1) * kDim + 3] = std::numeric_limits<float>::infinity();
    expect_error<std::invalid_argument>(
        [&] { index.train(data.data(), kNumPoints); },
        "SymqgIndex train data must contain only finite values"
    );
    data[(kNumPoints - 1) * kDim + 3] = 1e30F;
    expect_error<std::invalid_argument>(
        [&] { index.train(data.data(), kNumPoints); },
        "SymqgIndex train data exceeds the safe float32 coordinate range"
    );
}

TEST(SymqgIndexTest, DefaultEfSearchIsMaxOfKAnd64AndSmallerValuesAreRaisedToK) {
    constexpr size_t kMany = 4000;
    const auto data = gaussian(kMany, 6);
    SymqgOptions weak = options(3);
    weak.ef_construction = 8;
    weak.init = QGInitialization::Random;
    SymqgIndex index(kDim, weak);
    index.add(data.data(), kMany, 1);
    const auto queries = gaussian(200, 7);
    const auto with = [&](std::optional<size_t> ef_search, size_t k) {
        GraphSearchOptions search_options;
        search_options.ef_search = ef_search;
        return search(index, queries.data(), 200, k, search_options);
    };
    const auto by_default = with(std::nullopt, 10);
    EXPECT_EQ(by_default.ids, with(64, 10).ids);
    EXPECT_EQ(by_default.distances, with(64, 10).distances);
    EXPECT_NE(by_default.ids, with(10, 10).ids);
    EXPECT_NE(by_default.ids, with(200, 10).ids);
    // Above 64, the default is k itself, and a smaller breadth is raised to k.
    EXPECT_EQ(with(std::nullopt, 100).ids, with(100, 100).ids);
    EXPECT_EQ(with(5, 10).ids, with(10, 10).ids);
    EXPECT_EQ(with(1, 10).distances, with(10, 10).distances);
}

TEST(SymqgIndexTest, SearchRejectsZeroBreadthAndInvalidK) {
    const auto data = gaussian(kNumPoints, 8);
    SymqgIndex index(kDim, options(1));
    index.add(data.data(), kNumPoints, 1);
    std::vector<Id> ids(kNumPoints);
    GraphSearchOptions zero;
    zero.ef_search = 0;
    expect_error<std::invalid_argument>(
        [&] { index.search(data.data(), 1, 1, ids.data(), nullptr, zero); },
        "ef_search must be at least 1; leave it unset for the default"
    );
    for (size_t k : {size_t{0}, (size_t{1} << 31) + 1}) {
        expect_error<std::invalid_argument>(
            [&] { index.search(data.data(), 1, k, ids.data(), nullptr); },
            "k must be between 1 and 2^31"
        );
    }
    expect_error<std::invalid_argument>(
        [&] { index.search(data.data(), 1, 1, nullptr, nullptr); },
        "SymqgIndex search ids must not be null"
    );
    index.search(data.data(), 0, 1, nullptr, nullptr);  // an empty batch does nothing
}

TEST(SymqgIndexTest, SearchPadsMissingResults) {
    constexpr size_t kFew = 40;
    const auto data = gaussian(kFew, 9);
    SymqgIndex index(kDim, options(1));
    index.add(data.data(), kFew, 1);
    constexpr size_t kK = kFew + 5;
    const auto padded = search(index, data.data(), 2, kK);
    for (size_t query = 0; query < 2; ++query) {
        std::set<Id> found;
        for (size_t j = 0; j < kK; ++j) {
            const Id id = padded.ids[(query * kK) + j];
            const float distance = padded.distances[(query * kK) + j];
            if (j < kFew) {
                EXPECT_TRUE(id >= 0 && id < static_cast<Id>(kFew));
                EXPECT_TRUE(std::isfinite(distance));
                found.insert(id);
            } else {
                EXPECT_EQ(id, kNoId);
                EXPECT_TRUE(std::isinf(distance) && distance > 0);
            }
        }
        EXPECT_EQ(found.size(), kFew);
    }
    std::vector<Id> ids_only(2 * kK, 0);
    index.search(data.data(), 2, kK, ids_only.data(), nullptr);
    EXPECT_EQ(ids_only, padded.ids);
}

TEST(SymqgIndexTest, SaveAndLoadKeepSettingsSeedAndResults) {
    const auto data = gaussian(kNumPoints, 10);
    SymqgOptions configured = options(21);
    configured.metric = METRIC_COSINE;
    configured.nbits = 8;
    configured.ef_construction = 50;
    configured.init = QGInitialization::Random;
    SymqgIndex index(kDim, configured);
    index.add(data.data(), kNumPoints, 1);

    const rabitq_test::TempFile file("symqg_index_round_trip.index");
    index.save(file.path().c_str());
    SymqgIndex loaded = SymqgIndex::load(file.path().c_str());
    EXPECT_TRUE(loaded.is_trained());
    EXPECT_EQ(loaded.dim(), kDim);
    EXPECT_EQ(loaded.nbits(), 8U);
    EXPECT_EQ(loaded.max_degree(), 32U);
    EXPECT_EQ(loaded.metric(), METRIC_COSINE);
    EXPECT_EQ(loaded.ef_construction(), std::optional<size_t>(50));
    EXPECT_EQ(loaded.init(), std::optional(QGInitialization::Random));
    EXPECT_EQ(loaded.seed(), std::optional<uint32_t>(21));
    EXPECT_EQ(loaded.num_points(), kNumPoints);
    EXPECT_EQ(loaded.num_stored_points(), kNumPoints);

    const auto expected = search(index, data.data(), 3, 10);
    const auto actual = search(loaded, data.data(), 3, 10);
    EXPECT_EQ(actual.ids, expected.ids);
    EXPECT_EQ(actual.distances, expected.distances);
    expect_error<std::logic_error>(
        [&] { loaded.add(data.data(), kNumPoints); },
        "SymqgIndex does not support incremental addition; create a new index and add "
        "all vectors at once"
    );
}

TEST(SymqgIndexTest, LoadsFilesWithoutASeedOrConstructionSettings) {
    // Releases up to 0.8 recorded neither.
    const auto path =
        std::filesystem::path(RABITQ_TEST_FIXTURE_ROOT) / "v0.8.3" / "qg4_l2.index";
    const SymqgIndex old = SymqgIndex::load(path.string().c_str());
    EXPECT_FALSE(old.seed().has_value());
    EXPECT_FALSE(old.ef_construction().has_value());
    EXPECT_FALSE(old.init().has_value());
    EXPECT_TRUE(old.is_trained());
    EXPECT_EQ(old.nbits(), 4U);
    EXPECT_EQ(old.metric(), METRIC_L2);
    EXPECT_EQ(old.max_degree(), 32U);
    EXPECT_EQ(old.dim(), 65U);
    ASSERT_GT(old.num_points(), 0U);
    const std::vector<float> query(old.dim(), 0.5F);
    Id ids[1];
    old.search(query.data(), 1, 1, ids, nullptr);
    EXPECT_TRUE(ids[0] >= 0 && ids[0] < static_cast<Id>(old.num_points()));

    // A graph of raw vectors reports 32 bits.
    const SymqgIndex raw = SymqgIndex::load(
        (std::filesystem::path(RABITQ_TEST_FIXTURE_ROOT) / "v0.8.3" / "qg0_ip.index")
            .string()
            .c_str()
    );
    EXPECT_EQ(raw.nbits(), 32U);
    EXPECT_EQ(raw.metric(), METRIC_IP);
}

TEST(SymqgIndexTest, MovedIndexKeepsItsState) {
    const auto data = gaussian(kNumPoints, 11);
    SymqgIndex index(kDim, options(4));
    index.add(data.data(), kNumPoints, 1);
    const auto expected = search(index, data.data(), 2, 5);
    SymqgIndex moved(std::move(index));
    EXPECT_EQ(search(moved, data.data(), 2, 5).ids, expected.ids);
    SymqgIndex assigned(kDim, options(1));
    assigned = std::move(moved);
    EXPECT_EQ(assigned.seed(), std::optional<uint32_t>(4));
    EXPECT_EQ(assigned.num_points(), kNumPoints);
    EXPECT_EQ(search(assigned, data.data(), 2, 5).ids, expected.ids);
}

}  // namespace
}  // namespace rabitqlib
