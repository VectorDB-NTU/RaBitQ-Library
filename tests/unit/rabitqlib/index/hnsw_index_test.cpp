#include "rabitqlib/index/hnsw_index.hpp"

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

#include "rabitqlib/clustering/rabitqkmeans.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/types.hpp"
#include "temp_file.hpp"

namespace rabitqlib {
namespace {

static_assert(!std::is_copy_constructible_v<HnswIndex>);
static_assert(!std::is_copy_assignable_v<HnswIndex>);
static_assert(std::is_nothrow_move_constructible_v<HnswIndex>);
static_assert(std::is_nothrow_move_assignable_v<HnswIndex>);
static_assert(!std::is_default_constructible_v<HnswIndex>);

constexpr size_t kDim = 64;
constexpr size_t kNumPoints = 400;
constexpr size_t kNumClusters = 8;

std::vector<float> gaussian(size_t num_points, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> data(num_points * kDim);
    for (auto& value : data) {
        value = normal(rng);
    }
    return data;
}

HnswOptions options(uint32_t seed, size_t num_clusters = kNumClusters) {
    HnswOptions result;
    result.num_clusters = num_clusters;
    result.seed = seed;
    return result;
}

AddOptions single_thread() {
    AddOptions result;
    result.num_threads = 1;
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
    const HnswIndex& index,
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

size_t count_found(const Results& results) {
    size_t found = 0;
    for (size_t i = 0; i < results.ids.size(); ++i) {
        if (results.ids[i] != kNoId) {
            EXPECT_TRUE(std::isfinite(results.distances[i]));
            ++found;
        } else {
            EXPECT_TRUE(std::isinf(results.distances[i]));
        }
    }
    return found;
}

std::vector<char> saved_bytes(const HnswIndex& index, const std::string& name) {
    const rabitq_test::TempFile file(name);
    index.save(file.path().c_str());
    std::ifstream input(file.path(), std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

TEST(HnswIndexTest, DefaultsFollowThePythonClass) {
    const HnswIndex index(kDim);
    EXPECT_EQ(index.dim(), kDim);
    EXPECT_EQ(index.num_clusters(), 16U);
    EXPECT_EQ(index.nbits(), 7U);
    EXPECT_EQ(index.M(), 16U);
    EXPECT_EQ(index.ef_construction(), 200U);
    EXPECT_EQ(index.metric(), METRIC_L2);
    EXPECT_EQ(index.capacity(), 1U);
    ASSERT_TRUE(index.seed().has_value());
    EXPECT_FALSE(index.is_trained());
    EXPECT_EQ(index.num_points(), 0U);
    EXPECT_EQ(index.num_stored_points(), 0U);

    const auto& cp = index.cp();
    EXPECT_EQ(cp.niter, 10U);
    EXPECT_EQ(cp.max_points_per_centroid, 256U);
    EXPECT_FALSE(cp.spherical);
    EXPECT_EQ(cp.seed, *index.seed());

    HnswOptions configured;
    configured.metric = METRIC_COSINE;
    configured.nbits = 4;
    configured.num_clusters = 5;
    configured.M = 8;
    configured.ef_construction = 50;
    configured.seed = 9;
    const HnswIndex cosine(kDim, configured);
    EXPECT_EQ(cosine.metric(), METRIC_COSINE);
    EXPECT_EQ(cosine.nbits(), 4U);
    EXPECT_EQ(cosine.num_clusters(), 5U);
    EXPECT_EQ(cosine.M(), 8U);
    EXPECT_EQ(cosine.ef_construction(), 50U);
    EXPECT_EQ(cosine.seed(), std::optional<uint64_t>(9));
    EXPECT_EQ(cosine.cp().seed, 9U);
    EXPECT_TRUE(cosine.cp().spherical);

    for (size_t bad : {size_t{0}, (size_t{1} << 31) + 1}) {
        HnswOptions invalid;
        invalid.num_clusters = bad;
        expect_error<std::invalid_argument>(
            [&] { HnswIndex(kDim, invalid); }, "num_clusters must be in [1, 2147483648]"
        );
    }
}

TEST(HnswIndexTest, UnsetSeedsAreDrawn) {
    // Four equal draws of 32 bits would be a coincidence of probability 2^-96.
    std::set<uint64_t> seeds;
    for (int i = 0; i < 4; ++i) {
        const HnswIndex index(kDim);
        ASSERT_TRUE(index.seed().has_value());
        EXPECT_EQ(index.cp().seed, *index.seed());
        seeds.insert(*index.seed());
    }
    EXPECT_GT(seeds.size(), 1U);
}

TEST(HnswIndexTest, FirstAddTrainsLikeTrainThenAdd) {
    const auto data = gaussian(kNumPoints, 1);
    HnswIndex implicit(kDim, options(7));
    HnswIndex explicit_train(kDim, options(7));
    HnswIndex other_cp(kDim, options(7));
    // The first add trains with cp(), like train, and builds with the index seed.
    for (HnswIndex* index : {&implicit, &explicit_train}) {
        index->cp().niter = 3;
        index->cp().seed = 99;
    }

    EXPECT_EQ(implicit.add(data.data(), kNumPoints, single_thread()), 0);
    EXPECT_TRUE(implicit.is_trained());
    EXPECT_EQ(implicit.num_points(), kNumPoints);
    EXPECT_EQ(implicit.num_clusters(), kNumClusters);
    explicit_train.train(data.data(), kNumPoints, 1);
    EXPECT_EQ(explicit_train.add(data.data(), kNumPoints, single_thread()), 0);
    other_cp.add(data.data(), kNumPoints, single_thread());

    const auto bytes = saved_bytes(implicit, "hnsw_index_implicit.index");
    EXPECT_EQ(bytes, saved_bytes(explicit_train, "hnsw_index_explicit.index"));
    EXPECT_NE(bytes, saved_bytes(other_cp, "hnsw_index_other_cp.index"));

    // Later batches continue the IDs.
    EXPECT_EQ(implicit.add(data.data(), 5, single_thread()), static_cast<Id>(kNumPoints));
    EXPECT_EQ(implicit.add(data.data(), 0), static_cast<Id>(kNumPoints + 5));
    EXPECT_EQ(implicit.num_stored_points(), kNumPoints + 5);
}

TEST(HnswIndexTest, FirstAddRejectsNullDataAndAppliesFastQuantization) {
    const auto data = gaussian(kNumPoints, 16);
    HnswIndex untrained(kDim, options(8));
    expect_error<std::invalid_argument>(
        [&] { untrained.add(nullptr, kNumPoints); }, "HnswIndex add data must not be null"
    );
    EXPECT_FALSE(untrained.is_trained());

    HnswIndex plain(kDim, options(8));
    plain.add(data.data(), kNumPoints, single_thread());
    AddOptions fast = single_thread();
    fast.fast_quantization = true;
    HnswIndex fast_index(kDim, options(8));
    fast_index.add(data.data(), kNumPoints, fast);
    EXPECT_NE(
        saved_bytes(fast_index, "hnsw_index_first_fast.index"),
        saved_bytes(plain, "hnsw_index_first_plain.index")
    );
}

TEST(HnswIndexTest, FirstAddWithFewPointsTrainsOneCentroidPerPoint) {
    const auto data = gaussian(4, 2);
    HnswIndex index(kDim, options(5));
    EXPECT_EQ(index.add(data.data(), 4, single_thread()), 0);
    EXPECT_TRUE(index.is_trained());
    EXPECT_EQ(index.num_clusters(), 4U);
}

TEST(HnswIndexTest, FailedFirstAddLeavesTheIndexUntrainedEmptyAndItsCapacity) {
    auto data = gaussian(kNumPoints, 3);
    HnswIndex index(kDim, options(11));
    index.reserve(1000);
    data[5 * kDim] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(index.add(data.data(), kNumPoints), std::invalid_argument);
    EXPECT_FALSE(index.is_trained());
    EXPECT_EQ(index.num_stored_points(), 0U);
    EXPECT_EQ(index.num_clusters(), kNumClusters);
    EXPECT_EQ(index.capacity(), 1000U);
    EXPECT_EQ(index.seed(), std::optional<uint64_t>(11));

    data[5 * kDim] = 0.0F;
    EXPECT_EQ(index.add(data.data(), kNumPoints, single_thread()), 0);
    EXPECT_EQ(index.capacity(), 1000U);  // the first add keeps a reserved capacity
    HnswIndex fresh(kDim, options(11));
    fresh.reserve(1000);
    fresh.add(data.data(), kNumPoints, single_thread());
    EXPECT_EQ(
        saved_bytes(index, "hnsw_index_retried.index"),
        saved_bytes(fresh, "hnsw_index_fresh.index")
    );
}

TEST(HnswIndexTest, CapacityGrowsByAtLeastDoublingAndReserveOnlyGrows) {
    auto data = gaussian(kNumPoints, 4);
    HnswIndex index(kDim, options(1));
    index.train(data.data(), kNumPoints, 1);
    index.add(data.data(), 100, single_thread());
    EXPECT_EQ(index.capacity(), 100U);
    index.add(data.data() + (100 * kDim), 250, single_thread());
    EXPECT_EQ(index.capacity(), 350U);  // what the points need, above double
    index.add(data.data() + (350 * kDim), 10, single_thread());
    EXPECT_EQ(index.capacity(), 700U);  // doubled
    constexpr size_t kStored = 360;

    // Invalid points leave the capacity and the points unchanged.
    data[3 * kDim] = std::numeric_limits<float>::infinity();
    EXPECT_THROW(index.add(data.data(), 400), std::invalid_argument);
    EXPECT_EQ(index.capacity(), 700U);
    EXPECT_EQ(index.num_points(), kStored);

    // So do invalid cluster IDs and a missing data pointer.
    const std::vector<PID> out_of_range(400, kNumClusters);
    AddOptions bad_ids = single_thread();
    bad_ids.cluster_ids = out_of_range.data();
    data[3 * kDim] = 0.0F;
    expect_error<std::invalid_argument>(
        [&] { index.add(data.data(), 400, bad_ids); }, "HNSW cluster ID is out of range"
    );
    expect_error<std::invalid_argument>(
        [&] { index.add(nullptr, 400); }, "HnswIndex add data must not be null"
    );
    EXPECT_EQ(index.capacity(), 700U);
    EXPECT_EQ(index.num_points(), kStored);

    for (size_t smaller : {size_t{0}, size_t{10}, kStored, size_t{700}}) {
        index.reserve(smaller);
        EXPECT_EQ(index.capacity(), 700U);
    }
    index.reserve(1000);
    EXPECT_EQ(index.capacity(), 1000U);
    EXPECT_EQ(index.num_points(), kStored);
}

TEST(HnswIndexTest, OperationsBeforeTrainingThrowLogicError) {
    const auto data = gaussian(kNumPoints, 5);
    HnswIndex index(kDim, options(1));
    const rabitq_test::TempFile file("hnsw_index_untrained.index");
    const std::vector<PID> cluster_ids(kNumPoints, 0);
    AddOptions with_ids;
    with_ids.cluster_ids = cluster_ids.data();
    const Id remove_ids[] = {0};
    Id ids[1];

    const auto message = [](const std::string& operation) {
        return "HnswIndex must be trained or loaded before " + operation;
    };
    // With cluster IDs, or without points, add needs centroids instead of training.
    expect_error<std::logic_error>(
        [&] { index.add(data.data(), kNumPoints, with_ids); }, message("add")
    );
    expect_error<std::logic_error>([&] { index.add(data.data(), 0); }, message("add"));
    expect_error<std::logic_error>(
        [&] { index.search(data.data(), 1, 1, ids, nullptr); }, message("search")
    );
    expect_error<std::logic_error>([&] { index.remove(remove_ids, 1); }, message("remove"));
    expect_error<std::logic_error>(
        [&] { index.save(file.path().c_str()); }, message("save")
    );
    EXPECT_FALSE(std::filesystem::exists(file.path()));
    index.reserve(50);  // allowed before training
    EXPECT_EQ(index.capacity(), 50U);
    EXPECT_FALSE(index.is_trained());

    index.train(data.data(), kNumPoints, 1);
    expect_error<std::logic_error>(
        [&] { index.save(file.path().c_str()); },
        "HnswIndex has no points to save; add points first"
    );
    EXPECT_FALSE(std::filesystem::exists(file.path()));
}

TEST(HnswIndexTest, RetrainingIsAllowedOnlyWhileEmpty) {
    const auto data = gaussian(kNumPoints, 6);
    HnswIndex index(kDim, options(5));
    index.cp().num_threads = 3;
    index.train(data.data(), kNumPoints, 1);
    EXPECT_EQ(index.cp().num_threads, 3U);  // the override is for the call only
    index.set_centroids(data.data());
    index.train(data.data(), kNumPoints, 1);
    index.add(data.data(), 50, single_thread());

    const auto before = search(index, data.data(), 2, 5);
    const std::string message = "HnswIndex cannot be trained once it stores points";
    expect_error<std::logic_error>([&] { index.train(data.data(), kNumPoints); }, message);
    expect_error<std::logic_error>([&] { index.set_centroids(data.data()); }, message);
    EXPECT_EQ(search(index, data.data(), 2, 5).ids, before.ids);
}

TEST(HnswIndexTest, DefaultEfSearchIsKOrAtLeast64) {
    constexpr size_t kMany = 4000;
    const auto data = gaussian(kMany, 7);
    HnswOptions weak = options(3, 4);
    weak.M = 4;
    weak.ef_construction = 8;
    HnswIndex index(kDim, weak);
    index.add(data.data(), kMany, single_thread());
    const auto queries = gaussian(200, 8);
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
    // Above 64, the default is k itself.
    EXPECT_EQ(with(std::nullopt, 100).ids, with(100, 100).ids);
}

TEST(HnswIndexTest, SearchRejectsZeroBreadthAndInvalidK) {
    const auto data = gaussian(kNumPoints, 9);
    HnswIndex index(kDim, options(1));
    index.add(data.data(), kNumPoints, single_thread());
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
        "HnswIndex search ids must not be null"
    );
    index.search(data.data(), 0, 1, nullptr, nullptr);  // an empty batch does nothing
}

TEST(HnswIndexTest, SearchPadsMissingResults) {
    const auto data = gaussian(kNumPoints, 10);
    HnswIndex index(kDim, options(1));
    index.train(data.data(), kNumPoints, 1);
    const auto empty = search(index, data.data(), 2, 3);
    EXPECT_EQ(count_found(empty), 0U);
    EXPECT_EQ(empty.ids, std::vector<Id>(6, kNoId));

    index.add(data.data(), 5, single_thread());
    const auto padded = search(index, data.data(), 1, 8);
    EXPECT_EQ(count_found(padded), 5U);
    for (size_t i = 5; i < 8; ++i) {
        EXPECT_EQ(padded.ids[i], kNoId);
    }
    std::vector<Id> ids_only(8, 0);
    index.search(data.data(), 1, 8, ids_only.data(), nullptr);
    EXPECT_EQ(ids_only, padded.ids);
}

TEST(HnswIndexTest, RemoveChecksIdsBeforeNarrowingThem) {
    const auto data = gaussian(kNumPoints, 11);
    HnswIndex index(kDim, options(1));
    index.add(data.data(), kNumPoints, single_thread());
    const std::string message = "ids must be in [0, num_stored_points)";
    // (2^32 + 1) would remove point 1 if it were narrowed first.
    for (Id bad : {kNoId, static_cast<Id>(kNumPoints), (Id{1} << 32) + 1}) {
        const Id ids[] = {0, bad};
        expect_error<std::invalid_argument>([&] { index.remove(ids, 2); }, message);
    }
    EXPECT_EQ(index.num_points(), kNumPoints);
    expect_error<std::invalid_argument>(
        [&] { index.remove(nullptr, 1); }, "HnswIndex remove ids must not be null"
    );
    EXPECT_EQ(index.remove(nullptr, 0), 0U);

    const Id ids[] = {0, 1, 1};
    EXPECT_EQ(index.remove(ids, 3), 2U);
    EXPECT_EQ(index.num_points(), kNumPoints - 2);
    EXPECT_EQ(index.num_stored_points(), kNumPoints);
    GraphSearchOptions wide;
    wide.ef_search = kNumPoints;
    const auto results = search(index, data.data(), 2, kNumPoints, wide);
    for (const Id id : results.ids) {
        EXPECT_TRUE(id != 0 && id != 1);
    }
}

TEST(HnswIndexTest, AddAndSearchApplyTheirOptions) {
    const auto data = gaussian(kNumPoints, 12);
    const auto trained = [&] {
        HnswIndex index(kDim, options(6));
        index.set_centroids(data.data());
        return index;
    };
    HnswIndex routed = trained();
    routed.add(data.data(), kNumPoints, single_thread());
    const auto routed_bytes = saved_bytes(routed, "hnsw_index_routed.index");

    const std::vector<PID> first_cluster(kNumPoints, 0);
    AddOptions assigned = single_thread();
    assigned.cluster_ids = first_cluster.data();
    HnswIndex one_cluster = trained();
    one_cluster.add(data.data(), kNumPoints, assigned);
    EXPECT_NE(saved_bytes(one_cluster, "hnsw_index_one_cluster.index"), routed_bytes);

    AddOptions fast = single_thread();
    fast.fast_quantization = true;
    HnswIndex fast_index = trained();
    fast_index.add(data.data(), kNumPoints, fast);
    EXPECT_NE(saved_bytes(fast_index, "hnsw_index_fast.index"), routed_bytes);
}

TEST(HnswIndexTest, SaveAndLoadKeepSettingsSeedAndResults) {
    const auto data = gaussian(kNumPoints, 13);
    HnswOptions configured = options(21);
    configured.metric = METRIC_COSINE;
    configured.nbits = 4;
    configured.M = 8;
    HnswIndex index(kDim, configured);
    index.add(data.data(), kNumPoints, single_thread());
    const Id removed[] = {3};
    index.remove(removed, 1);

    const rabitq_test::TempFile file("hnsw_index_round_trip.index");
    index.save(file.path().c_str());
    const HnswIndex loaded = HnswIndex::load(file.path().c_str());
    EXPECT_TRUE(loaded.is_trained());
    EXPECT_EQ(loaded.dim(), kDim);
    EXPECT_EQ(loaded.num_clusters(), kNumClusters);
    EXPECT_EQ(loaded.nbits(), 4U);
    EXPECT_EQ(loaded.M(), 8U);
    EXPECT_EQ(loaded.ef_construction(), 200U);
    EXPECT_EQ(loaded.metric(), METRIC_COSINE);
    EXPECT_EQ(loaded.seed(), std::optional<uint64_t>(21));
    EXPECT_EQ(loaded.cp().seed, 21U);
    EXPECT_TRUE(loaded.cp().spherical);
    EXPECT_EQ(loaded.num_points(), kNumPoints - 1);
    EXPECT_EQ(loaded.num_stored_points(), kNumPoints);

    const auto expected = search(index, data.data(), 3, 10);
    const auto actual = search(loaded, data.data(), 3, 10);
    EXPECT_EQ(actual.ids, expected.ids);
    EXPECT_EQ(actual.distances, expected.distances);
}

TEST(HnswIndexTest, LoadsAnUntrainedFileSavedBy07) {
    // The C++ API of releases before 0.8 could save an index before training it. The
    // file has no centroids and no seed, so the first add cannot train it.
    constexpr size_t kFileDim = 65;
    const auto path =
        std::filesystem::path(RABITQ_TEST_FIXTURE_ROOT) / "v0.7.2" / "hnsw4_empty_l2.index";
    HnswIndex index = HnswIndex::load(path.string().c_str());
    EXPECT_FALSE(index.is_trained());
    EXPECT_FALSE(index.seed().has_value());
    EXPECT_EQ(index.dim(), kFileDim);
    EXPECT_EQ(index.nbits(), 4U);
    EXPECT_EQ(index.metric(), METRIC_L2);
    EXPECT_EQ(index.capacity(), 130U);
    EXPECT_EQ(index.num_points(), 0U);
    EXPECT_EQ(index.num_clusters(), 16U);

    std::mt19937 rng(21);
    std::normal_distribution<float> normal;
    std::vector<float> data(kNumPoints * kFileDim);
    for (auto& value : data) {
        value = normal(rng);
    }
    const auto message = [](const std::string& operation) {
        return "HnswIndex loaded from an untrained file must be trained before " +
               operation + "; call train or set_centroids";
    };
    expect_error<std::logic_error>(
        [&] { index.add(data.data(), kNumPoints, single_thread()); }, message("add")
    );
    Id ids[1];
    const Id remove_ids[] = {0};
    expect_error<std::logic_error>(
        [&] { index.search(data.data(), 1, 1, ids, nullptr); }, message("search")
    );
    expect_error<std::logic_error>([&] { index.remove(remove_ids, 1); }, message("remove"));
    const rabitq_test::TempFile file("hnsw_index_untrained_07.index");
    expect_error<std::logic_error>(
        [&] { index.save(file.path().c_str()); }, message("save")
    );
    EXPECT_EQ(index.num_points(), 0U);

    index.train(data.data(), kNumPoints, 1);
    EXPECT_EQ(index.num_clusters(), 16U);
    EXPECT_EQ(index.add(data.data(), kNumPoints, single_thread()), 0);
    const auto expected = search(index, data.data(), 3, 5);
    EXPECT_EQ(count_found(expected), 15U);
    index.save(file.path().c_str());
    const HnswIndex reloaded = HnswIndex::load(file.path().c_str());
    EXPECT_TRUE(reloaded.is_trained());
    EXPECT_EQ(reloaded.num_points(), kNumPoints);
    EXPECT_EQ(reloaded.num_clusters(), 16U);
    EXPECT_EQ(search(reloaded, data.data(), 3, 5).ids, expected.ids);

    // Setting centroids makes it trained as well.
    HnswIndex other = HnswIndex::load(path.string().c_str());
    other.set_centroids(data.data());  // the first 16 points
    EXPECT_EQ(other.add(data.data(), kNumPoints, single_thread()), 0);
    EXPECT_EQ(other.num_points(), kNumPoints);
}

TEST(HnswIndexTest, LoadsFilesWithoutASeedAndWithA64BitSeed) {
    // Releases up to 0.8 did not record the seed; training defaults to the k-means seed.
    const auto path =
        std::filesystem::path(RABITQ_TEST_FIXTURE_ROOT) / "v0.8.3" / "hnsw1_l2.index";
    HnswIndex old = HnswIndex::load(path.string().c_str());
    EXPECT_TRUE(old.is_trained());
    EXPECT_FALSE(old.seed().has_value());
    EXPECT_EQ(old.cp().seed, rabitqkmeans::RaBitQKMeansParameters{}.seed);
    const size_t stored = old.num_stored_points();
    ASSERT_GT(stored, 0U);
    const std::vector<float> point(old.dim(), 0.5F);
    EXPECT_EQ(old.add(point.data(), 1, single_thread()), static_cast<Id>(stored));

    // The C++ HierarchicalNSW class can record a seed above 32 bits; cp takes its low
    // 32 bits.
    constexpr uint64_t kWideSeed = (uint64_t{1} << 40) + 5;
    const auto data = gaussian(kNumPoints, 14);
    hnsw::HierarchicalNSW wide(kNumPoints, kDim, 7, 16, 200, kWideSeed);
    wide.train(
        kNumClusters, kNumPoints, data.data(), rabitqkmeans::RaBitQKMeansParameters{}
    );
    wide.add(data.data(), kNumPoints);
    const rabitq_test::TempFile file("hnsw_index_wide_seed.index");
    wide.save(file.path().c_str());
    const HnswIndex loaded = HnswIndex::load(file.path().c_str());
    EXPECT_EQ(loaded.seed(), std::optional<uint64_t>(kWideSeed));
    EXPECT_EQ(loaded.cp().seed, 5U);
}

TEST(HnswIndexTest, MovedIndexKeepsItsState) {
    const auto data = gaussian(kNumPoints, 15);
    HnswIndex index(kDim, options(4));
    index.add(data.data(), kNumPoints, single_thread());
    const auto expected = search(index, data.data(), 2, 5);
    HnswIndex moved(std::move(index));
    EXPECT_EQ(search(moved, data.data(), 2, 5).ids, expected.ids);
    HnswIndex assigned(kDim, options(1, 2));
    assigned = std::move(moved);
    EXPECT_EQ(assigned.seed(), std::optional<uint64_t>(4));
    EXPECT_EQ(search(assigned, data.data(), 2, 5).ids, expected.ids);
}

}  // namespace
}  // namespace rabitqlib
