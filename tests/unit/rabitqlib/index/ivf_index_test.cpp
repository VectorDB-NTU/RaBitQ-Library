#include "rabitqlib/index/ivf_index.hpp"

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
#include "rabitqlib/index/ivf/types.hpp"
#include "rabitqlib/index/types.hpp"
#include "temp_file.hpp"

namespace rabitqlib {
namespace {

static_assert(!std::is_copy_constructible_v<IvfIndex>);
static_assert(!std::is_copy_assignable_v<IvfIndex>);
static_assert(std::is_nothrow_move_constructible_v<IvfIndex>);
static_assert(std::is_nothrow_move_assignable_v<IvfIndex>);
static_assert(!std::is_default_constructible_v<IvfIndex>);

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

IvfOptions seeded(uint32_t seed) {
    IvfOptions options;
    options.seed = seed;
    return options;
}

AddOptions single_thread() {
    AddOptions options;
    options.num_threads = 1;
    return options;
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
    const IvfIndex& index,
    const float* queries,
    size_t num_queries,
    size_t k,
    const IvfSearchOptions& options = {}
) {
    Results results{
        std::vector<Id>(num_queries * k, 0), std::vector<float>(num_queries * k, 0.0F)};
    index.search(
        queries, num_queries, k, results.ids.data(), results.distances.data(), options
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

std::vector<char> saved_bytes(const IvfIndex& index, const std::string& name) {
    const rabitq_test::TempFile file(name);
    index.save(file.path().c_str());
    std::ifstream input(file.path(), std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

TEST(IvfIndexTest, DefaultsFollowThePythonClass) {
    const IvfIndex index(kDim, kNumClusters);
    EXPECT_EQ(index.dim(), kDim);
    EXPECT_EQ(index.num_clusters(), kNumClusters);
    EXPECT_EQ(index.nbits(), 7U);
    EXPECT_EQ(index.metric(), METRIC_L2);
    EXPECT_EQ(index.routing(), ivf::Routing::Flat);  // Auto below 5,000 clusters
    ASSERT_TRUE(index.seed().has_value());
    EXPECT_FALSE(index.is_trained());
    EXPECT_EQ(index.num_points(), 0U);
    EXPECT_EQ(index.num_stored_points(), 0U);

    const auto& cp = index.cp();
    EXPECT_EQ(cp.niter, 10U);
    EXPECT_EQ(cp.max_points_per_centroid, 256U);
    EXPECT_FALSE(cp.spherical);
    EXPECT_EQ(cp.seed, *index.seed());

    IvfOptions inner_product;
    inner_product.metric = METRIC_IP;
    const IvfIndex ip(kDim, kNumClusters, inner_product);
    EXPECT_EQ(ip.metric(), METRIC_IP);
    EXPECT_TRUE(ip.cp().spherical);

    IvfOptions options;
    options.metric = METRIC_COSINE;
    options.nbits = 32;
    options.routing = ivf::Routing::HNSW;
    options.seed = 9;
    const IvfIndex configured(kDim, kNumClusters, options);
    EXPECT_EQ(configured.metric(), METRIC_COSINE);
    EXPECT_EQ(configured.nbits(), 32U);
    EXPECT_EQ(configured.routing(), ivf::Routing::HNSW);
    EXPECT_EQ(configured.seed(), std::optional<uint32_t>(9));
    EXPECT_EQ(configured.cp().seed, 9U);
    EXPECT_TRUE(configured.cp().spherical);
}

TEST(IvfIndexTest, UnsetSeedsAreDrawn) {
    // Four equal draws of 32 bits would be a coincidence of probability 2^-96.
    std::set<uint32_t> seeds;
    for (int i = 0; i < 4; ++i) {
        const IvfIndex index(kDim, kNumClusters);
        ASSERT_TRUE(index.seed().has_value());
        EXPECT_EQ(index.cp().seed, *index.seed());
        seeds.insert(*index.seed());
    }
    EXPECT_GT(seeds.size(), 1U);
}

TEST(IvfIndexTest, FirstAddTrainsLikeTrainThenAdd) {
    const auto data = gaussian(kNumPoints, 1);
    IvfIndex implicit(kDim, kNumClusters, seeded(7));
    IvfIndex explicit_train(kDim, kNumClusters, seeded(7));
    IvfIndex other_cp(kDim, kNumClusters, seeded(7));
    // The first add trains with cp(), like train, and rotates with the index seed.
    for (IvfIndex* index : {&implicit, &explicit_train}) {
        index->cp().niter = 3;
        index->cp().seed = 99;
    }

    EXPECT_EQ(implicit.add(data.data(), kNumPoints, single_thread()), 0);
    EXPECT_TRUE(implicit.is_trained());
    EXPECT_EQ(implicit.num_points(), kNumPoints);
    explicit_train.train(data.data(), kNumPoints, 1);
    EXPECT_EQ(explicit_train.add(data.data(), kNumPoints, single_thread()), 0);
    other_cp.add(data.data(), kNumPoints, single_thread());

    const auto bytes = saved_bytes(implicit, "ivf_index_implicit.index");
    EXPECT_EQ(bytes, saved_bytes(explicit_train, "ivf_index_explicit.index"));
    EXPECT_NE(bytes, saved_bytes(other_cp, "ivf_index_other_cp.index"));

    // Later batches continue the IDs.
    EXPECT_EQ(implicit.add(data.data(), 5, single_thread()), static_cast<Id>(kNumPoints));
    EXPECT_EQ(implicit.add(data.data(), 0), static_cast<Id>(kNumPoints + 5));
    EXPECT_EQ(implicit.num_stored_points(), kNumPoints + 5);
}

TEST(IvfIndexTest, FailedFirstAddLeavesTheIndexUntrainedAndEmpty) {
    auto data = gaussian(kNumPoints, 2);
    IvfIndex index(kDim, kNumClusters, seeded(11));
    expect_error<std::invalid_argument>(
        [&] { index.add(data.data(), kNumClusters - 1); },
        "the first add trains the index and needs at least num_clusters rows; "
        "expected at least 8, got shape (7, 64)"
    );
    data[5 * kDim] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(index.add(data.data(), kNumPoints), std::invalid_argument);
    EXPECT_FALSE(index.is_trained());
    EXPECT_EQ(index.num_stored_points(), 0U);
    EXPECT_EQ(index.seed(), std::optional<uint32_t>(11));

    data[5 * kDim] = 0.0F;
    EXPECT_EQ(index.add(data.data(), kNumPoints, single_thread()), 0);
    IvfIndex fresh(kDim, kNumClusters, seeded(11));
    fresh.add(data.data(), kNumPoints, single_thread());
    EXPECT_EQ(
        saved_bytes(index, "ivf_index_retried.index"),
        saved_bytes(fresh, "ivf_index_fresh.index")
    );
}

TEST(IvfIndexTest, OperationsBeforeTrainingThrowLogicError) {
    const auto data = gaussian(kNumPoints, 3);
    IvfIndex index(kDim, kNumClusters, seeded(1));
    const rabitq_test::TempFile file("ivf_index_untrained.index");
    const std::vector<PID> cluster_ids(kNumPoints, 0);
    AddOptions with_ids;
    with_ids.cluster_ids = cluster_ids.data();
    const Id remove_ids[] = {0};
    Id ids[1];

    const auto message = [](const std::string& operation) {
        return "IvfIndex must be trained or loaded before " + operation;
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
    expect_error<std::logic_error>(
        [&] { index.reserve_cluster(0, 10); }, message("reserve_cluster")
    );
    EXPECT_FALSE(std::filesystem::exists(file.path()));
    EXPECT_FALSE(index.is_trained());

    index.train(data.data(), kNumPoints);
    expect_error<std::logic_error>(
        [&] { index.save(file.path().c_str()); },
        "IvfIndex has no points to save; add points first"
    );
    EXPECT_FALSE(std::filesystem::exists(file.path()));
}

TEST(IvfIndexTest, RetrainingIsAllowedOnlyWhileEmpty) {
    const auto data = gaussian(kNumPoints, 4);
    IvfIndex index(kDim, kNumClusters, seeded(5));
    index.cp().num_threads = 3;
    index.train(data.data(), kNumPoints, 1);
    EXPECT_EQ(index.cp().num_threads, 3U);  // the override is for the call only
    index.set_centroids(data.data(), 1);
    index.train(data.data(), kNumPoints);
    index.add(data.data(), 50);

    const auto before = search(index, data.data(), 2, 5);
    const std::string message = "IvfIndex cannot be trained once it stores points";
    expect_error<std::logic_error>([&] { index.train(data.data(), kNumPoints); }, message);
    expect_error<std::logic_error>([&] { index.set_centroids(data.data()); }, message);
    const auto after = search(index, data.data(), 2, 5);
    EXPECT_EQ(before.ids, after.ids);
}

TEST(IvfIndexTest, DefaultNprobeIsOneSixteenthOfTheClustersRounded) {
    constexpr size_t kManyClusters = 24;  // 1.5 rounds to 2
    const auto data = gaussian(kNumPoints, 5);
    IvfIndex index(kDim, kManyClusters, seeded(3));
    index.add(data.data(), kNumPoints, single_thread());
    const auto probe = [&](std::optional<size_t> nprobe) {
        IvfSearchOptions options;
        options.nprobe = nprobe;
        return search(index, data.data(), 1, kNumPoints, options);
    };
    const auto by_default = probe(std::nullopt);
    const auto two = probe(2);
    EXPECT_EQ(by_default.ids, two.ids);
    EXPECT_EQ(by_default.distances, two.distances);
    EXPECT_LT(count_found(probe(1)), count_found(two));
    EXPECT_LT(count_found(two), count_found(probe(3)));

    // At least one cluster.
    IvfIndex small(kDim, 4, seeded(3));
    small.add(data.data(), kNumPoints, single_thread());
    IvfSearchOptions one;
    one.nprobe = 1;
    EXPECT_EQ(
        search(small, data.data(), 1, kNumPoints).ids,
        search(small, data.data(), 1, kNumPoints, one).ids
    );
}

TEST(IvfIndexTest, SearchRejectsZeroBreadthAndInvalidK) {
    const auto data = gaussian(kNumPoints, 6);
    IvfIndex index(kDim, kNumClusters, seeded(1));
    index.add(data.data(), kNumPoints, single_thread());
    std::vector<Id> ids(kNumPoints);
    IvfSearchOptions zero;
    zero.nprobe = 0;
    expect_error<std::invalid_argument>(
        [&] { index.search(data.data(), 1, 1, ids.data(), nullptr, zero); },
        "nprobe must be at least 1; leave it unset for the default"
    );
    for (size_t k : {size_t{0}, (size_t{1} << 31) + 1}) {
        expect_error<std::invalid_argument>(
            [&] { index.search(data.data(), 1, k, ids.data(), nullptr); },
            "k must be between 1 and 2^31"
        );
    }
    expect_error<std::invalid_argument>(
        [&] { index.search(data.data(), 1, 1, nullptr, nullptr); },
        "IvfIndex search ids must not be null"
    );
    index.search(data.data(), 0, 1, nullptr, nullptr);  // an empty batch does nothing
}

TEST(IvfIndexTest, SearchPadsMissingResults) {
    const auto data = gaussian(kNumPoints, 7);
    IvfIndex index(kDim, kNumClusters, seeded(1));
    index.train(data.data(), kNumPoints, 1);
    const auto empty = search(index, data.data(), 2, 3);
    EXPECT_EQ(count_found(empty), 0U);
    EXPECT_EQ(empty.ids, std::vector<Id>(6, kNoId));

    index.add(data.data(), 5);
    IvfSearchOptions every_cluster;
    every_cluster.nprobe = kNumClusters;
    const auto padded = search(index, data.data(), 1, 8, every_cluster);
    EXPECT_EQ(count_found(padded), 5U);
    for (size_t i = 5; i < 8; ++i) {
        EXPECT_EQ(padded.ids[i], kNoId);
    }
    std::vector<Id> ids_only(8, 0);
    index.search(data.data(), 1, 8, ids_only.data(), nullptr, every_cluster);
    EXPECT_EQ(ids_only, padded.ids);
}

TEST(IvfIndexTest, RemoveChecksIdsBeforeNarrowingThem) {
    const auto data = gaussian(kNumPoints, 8);
    IvfIndex index(kDim, kNumClusters, seeded(1));
    index.add(data.data(), kNumPoints, single_thread());
    const std::string message = "ids must be in [0, num_stored_points)";
    // (2^32 + 1) would remove point 1 if it were narrowed first.
    for (Id bad : {kNoId, static_cast<Id>(kNumPoints), (Id{1} << 32) + 1}) {
        const Id ids[] = {0, bad};
        expect_error<std::invalid_argument>([&] { index.remove(ids, 2); }, message);
    }
    EXPECT_EQ(index.num_points(), kNumPoints);
    expect_error<std::invalid_argument>(
        [&] { index.remove(nullptr, 1); }, "IvfIndex remove ids must not be null"
    );
    EXPECT_EQ(index.remove(nullptr, 0), 0U);

    const Id ids[] = {0, 1, 1};
    EXPECT_EQ(index.remove(ids, 3), 2U);
    EXPECT_EQ(index.num_points(), kNumPoints - 2);
    EXPECT_EQ(index.num_stored_points(), kNumPoints);
    IvfSearchOptions every_cluster;
    every_cluster.nprobe = kNumClusters;
    const auto results = search(index, data.data(), 2, kNumPoints, every_cluster);
    EXPECT_EQ(count_found(results), 2 * (kNumPoints - 2));
    for (const Id id : results.ids) {
        EXPECT_TRUE(id != 0 && id != 1);
    }
}

TEST(IvfIndexTest, ReserveClusterChecksTheCluster) {
    const auto data = gaussian(kNumPoints, 9);
    IvfIndex index(kDim, kNumClusters, seeded(1));
    index.add(data.data(), kNumPoints, single_thread());
    expect_error<std::invalid_argument>(
        [&] { index.reserve_cluster(kNumClusters, 10); },
        "cluster_id must be in [0, num_clusters)"
    );
    index.reserve_cluster(0, 1000);
    EXPECT_EQ(index.num_stored_points(), kNumPoints);
}

TEST(IvfIndexTest, SaveAndLoadKeepSettingsSeedAndResults) {
    const auto data = gaussian(kNumPoints, 10);
    IvfOptions options;
    options.metric = METRIC_COSINE;
    options.nbits = 4;
    options.seed = 21;
    IvfIndex index(kDim, kNumClusters, options);
    index.add(data.data(), kNumPoints);
    const Id removed[] = {3};
    index.remove(removed, 1);

    const rabitq_test::TempFile file("ivf_index_round_trip.index");
    index.save(file.path().c_str());
    const IvfIndex loaded = IvfIndex::load(file.path().c_str());
    EXPECT_TRUE(loaded.is_trained());
    EXPECT_EQ(loaded.dim(), kDim);
    EXPECT_EQ(loaded.num_clusters(), kNumClusters);
    EXPECT_EQ(loaded.nbits(), 4U);
    EXPECT_EQ(loaded.metric(), METRIC_COSINE);
    EXPECT_EQ(loaded.routing(), index.routing());
    EXPECT_EQ(loaded.seed(), std::optional<uint32_t>(21));
    EXPECT_EQ(loaded.cp().seed, 21U);
    EXPECT_TRUE(loaded.cp().spherical);
    EXPECT_EQ(loaded.num_points(), kNumPoints - 1);
    EXPECT_EQ(loaded.num_stored_points(), kNumPoints);

    const auto expected = search(index, data.data(), 3, 10);
    const auto actual = search(loaded, data.data(), 3, 10);
    EXPECT_EQ(actual.ids, expected.ids);
    EXPECT_EQ(actual.distances, expected.distances);
}

TEST(IvfIndexTest, AddAndSearchApplyTheirOptions) {
    const auto data = gaussian(kNumPoints, 12);
    const auto trained = [&] {
        IvfIndex index(kDim, kNumClusters, seeded(6));
        index.set_centroids(data.data(), 1);
        return index;
    };
    IvfIndex routed = trained();
    routed.add(data.data(), kNumPoints, single_thread());
    const auto routed_bytes = saved_bytes(routed, "ivf_index_routed.index");

    const std::vector<PID> first_cluster(kNumPoints, 0);
    AddOptions assigned = single_thread();
    assigned.cluster_ids = first_cluster.data();
    IvfIndex one_cluster = trained();
    one_cluster.add(data.data(), kNumPoints, assigned);
    EXPECT_NE(saved_bytes(one_cluster, "ivf_index_one_cluster.index"), routed_bytes);
    IvfSearchOptions first_probe;
    first_probe.nprobe = 1;
    // Every point is in cluster 0, so probing the one nearest cluster finds all or none.
    const size_t found =
        count_found(search(one_cluster, data.data(), 1, kNumPoints, first_probe));
    EXPECT_TRUE(found == 0 || found == kNumPoints) << found;

    AddOptions fast = single_thread();
    fast.fast_quantization = true;
    IvfIndex fast_index = trained();
    fast_index.add(data.data(), kNumPoints, fast);
    EXPECT_NE(saved_bytes(fast_index, "ivf_index_fast.index"), routed_bytes);

    IvfSearchOptions standard;
    standard.high_accuracy = false;
    IvfSearchOptions hacc;
    hacc.high_accuracy = true;
    EXPECT_NE(
        search(routed, data.data(), 20, 10, standard).distances,
        search(routed, data.data(), 20, 10, hacc).distances
    );
    EXPECT_EQ(
        search(routed, data.data(), 20, 10).distances,
        search(routed, data.data(), 20, 10, hacc).distances
    );  // 7 bits select HACC
}

TEST(IvfIndexTest, LoadsAFileWithoutASeed) {
    // Releases up to 0.8 did not record the seed; training defaults to the k-means seed.
    const auto path =
        std::filesystem::path(RABITQ_TEST_FIXTURE_ROOT) / "v0.8.3" / "ivf4_flat_l2.index";
    IvfIndex index = IvfIndex::load(path.string().c_str());
    EXPECT_TRUE(index.is_trained());
    EXPECT_FALSE(index.seed().has_value());
    EXPECT_EQ(index.cp().seed, rabitqkmeans::RaBitQKMeansParameters{}.seed);
    EXPECT_FALSE(index.cp().spherical);
    const size_t stored = index.num_stored_points();
    ASSERT_GT(stored, 0U);
    const std::vector<float> data(3 * index.dim(), 0.5F);
    EXPECT_EQ(index.add(data.data(), 3, single_thread()), static_cast<Id>(stored));
}

TEST(IvfIndexTest, MovedIndexKeepsItsState) {
    const auto data = gaussian(kNumPoints, 11);
    IvfIndex index(kDim, kNumClusters, seeded(4));
    index.add(data.data(), kNumPoints);
    const auto expected = search(index, data.data(), 2, 5);
    IvfIndex moved(std::move(index));
    EXPECT_EQ(search(moved, data.data(), 2, 5).ids, expected.ids);
    IvfIndex assigned(kDim, 2);
    assigned = std::move(moved);
    EXPECT_EQ(assigned.seed(), std::optional<uint32_t>(4));
    EXPECT_EQ(search(assigned, data.data(), 2, 5).ids, expected.ids);
}

}  // namespace
}  // namespace rabitqlib
