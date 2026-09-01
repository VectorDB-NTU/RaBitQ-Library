#include "rabitqlib/utils/visited_set.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

#include "rabitqlib/utils/hashset.hpp"

namespace {

using rabitqlib::EpochBasedVisitedSet;
using rabitqlib::HashBasedBooleanSet;
using rabitqlib::HashBasedVisitedSet;
using rabitqlib::PID;
using rabitqlib::VisitedSet;

static_assert(
    std::is_same_v<VisitedSet, HashBasedVisitedSet>,
    "HashBasedVisitedSet must remain the default visited-set implementation"
);
static_assert(
    std::is_same_v<HashBasedBooleanSet, HashBasedVisitedSet>,
    "HashBasedBooleanSet must remain available as a compatibility alias"
);

template <typename Set>
void expect_basic_visited_set_behavior() {
    constexpr size_t kNumElements = 1024;
    Set visited(kNumElements, 8);

    EXPECT_FALSE(visited.get(0));
    EXPECT_FALSE(visited.get(kNumElements - 1));

    visited.set(0);
    visited.set(17);
    visited.set(kNumElements - 1);

    EXPECT_TRUE(visited.get(0));
    EXPECT_TRUE(visited.get(17));
    EXPECT_TRUE(visited.get(kNumElements - 1));

    visited.set(17);
    EXPECT_TRUE(visited.get(17));

    visited.clear();
    EXPECT_FALSE(visited.get(0));
    EXPECT_FALSE(visited.get(17));
    EXPECT_FALSE(visited.get(kNumElements - 1));

    visited.set(23);
    EXPECT_TRUE(visited.get(23));
}

TEST(VisitedSet, HashBackendSatisfiesContract) {
    expect_basic_visited_set_behavior<HashBasedVisitedSet>();
}

TEST(VisitedSet, EpochBackendSatisfiesContract) {
    expect_basic_visited_set_behavior<EpochBasedVisitedSet>();
}

TEST(VisitedSet, BackendsRemainBehaviorallyEquivalent) {
    constexpr size_t kNumElements = 4096;
    constexpr size_t kNumOperations = 20000;

    HashBasedVisitedSet hash_visited(kNumElements, 16);
    EpochBasedVisitedSet epoch_visited(kNumElements);
    std::vector<bool> expected(kNumElements, false);
    std::mt19937 generator(42);
    std::uniform_int_distribution<PID> id_distribution(0, kNumElements - 1);

    for (size_t operation = 0; operation < kNumOperations; ++operation) {
        if (operation % 257 == 0) {
            hash_visited.clear();
            epoch_visited.clear();
            std::fill(expected.begin(), expected.end(), false);
        } else {
            const PID id = id_distribution(generator);
            if (operation % 3 == 0) {
                EXPECT_EQ(hash_visited.get(id), expected[id]);
                EXPECT_EQ(epoch_visited.get(id), expected[id]);
            } else {
                hash_visited.set(id);
                epoch_visited.set(id);
                expected[id] = true;
            }
        }
    }

    for (PID id = 0; id < kNumElements; ++id) {
        EXPECT_EQ(hash_visited.get(id), expected[id]) << "id=" << id;
        EXPECT_EQ(epoch_visited.get(id), expected[id]) << "id=" << id;
    }
}

TEST(HashBasedVisitedSet, HandlesCollidingIds) {
    HashBasedVisitedSet visited(1024, 1);
    const std::vector<PID> colliding_ids{0, 32, 64, 96, 128, 1023};

    for (PID id : colliding_ids) {
        visited.set(id);
    }
    for (PID id : colliding_ids) {
        EXPECT_TRUE(visited.get(id)) << "id=" << id;
    }

    visited.clear();
    for (PID id : colliding_ids) {
        EXPECT_FALSE(visited.get(id)) << "id=" << id;
    }
}

TEST(EpochBasedVisitedSet, ClearsStaleValuesWhenEpochWraps) {
    EpochBasedVisitedSet visited(4);
    visited.set(0);

    // Construction starts at epoch 1. Advance to the last uint16_t epoch.
    for (size_t epoch = 0; epoch < UINT16_MAX - 1; ++epoch) {
        visited.clear();
    }
    visited.set(1);
    EXPECT_TRUE(visited.get(1));

    // The next clear wraps the counter and must reset all stored stamps.
    visited.clear();
    EXPECT_FALSE(visited.get(0));
    EXPECT_FALSE(visited.get(1));

    visited.set(3);
    EXPECT_TRUE(visited.get(3));
}

}  // namespace
