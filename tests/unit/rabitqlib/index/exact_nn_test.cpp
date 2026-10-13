#include "rabitqlib/index/detail/exact_nn.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/space.hpp"

using namespace rabitqlib;

TEST(IndexTypes, PreserveIdentifierAndCandidateRepresentation) {
    static_assert(std::is_same_v<PID, uint32_t>);
    EXPECT_EQ(kPidMax, std::numeric_limits<uint32_t>::max());
    const AnnCandidate<float> candidate(7, 1.5F);
    EXPECT_EQ(candidate.id, 7U);
    EXPECT_EQ(candidate.distance, 1.5F);
}

TEST(ExactNearestNeighbor, SupportsRequestedWorkerCounts) {
    const std::array<float, 12> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::array<float, 3> query{7, 8, 9};
    for (size_t workers : {0U, 1U, 2U, 64U}) {
        EXPECT_EQ(
            exact_nn(data.data(), query.data(), 4, 3, workers, euclidean_sqr<float>), 2U
        );
        EXPECT_EQ(
            exact_nn(data.data(), query.data(), 4, 3, workers, dot_product_dis<float>), 3U
        );
    }
}
