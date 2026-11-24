#pragma once

#include "rabitqlib/index/types.hpp"

namespace rabitqlib::index_detail {

// Flat routing orders by the returned float distance, including sqrt rounding
// for L2, then by centroid ID. Use the same ordering for scalar and batched search.
struct CandidateDistanceIdLess {
    bool operator()(const AnnCandidate<float>& a, const AnnCandidate<float>& b) const {
        return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
    }
};

}  // namespace rabitqlib::index_detail
