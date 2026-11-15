#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace rabitqlib {
using PID = uint32_t;
inline constexpr PID kPidMax = 0xFFFFFFFF;

// A point ID at the IvfIndex API: the row number the index assigned to the point.
using Id = int64_t;
// A search result slot without a point.
inline constexpr Id kNoId = -1;

// Settings of IvfIndex::add.
struct AddOptions {
    // Cluster of each point, or nullptr to route each point with the index's routing,
    // which may be approximate.
    const PID* cluster_ids = nullptr;
    // Worker count. No value trains an untrained index with cp().num_threads and adds
    // with every available CPU; 0 selects every available CPU, capped by OMP_NUM_THREADS.
    std::optional<size_t> num_threads;
    // Fixed-scale extra-bit quantization: faster, possibly less accurate. No effect on
    // one-bit or raw storage.
    bool fast_quantization = false;
};

template <typename T, typename TP = PID>
struct AnnCandidate {
    TP id = 0;
    T distance = std::numeric_limits<T>::max();

    AnnCandidate() = default;
    explicit AnnCandidate(TP candidate_id, T candidate_distance)
        : id(candidate_id), distance(candidate_distance) {}

    friend bool operator<(const AnnCandidate& first, const AnnCandidate& second) {
        return first.distance < second.distance;
    }
    friend bool operator>(const AnnCandidate& first, const AnnCandidate& second) {
        return first.distance > second.distance;
    }
    friend bool operator>=(const AnnCandidate& first, const AnnCandidate& second) {
        return first.distance >= second.distance;
    }
    friend bool operator<=(const AnnCandidate& first, const AnnCandidate& second) {
        return first.distance <= second.distance;
    }
};

}  // namespace rabitqlib
