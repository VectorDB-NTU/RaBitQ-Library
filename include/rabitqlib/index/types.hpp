#pragma once

#include <cstdint>
#include <limits>

namespace rabitqlib {
using PID = uint32_t;
inline constexpr PID kPidMax = 0xFFFFFFFF;

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
