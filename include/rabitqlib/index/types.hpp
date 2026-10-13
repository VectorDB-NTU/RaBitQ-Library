#pragma once

#include <cstdint>
#include <limits>

namespace rabitqlib {
using PID = uint32_t;
constexpr uint32_t kPidMax = 0xFFFFFFFF;

template <typename T, typename TP = PID>
struct AnnCandidate {
    TP id = 0;
    T distance = std::numeric_limits<T>::max();

    AnnCandidate() = default;
    explicit AnnCandidate(TP vec_id, T dis) : id(vec_id), distance(dis) {}

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
