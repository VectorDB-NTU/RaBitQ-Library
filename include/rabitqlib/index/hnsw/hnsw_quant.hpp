#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "rabitqlib/core/query.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"  // IWYU pragma: export
#include "rabitqlib/index/types.hpp"

namespace rabitqlib::hnsw {

namespace detail {
// Owned by one add() worker. Codes and index identity remain fixed for its lifetime.
// Reuse the prepared query while its ID is unchanged, including across insertions.
struct QuantQuery {
    PID id = kPidMax;
    std::vector<float> rotated;
    std::optional<SplitSingleQuery<float>> prepared;

    QuantQuery() = default;
    // prepared borrows rotated; copying would retain the source's storage pointer.
    QuantQuery(const QuantQuery&) = delete;
    QuantQuery& operator=(const QuantQuery&) = delete;

    QuantQuery(QuantQuery&& other) noexcept
        : id(std::exchange(other.id, kPidMax))
        , rotated(std::move(other.rotated))
        , prepared(std::move(other.prepared)) {
        other.prepared.reset();
    }

    QuantQuery& operator=(QuantQuery&& other) noexcept {
        if (this != &other) {
            // std::allocator-backed vector moves retain the borrowed data address.
            rotated = std::move(other.rotated);
            prepared = std::move(other.prepared);
            id = std::exchange(other.id, kPidMax);
            other.prepared.reset();
        }
        return *this;
    }
};
}  // namespace detail

}  // namespace rabitqlib::hnsw
