#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/memory.hpp"

namespace rabitqlib {
/**
 * @brief Set of visited vertices, backed by a generation-stamped array.
 *
 * One byte-pair per vertex holds the generation in which that vertex was last
 * marked; a vertex is "visited" iff its stamp equals the current generation.
 * clear() therefore just bumps the generation -- O(1), and only the rare
 * 16-bit wraparound pays for an actual refill.
 *
 * This replaces HashBasedVisitedSet (visited_set_hash.hpp), whose direct-mapped
 * table was sized at max_elements/10 rounded down to a power of two, so a
 * single ef=100 search collided on nearly every visit and allocated an
 * unordered_set node each time: measured at ~1200 mallocs per query, 98% of all
 * allocation on the search path. Removing them was worth ~1.6x on HNSW search.
 * The trade is memory: two bytes per id in the whole id space, resident for the
 * lifetime of the set. See visited_set.hpp for how one of the two is selected.
 *
 * The constructor argument is the ID SPACE (number of vertices) -- ids are used
 * to index directly, so it must cover every id passed to get()/set().
 */
class EpochBasedVisitedSet {
   private:
    std::vector<uint16_t, memory::AlignedAllocator<uint16_t>> stamp_;
    uint16_t cur_ = 0;

   public:
    EpochBasedVisitedSet() = default;
    ~EpochBasedVisitedSet() = default;

    EpochBasedVisitedSet(const EpochBasedVisitedSet&) = default;
    EpochBasedVisitedSet& operator=(const EpochBasedVisitedSet&) = default;
    EpochBasedVisitedSet(EpochBasedVisitedSet&&) noexcept = default;
    EpochBasedVisitedSet& operator=(EpochBasedVisitedSet&&) noexcept = default;

    explicit EpochBasedVisitedSet(size_t num_elements) { initialize(num_elements); }

    void initialize(size_t num_elements) {
        stamp_.assign(num_elements, 0);
        cur_ = 0;
        // Leaves the set usable without an explicit clear(): every stamp is 0
        // while the live generation is 1, so nothing reads as visited.
        clear();
    }

    void clear() {
        if (++cur_ == 0) {
            std::fill(stamp_.begin(), stamp_.end(), 0);
            cur_ = 1;
        }
    }

    // get if data_id is in the visited set
    [[nodiscard]] bool get(PID data_id) const { return stamp_[data_id] == cur_; }

    void set(PID data_id) { stamp_[data_id] = cur_; }
};
}  // namespace rabitqlib
