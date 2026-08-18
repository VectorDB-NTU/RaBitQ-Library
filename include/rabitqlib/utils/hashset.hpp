// This code is modified based on NGT from Yahoo Japan
// https://github.com/yahoojapan/NGT
//
// Copyright (C) 2015 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

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
 * This replaces a direct-mapped table with an std::unordered_set overflow. That
 * table was sized at max_elements/10 rounded down to a power of two, so a
 * single ef=100 search collided on nearly every visit and allocated an
 * unordered_set node each time: measured at ~1200 mallocs per query, 98% of all
 * allocation on the search path. Removing them was worth ~1.6x on HNSW search.
 *
 * The constructor argument is now the ID SPACE (number of vertices), not a
 * bucket-count hint -- ids are used to index directly, so it must cover every
 * id that will be passed to get()/set().
 */
class HashBasedBooleanSet {
   private:
    std::vector<uint16_t, memory::AlignedAllocator<uint16_t>> stamp_;
    uint16_t cur_ = 0;

   public:
    HashBasedBooleanSet() = default;
    ~HashBasedBooleanSet() = default;

    HashBasedBooleanSet(const HashBasedBooleanSet&) = default;
    HashBasedBooleanSet& operator=(const HashBasedBooleanSet&) = default;
    HashBasedBooleanSet(HashBasedBooleanSet&&) noexcept = default;
    HashBasedBooleanSet& operator=(HashBasedBooleanSet&&) noexcept = default;

    explicit HashBasedBooleanSet(size_t num_elements) { initialize(num_elements); }

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

    // get if data_id is in the hashset
    [[nodiscard]] bool get(PID data_id) const { return stamp_[data_id] == cur_; }

    void set(PID data_id) { stamp_[data_id] = cur_; }
};
}  // namespace rabitqlib
