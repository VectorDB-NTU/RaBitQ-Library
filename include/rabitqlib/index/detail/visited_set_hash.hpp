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
#include <cassert>
#include <cstddef>
#include <limits>
#include <unordered_set>
#include <vector>

#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/memory.hpp"

namespace rabitqlib {
class HashBasedVisitedSet {
   public:
    static constexpr size_t kNoSizeHint = std::numeric_limits<size_t>::max();

   private:
    PID mask_ = 0;
    std::vector<PID, memory::AlignedAllocator<PID>> table_;
    std::unordered_set<PID> stl_hash_;

    [[nodiscard]] auto hash1(const PID value) const { return value & mask_; }

    // The spill-over lookups are compiled once in src/index/visited_set_hash.cpp, so
    // ISA-specific search sources never instantiate the hash table themselves.
    [[nodiscard]] bool spilled(PID internal_id) const;
    void spill(PID internal_id);

    void initialize_table(const size_t table_size) {
        // Allocate before changing the mask so failure preserves the previous table.
        table_ = std::vector<PID, memory::AlignedAllocator<PID>>(table_size);
        mask_ = static_cast<PID>(table_size - 1);
        assert(hash1(static_cast<PID>(table_size)) == 0);

        std::fill(table_.begin(), table_.end(), kPidMax);
        stl_hash_.clear();
    }

   public:
    HashBasedVisitedSet() = default;
    ~HashBasedVisitedSet() = default;

    HashBasedVisitedSet(const HashBasedVisitedSet&) = default;
    HashBasedVisitedSet& operator=(const HashBasedVisitedSet& other) {
        if (this != &other) {
            // Both containers may allocate; publish only a complete replacement.
            *this = HashBasedVisitedSet(other);
        }
        return *this;
    }
    HashBasedVisitedSet(HashBasedVisitedSet&&) noexcept = default;
    HashBasedVisitedSet& operator=(HashBasedVisitedSet&&) noexcept = default;

    explicit HashBasedVisitedSet(size_t num_elements, size_t size_hint = kNoSizeHint) {
        initialize(num_elements, size_hint);
    }

    void initialize(size_t num_elements, size_t size_hint = kNoSizeHint) {
        size_t size = std::min(num_elements, size_hint);
        size_t bit_size = 0;
        size_t bit = size;
        while (bit != 0) {
            ++bit_size;
            bit >>= 1;
        }
        // Limit per-visit spillover allocations while keeping the direct table sublinear.
        size_t bucket_size = static_cast<size_t>(0x1) << ((bit_size + 4) / 2 + 4);
        initialize_table(bucket_size);
    }

    void clear() {
        std::fill(table_.begin(), table_.end(), kPidMax);
        stl_hash_.clear();
    }

    [[nodiscard]] bool initialized() const noexcept { return !table_.empty(); }

    [[nodiscard]] bool get(PID internal_id) const {
        PID val = table_[hash1(internal_id)];
        if (val == internal_id) {
            return true;
        }
        return val != kPidMax && spilled(internal_id);
    }

    void set(PID internal_id) {
        PID& val = table_[hash1(internal_id)];
        if (val == internal_id) {
            return;
        }
        if (val == kPidMax) {
            val = internal_id;
        } else {
            spill(internal_id);
        }
    }
};
}  // namespace rabitqlib
