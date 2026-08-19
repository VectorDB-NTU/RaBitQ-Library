#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/memory.hpp"

namespace rabitqlib {
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

    explicit EpochBasedVisitedSet(size_t num_elements, size_t = 0) {
        initialize(num_elements);
    }

    void initialize(size_t num_elements, size_t = 0) {
        stamp_.assign(num_elements, 0);
        cur_ = 0;
        clear();
    }

    void clear() {
        if (++cur_ == 0) {
            std::fill(stamp_.begin(), stamp_.end(), 0);
            cur_ = 1;
        }
    }

    [[nodiscard]] bool get(PID data_id) const { return stamp_[data_id] == cur_; }

    void set(PID data_id) { stamp_[data_id] = cur_; }
};
}  // namespace rabitqlib
