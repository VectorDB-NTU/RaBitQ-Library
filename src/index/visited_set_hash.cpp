#include "rabitqlib/index/detail/visited_set_hash.hpp"

#include "rabitqlib/index/types.hpp"

namespace rabitqlib {

bool HashBasedVisitedSet::spilled(PID internal_id) const {
    return stl_hash_.find(internal_id) != stl_hash_.end();
}

void HashBasedVisitedSet::spill(PID internal_id) { stl_hash_.emplace(internal_id); }

}  // namespace rabitqlib
