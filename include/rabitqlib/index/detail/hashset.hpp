// This compatibility header preserves the original visited-set API.
#pragma once

#include "rabitqlib/index/detail/visited_set_hash.hpp"

namespace rabitqlib {
using HashBasedBooleanSet = HashBasedVisitedSet;
}  // namespace rabitqlib
