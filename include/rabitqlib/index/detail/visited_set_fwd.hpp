#pragma once

namespace rabitqlib {
// The library names its visited-set backend only here. HashBasedVisitedSet keeps
// memory sublinear in the ID space. EpochBasedVisitedSet is kept as the drop-in
// alternative: it clears in amortized O(1) and never allocates during search, at
// 2 bytes per ID. Both satisfy detail::CheckVisitedSet, and tests keep them
// behaviorally equivalent so this alias can switch backends.
class HashBasedVisitedSet;
using VisitedSet = HashBasedVisitedSet;
}  // namespace rabitqlib
