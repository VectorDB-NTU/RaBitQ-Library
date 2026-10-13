#include <cstdint>
#include <type_traits>

#include "rabitqlib/index/types.hpp"

static_assert(std::is_same_v<rabitqlib::PID, uint32_t>);
static_assert(rabitqlib::kPidMax == UINT32_MAX);
static_assert(std::is_trivially_copyable_v<rabitqlib::AnnCandidate<float>>);
