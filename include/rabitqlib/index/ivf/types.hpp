#pragma once

#include <cstdint>

namespace rabitqlib::ivf {
enum class InitializerType : uint32_t { Auto = 0, Flat = 1, FlatRaBitQ = 2, HNSW = 3 };
}  // namespace rabitqlib::ivf
