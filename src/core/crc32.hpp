#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::detail {

// CRC-32 as zlib.crc32 computes it; pass an earlier result as `crc` to continue it.
// Byte-wise: meant for headers and metadata, not bulk data.
uint32_t crc32(const void* data, size_t size, uint32_t crc = 0);

}  // namespace rabitqlib::detail
